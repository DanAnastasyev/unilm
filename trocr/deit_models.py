import torch.nn as nn

import torch
import logging
import argparse
import os

from timm.models import create_model
from typing import Optional
from fairseq.models import FairseqEncoder, register_model, FairseqEncoderDecoderModel, register_model_architecture
from fairseq.models.transformer import TransformerEncoder, TransformerDecoder, Embedding, TransformerModel
from fairseq.models.transformer import base_architecture as base_transformer
from fairseq import utils
from argparse import Namespace
from omegaconf import DictConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models.bart import BARTModel


logger = logging.getLogger(__name__)

DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model('DeiT_TR')
class DeiTTRModel(FairseqEncoderDecoderModel):
    def load_state_dict(
        self,
        state_dict,
        strict=True,
        model_cfg: Optional[DictConfig] = None,
        args: Optional[Namespace] = None,
    ):

        if model_cfg is None and args is not None:
            logger.warn("using 'args' is deprecated, please update your code to use dataclass config")
            model_cfg = convert_namespace_to_omegaconf(args).model

        self.upgrade_state_dict(state_dict)

        from fairseq.checkpoint_utils import prune_state_dict

        new_state_dict = prune_state_dict(state_dict, model_cfg)
        if not model_cfg.ape:
            model_seq_len = self.state_dict()['encoder.deit.pos_embed'].shape[1]
            ckpt_seq_len = new_state_dict['encoder.deit.pos_embed'].shape[1]
            logger.info('Load from {:d} seq len to {:d}'.format(ckpt_seq_len, model_seq_len))
            if model_seq_len <= ckpt_seq_len:
                new_state_dict['encoder.deit.pos_embed'] = new_state_dict['encoder.deit.pos_embed'][:, :model_seq_len, :]
            else:
                t = self.state_dict()['encoder.deit.pos_embed']
                t[:, :ckpt_seq_len, :] = new_state_dict['encoder.deit.pos_embed']
                new_state_dict['encoder.deit.pos_embed'] = t

        return super().load_state_dict(new_state_dict, strict=False)

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        parser.add_argument(
            '--deit-arch', type=str,
            help='the arch name for the DeiT encoder'
        )
        parser.add_argument(
            '--ape', action='store_true',
            help='if use absolute_pos_embed'
        )
        parser.set_defaults(ape=False)
        parser.add_argument(
            '--mask-ratio', default=0.0, type=float,
            help='the mask ratio for the encoder output masking.'
        )

    @staticmethod
    def read_args_from_roberta(roberta_args: argparse.Namespace):
        # TODO: this would become easier if encoder/decoder where using a similar
        # TransformerConfig object
        args = argparse.Namespace(**vars(roberta_args))
        attr_map = [
            ("encoder_attention_heads", "decoder_attention_heads"),
            ("encoder_embed_dim", "decoder_embed_dim"),
            ("encoder_embed_dim", "decoder_output_dim"),
            ("encoder_normalize_before", "decoder_normalize_before"),
            ("encoder_layers_to_keep", "decoder_layers_to_keep"),
            ("encoder_ffn_embed_dim", "decoder_ffn_embed_dim"),
            ("encoder_layerdrop", "decoder_layerdrop"),
            ("encoder_layers", "decoder_layers"),
            ("encoder_learned_pos", "decoder_learned_pos"),
            # should this be set from here ?
            ("max_positions", "max_target_positions"),
        ]
        for k1, k2 in attr_map:
            setattr(args, k2, getattr(roberta_args, k1))

        args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
        args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
        args.share_decoder_input_output_embed = not roberta_args.untie_weights_roberta
        return args

    @classmethod
    def build_model(cls, args, task):
        decoder_pretrained = getattr(args, "decoder_pretrained", None)
        bart_encoder, bart_decoder = None, None
        if decoder_pretrained == 'bart':
            bart_dir = os.path.dirname(args.decoder_pretrained_path)
            add_adapter = getattr(args, 'add_adapter', False)
            print(f'Loading BART from {bart_dir} to init encoder-decoder')
            bart_model = BARTModel.from_pretrained(
                bart_dir, os.path.basename(args.decoder_pretrained_path), bpe='sentencepiece',
                sentencepiece_model=os.path.join(bart_dir, 'sentence.bpe.model'),
                add_adapter=add_adapter, strict=not add_adapter
            ).model
            bart_encoder = bart_model.encoder
            bart_decoder = bart_model.decoder

            unfreeze_layers = args.unfreeze_layers or []

            frozen_params = ['embed_tokens', 'output_projection']
            trainable_params = ['norm', 'embed_positions']
            for i in range(bart_model.encoder.num_layers):
                if i not in unfreeze_layers:
                    frozen_params.append(f'encoder.layers.{i}')
            for i in range(bart_model.decoder.num_layers):
                if i not in unfreeze_layers:
                    frozen_params.append(f'decoder.layers.{i}')
            for param_name, param in bart_model.named_parameters():
                is_frozen = False
                if 'adapter' not in param_name:
                    for frozen_param in frozen_params:
                        is_frozen = is_frozen or (frozen_param in param_name)
                    for trainable_param in trainable_params:
                        is_frozen = is_frozen and trainable_param not in param_name

                param.requires_grad = not is_frozen

        encoder = DeiTTREncoder(
            args=args,
            dictionary=task.source_dictionary,
            bart_encoder=bart_encoder
        )

        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        decoder_embed_tokens = cls.build_embedding(
            args, task.target_dictionary, args.decoder_embed_dim, args.decoder_embed_path
        )

        if decoder_pretrained == 'unilm':
            args.decoder_attention_heads = 12

        if decoder_pretrained.startswith('roberta2'):
            logger.info('Using the tengchao version loading roberta.')
            specified = decoder_pretrained.find('-')!=-1

            if specified:
                decoder_pretrained = decoder_pretrained.replace('-', '.')
                logger.info('Load pre-trained decoder parameters from {}'.format(decoder_pretrained))
                roberta = torch.hub.load('pytorch/fairseq:main', decoder_pretrained)
            elif args.decoder_layers == 6:
                logger.info('Load pre-trained decoder parameters from roberta.base')
                roberta = torch.hub.load('pytorch/fairseq:main', 'roberta.base')
            elif args.decoder_layers == 12:
                logger.info('Load pre-trained decoder parameters from roberta.large')
                roberta = torch.hub.load('pytorch/fairseq:main', 'roberta.large')
            else:
                raise AttributeError('Cannot determind the pre-trained model')

            roberta.model.args.encoder_layers = args.decoder_layers
            roberta.model.args.fp16 = args.fp16
            roberta_args = DeiTTRModel.read_args_from_roberta(roberta.model.args)
            roberta_args.encoder_embed_dim = args.encoder_embed_dim

            decoder = TransformerDecoder(
                roberta_args,
                task.target_dictionary,
                decoder_embed_tokens,
                no_encoder_attn=False,
            )

            roberta_layers = roberta.model.encoder.sentence_encoder.layers
            decoder_layers = decoder.layers
            offset = len(roberta_layers) - len(decoder_layers)
            assert offset >= 0

            decoder_dict = roberta.state_dict()
            new_decoder_dict = {}
            for key, val in decoder_dict.items():
                if key.startswith('model.encoder.sentence_encoder.layers.'):
                    layer_num = int(key[len('model.encoder.sentence_encoder.layers.'):].split('.')[0])
                    if layer_num - offset < 0:
                        continue
                    else:
                        new_key = 'model.encoder.sentence_encoder.layers.{}.'.format(
                            str(layer_num - offset)) + '.'.join(
                            key[len('model.encoder.sentence_encoder.layers.'):].split('.')[1:])
                        new_decoder_dict[new_key] = val
                else:
                    new_decoder_dict[key] = val
            decoder_dict = new_decoder_dict

            for k, w in list(decoder_dict.items()):
                if '.lm_head' in k:
                    k_proj = "output_projection." + k[len('model.encoder.lm_head.'):]
                    decoder_dict[k_proj] = w.detach().clone()
                    del decoder_dict[k]

            del decoder_dict['_float_tensor']
            del decoder_dict['output_projection.weight']
            del decoder_dict['output_projection.bias']
            del decoder_dict['output_projection.dense.weight']
            del decoder_dict['output_projection.dense.bias']
            del decoder_dict['output_projection.layer_norm.weight']
            del decoder_dict['output_projection.layer_norm.bias']

            new_decoder_dict = {}
            for key, val in decoder_dict.items():
                if "sentence_encoder" in key:
                    key = key[len('model.encoder.sentence_encoder.'):]
                elif "encoder" in key:
                    key = key[len('model.encoder.'):]
                new_decoder_dict[key] = val

            missing_keys, unexpected_keys = decoder.load_state_dict(
                new_decoder_dict, strict=False
            )

        elif decoder_pretrained.startswith('roberta'):
            decoder = TransformerDecoder(
                args = args,
                dictionary=task.target_dictionary,
                embed_tokens=decoder_embed_tokens,
                no_encoder_attn=False
            )

            specified = decoder_pretrained.find('-') != -1

            if specified:
                decoder_pretrained = decoder_pretrained.replace('-', '.')
                logger.info('Load pre-trained decoder parameters from {}'.format(decoder_pretrained))
                roberta = torch.hub.load('pytorch/fairseq:main', decoder_pretrained)
            elif args.decoder_layers == 6:
                logger.info('Load pre-trained decoder parameters from roberta.base')
                roberta = torch.hub.load('pytorch/fairseq:main', 'roberta.base')
            elif args.decoder_layers == 12:
                logger.info('Load pre-trained decoder parameters from roberta.large')
                roberta = torch.hub.load('pytorch/fairseq:main', 'roberta.large')
            else:
                raise AttributeError('Cannot determind the pre-trained model')

            decoder.embed_tokens.load_state_dict(roberta.model.encoder.sentence_encoder.embed_tokens.state_dict())
            roberta_layers = roberta.model.encoder.sentence_encoder.layers
            decoder_layers = decoder.layers
            offset = len(roberta_layers) - len(decoder_layers)
            assert offset >= 0

            for i in range(len(decoder_layers)):
                roberta_i = i + offset
                decoder_layers[i].self_attn.load_state_dict(roberta_layers[roberta_i].self_attn.state_dict())
                decoder_layers[i].self_attn_layer_norm.load_state_dict(roberta_layers[roberta_i].self_attn_layer_norm.state_dict())

        elif decoder_pretrained == 'bart':
            decoder = bart_decoder

        model = cls(encoder, decoder)
        return model

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    def forward(self, imgs, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(imgs, **kwargs)
        decoder_out = self.decoder(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        return decoder_out


class DeiTTREncoder(FairseqEncoder):
    def __init__(self, args, dictionary, bart_encoder=None):
        super().__init__(dictionary)

        if 'custom_size' in args.deit_arch:
            self.deit = create_model(args.deit_arch, pretrained=True, img_size=args.input_size, ape=args.ape, mask_ratio=args.mask_ratio)
        else:
            self.deit = create_model(args.deit_arch, pretrained=True, ape=args.ape, mask_ratio=args.mask_ratio)

        self.bart_encoder = bart_encoder

        self.fp16 = args.fp16

    def forward(self, imgs):
        if self.fp16:
            imgs = imgs.half()

        x, encoder_embedding = self.deit.forward_features(imgs)  # bs, n + 2, dim
        if self.bart_encoder is not None:
            mask = torch.zeros(*x.shape[:2], dtype=torch.bool, device=x.device)
            x = self.bart_encoder(
                src_tokens=None,
                token_embeddings=x,
                skip_embeddings=True,
                mask=mask
            )['encoder_out'][0]
        else:
            x = x.transpose(0, 1) # n + 2, bs, dim

        encoder_padding_mask = torch.zeros(*x.shape[:2]).transpose(0, 1).to(imgs.device)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def reorder_encoder_out(self, encoder_out, new_order):
          """
          Reorder encoder output according to `new_order`.

          Args:
              encoder_out: output from the ``forward()`` method
              new_order (LongTensor): desired order

          Returns:
              `encoder_out` rearranged according to `new_order`
          """
          _encoder_out = encoder_out['encoder_out'][0]
          _encoder_padding_mask = encoder_out['encoder_padding_mask'][0]
          return {
              "encoder_out": [_encoder_out.index_select(1, new_order)],
                "encoder_padding_mask": [_encoder_padding_mask.index_select(0, new_order)],  # B x T
                "encoder_embedding": [_encoder_padding_mask.index_select(0, new_order)],  # B x T x C
                "encoder_states": [],
                "src_tokens": [],
                "src_lengths": [],
        }


# @register_model_architecture('DeiT_TR', 'DeiT_TR_base')
# def DeiT_TR_base(args):
#     # DeiT Encoder  deit_base_distilled_patch16_224
#     args.deit_arch = getattr(args, "deit_arch", "deit_base_distilled_patch16_224")
#     # Transformer Decoder
#     args.encoder_embed_dim = 768
#     base_transformer(args)

@register_model_architecture('DeiT_TR', 'deit_base_decoder_base')
def deit_base_decoder_base(args):
    # DeiT Encoder  deit_base_distilled_patch16_384
    args.deit_arch = getattr(args, "deit_arch", "deit_base_distilled_patch16_384")
    # Transformer Decoder
    args.encoder_embed_dim = 768
    base_transformer(args)

# @register_model_architecture('DeiT_TR', 'DeiT_TR_large_12layers')
# def DeiT_TR_large_12layers(args):
#     # DeiT Encoder  deit_base_distilled_patch16_384
#     args.deit_arch = getattr(args, "deit_arch", "deit_base_distilled_patch16_384")
#     # Transformer Decoder
#     args.encoder_embed_dim = 768
#     args.decoder_layers = getattr(args, "decoder_layers", 12)
#     base_transformer(args)

@register_model_architecture('DeiT_TR', 'deit_base_decoder_large')
def deit_base_decoder_large(args):
    # DeiT Encoder  deit_base_distilled_patch16_384
    args.deit_arch = getattr(args, "deit_arch", "deit_base_distilled_patch16_384")
    # Transformer Decoder
    args.encoder_embed_dim = 768
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_transformer(args)

@register_model_architecture('DeiT_TR', 'beit_small_bart_base')
def beit_small_bart_base(args):
    args.deit_arch = getattr(args, "deit_arch", "beit_small_patch16_384")
    base_transformer(args)

@register_model_architecture('DeiT_TR', 'beit_base_bart_base')
def beit_base_bart_base(args):
    args.deit_arch = getattr(args, "deit_arch", "beit_base_patch16_384")
    base_transformer(args)

@register_model_architecture('DeiT_TR', 'beit_base_decoder_large')
def beit_base_decoder_large(args):
    # DeiT Encoder  deit_base_distilled_patch16_384
    args.deit_arch = getattr(args, "deit_arch", "beit_base_patch16_384")
    # Transformer Decoder
    args.encoder_embed_dim = 768
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_transformer(args)

@register_model_architecture('DeiT_TR', 'beit_large_decoder_large')
def beit_large_decoder_large(args):
    # DeiT Encoder  deit_base_distilled_patch16_384
    args.deit_arch = getattr(args, "deit_arch", "beit_large_patch16_384")
    # Transformer Decoder
    args.encoder_embed_dim = 1024
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_transformer(args)

@register_model_architecture('DeiT_TR', 'DeiT_TR_LargeR_BEiT_Large')
def beit_large_decoder_large(args):
    # DeiT Encoder  deit_base_distilled_patch16_384
    args.deit_arch = getattr(args, "deit_arch", "beit_large_patch16_384")
    # Transformer Decoder
    args.encoder_embed_dim = 1024
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_transformer(args)

@register_model_architecture('DeiT_TR', 'deit_base_decoder_large_custom_size')
def deit_base_decoder_large_custom_size(args):
    # DeiT Encoder  deit_base_distilled_patch16_custom_size
    args.deit_arch = getattr(args, "deit_arch", "deit_base_distilled_patch16_custom_size")
    # Transformer Decoder
    args.encoder_embed_dim = 768
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_transformer(args)


if __name__ == '__main__':
    pass
