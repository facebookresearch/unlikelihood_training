# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from fairseq.models import (
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer_lm import transformer_lm_big


@register_model_architecture('transformer_lm', 'transformer_lm_ul')
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, 'decoder_layers', 16)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.0)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.1)
    args.no_decoder_final_norm = getattr(args, 'no_decoder_final_norm', False)
    args.tie_adaptive_proj = getattr(args, 'tie_adaptive_proj', True)
    transformer_lm_big(args)
