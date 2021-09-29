# coding=utf-8
# Copyright <authors-yet-to-name> and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" HTransformer1D model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

H_TRANSFORMER_1D_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "<checkpoint-yet-to-create>": "https://huggingface.co/<checkpoint-yet-to-create>/resolve/main/config.json",
    # See all HTransformer1D models at https://huggingface.co/models?filter=h_transformer_1d
}


class HTransformer1dConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.HTransformer1dModel`.
    It is used to instantiate an HTransformer1D model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the HTransformer1D `<checkpoint-yet-to-create> <https://huggingface.co/<checkpoint-yet-to-create>>`__ architecture.

    Configuration objects inherit from  :class:`~transformers.PretrainedConfig` and can be used
    to control the model outputs. Read the documentation from  :class:`~transformers.PretrainedConfig`
    for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the HTransformer1D model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.HTransformer1dModel` or
            :class:`~transformers.TFHTransformer1dModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
            If string, :obj:`"gelu"`, :obj:`"relu"`, :obj:`"selu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.HTransformer1dModel` or
            :class:`~transformers.TFHTransformer1dModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if ``config.is_decoder=True``.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If :obj:`True`, use gradient checkpointing to save memory at the expense of slower backward pass.
        Example::

        >>> from transformers import HTransformer1dModel, HTransformer1dConfig

        >>> # Initializing a HTransformer1D <checkpoint-yet-to-create> style configuration
        >>> configuration = HTransformer1dConfig()

        >>> # Initializing a model from the <checkpoint-yet-to-create> style configuration
        >>> model = HTransformer1dModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "h_transformer_1d"
    

    def __init__(
        self,
        # REQUIRED for lucidrains implementation:
        vocab_size=30522,  # ^ called "num_tokens"
        hidden_size=512,  # ^ called "dim" --- SHOULD BE POWER OF 2 to avoid padding for hierarchical splitting
        num_hidden_layers=12,  # ^ called "depth"
        num_attention_heads=8,  # ^ called "heads"
        is_decoder=False,  # ^ called "causal"
        max_seq_len=8192,  # required for decoder-style causal attention
        # Next one: Probably not required since in Bert it's calculated as int(hidden_size/num_attention_heads):
        # dim_head=64,  # dimensions per head
        block_size=128,  # block size for hierarchical attention
        # attention_probs_dropout_prob=0.1,  # no dropout in attention implemented

        # NOT YET IMPLEMENTED:
        reversible=True,  # reversibility to save memory with increased depth
        shift_tokens=True,  # shift half the feature space by one along the sequence dimension,
                            # for faster convergence (experimental)

        # network configs independent of attention implementation:
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,

        # REQUIRED FOR SPECIFIC TASKS
        type_vocab_size=1,  # set to 2 for token-type-embeddings (= 2-sentence-embedding)
        use_cache=True,
        # depending on your tokenizer you have to configure the token ids
        pad_token_id=1,  # Roberta-Tokenizer:1
        bos_token_id=0,  # Roberta-Tokenizer:0
        eos_token_id=2,  # Roberta-Tokenizer:2

        # for the previous positional embedding implementation before the addition of rotary embeddings
        # if you set this to "absolute", it will add an absolute pos.emb. at the start of the network (unnecessary)
        position_embedding_type="relative_key_query",  # https://arxiv.org/abs/2009.13658
        max_position_embeddings=512,

        **kwargs
    ):
        self.vocab_size = vocab_size
        self.position_embedding_type = position_embedding_type
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.is_decoder = is_decoder
        self.max_seq_len = max_seq_len
        # self.dim_head = dim_head,
        self.block_size = block_size
        self.reversible = reversible
        self.shift_tokens = shift_tokens
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        # self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )

    