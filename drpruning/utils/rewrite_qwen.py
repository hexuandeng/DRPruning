from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2RotaryEmbedding, logger
from torch import nn
import transformers

def rewrite_qwen(HEAD_DIM):
    def _custom_init(self, config, layer_idx=None):
            super(Qwen2Attention, self).__init__()
            self.config = config
            self.layer_idx = layer_idx
            if layer_idx is None:
                logger.warning_once(
                    f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                    "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                    "when creating this class."
                )

            self.hidden_size = config.hidden_size
            self.num_heads = config.num_attention_heads
            self.head_dim = HEAD_DIM
            self.num_key_value_heads = config.num_key_value_heads
            self.num_key_value_groups = self.num_heads // self.num_key_value_heads
            self.max_position_embeddings = config.max_position_embeddings
            self.rope_theta = config.rope_theta
            self.is_causal = True
            self.attention_dropout = config.attention_dropout

            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
            self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
            self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
            print(self.q_proj)

            self.rotary_emb = Qwen2RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
            self.hidden_size = self.num_heads * self.head_dim
    transformers.models.qwen2.modeling_qwen2.Qwen2Attention.__init__ = _custom_init

original_init = transformers.Qwen2ForCausalLM.__init__
def _Qwen2_init(self, config):
    if hasattr(config, "head_dim"):
        print("Use Custom! Head Dim = ", config.head_dim)
        rewrite_qwen(config.head_dim)
    original_init(self, config)
transformers.Qwen2ForCausalLM.__init__ = _Qwen2_init
