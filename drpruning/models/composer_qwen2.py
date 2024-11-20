import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from composer.metrics import METRIC_DEFAULT_CTORS
from composer.metrics.nlp import LanguageCrossEntropy, LanguagePerplexity
from composer.models.base import ComposerModel
from einops import rearrange
from omegaconf import DictConfig
from torch.nn import functional as F
from transformers.pytorch_utils import (find_pruneable_heads_and_indices,
                                        prune_linear_layer)

from drpruning.models.l0_module3 import L0Module
from drpruning.models.metrics import DomainCount, DomainLanguageCrossEntropy
from drpruning.models.composer_llama import LlamaRMSNorm, LlamaMLP, LlamaRotaryEmbedding, normal_attn_fn, flash_attn_fn, apply_rotary_pos_emb, prepare_decoder_attention_mask


class ComposerMosaicQwen2(ComposerModel):
    """ Llama model with the Composer model interface. """
    def __init__(self, cfg):
        super().__init__()
        self.model = Qwen2Model(cfg)
        self.ref_model = None
        self.num_fwd_flops = self._compute_num_fwd_flops()
        self.train_metrics = {
            'LanguageCrossEntropy': LanguageCrossEntropy(),
            'Perplexity': LanguagePerplexity(),
        }
        self.eval_metrics = {
            'LanguageCrossEntropy': LanguageCrossEntropy(),
            'Perplexity': LanguagePerplexity(),
        }

        self.set_names = getattr(cfg, "set_names", None)
        if self.set_names is not None:
            self.set_name_to_id = {set_name: i for i, set_name in enumerate(self.set_names)}
            self.set_id_to_name = {i: set_name for i, set_name in enumerate(self.set_names)}
        
            for set_name in self.set_names:
                # add train and eval metrics for each set
                self.train_metrics[f'{set_name}_LanguageCrossEntropy'] = DomainLanguageCrossEntropy(set_name=set_name)
                self.eval_metrics[f'{set_name}_LanguageCrossEntropy'] = DomainLanguageCrossEntropy(set_name=set_name)
                self.train_metrics[f'{set_name}_count'] = DomainCount(set_name=set_name, set_index=self.set_name_to_id[set_name]) 

    def prune_params(self, zs=None):
        self.model.prune_params(zs)
        
    def get_targets(self, batch):
        targets = torch.roll(batch['labels'], shifts=-1)
        targets[:, -1] = -100
        return targets
        
    def forward(self, batch):
        input_ids = batch['input_ids']
        key_padding_mask = batch['attention_mask'].bool(
        ) if 'attention_mask' in batch else None
        pruned_steps = batch.get('pruned_steps', None)
        if pruned_steps is not None:
            pruned_steps = pruned_steps[0].item()
        zs = {key: batch[key] for key in batch if "_z" in key}

        model_output = self.model(input_ids=input_ids, key_padding_mask=key_padding_mask, pruned_steps=pruned_steps, **zs)
        return model_output

    def eval_forward(self, batch, outputs=None):
        return outputs if outputs is not None else self.forward(batch)

    def loss(self, outputs, batch):
        logits = outputs["logits"]
        l0_output = outputs["l0_output"]
        targets = self.get_targets(batch)

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1),
                                   ignore_index=-100)
        return_loss = {"ce_loss": loss}
        if l0_output is not None:
            lag_loss = l0_output[0]
            return_loss["lag_loss"] = lag_loss
        return_loss["total"] = sum(return_loss.values())

        # reference to compute_loss in ChiSquareResampleLabelSmoothedCrossEntropyCriterion
        with torch.no_grad():
            mask = (targets != -100).float()
            ind_loss = F.cross_entropy(logits.movedim(-1, 1),
                                    targets,
                                    ignore_index=-100,
                                    reduce=False)
            ind_loss = (ind_loss * mask).sum(1)
            zero_vec = torch.zeros(len(self.set_names), device='cuda')  # G
            return_loss["group_losses"] = zero_vec.scatter_add(0, batch['set'], ind_loss)
            return_loss["group_counts"] = zero_vec.scatter_add(0, batch['set'], mask.sum(1))

        return return_loss

    def get_metrics(self, is_train=False):
        return self.train_metrics if is_train else self.eval_metrics

    def update_metric(self, batch, outputs, metric) -> None:
        logits = outputs["logits"]
        if isinstance(metric, DomainLanguageCrossEntropy):
            targets = self.get_targets(batch)
            set_id = self.set_name_to_id[metric.set_name]
            targets[batch["set"] != set_id] = -100
            metric.update(logits, targets)
        elif isinstance(metric, DomainCount):
            with torch.inference_mode():
                idx = None
                selected_sets = batch['set']
            metric.update(selected_sets, idx)
        else:
            logits = logits.view(-1, logits.size(-1))
            targets = self.get_targets(batch).view(-1)
            metric.update(logits, targets)

    def add_eval_metrics(self, evaluator):
        evaluator_metrics = {
            m: METRIC_DEFAULT_CTORS[m]() for m in evaluator.metric_names
        }
        if self.eval_metrics is not None:
            self.eval_metrics.update(evaluator_metrics)
        else:
            self.eval_metrics = evaluator_metrics

    def _compute_num_fwd_flops(self):
        # Might not be correct for LLaMA structures
        n_params = sum(p.numel() for p in self.parameters())
        # the number of paramters is approximately the number of multiply-accumulates (MAC) in the network
        # each MAC has 2 FLOPs - we multiply by 2 ie 2 * n_param
        # this gets us FLOPs / token
        params_flops_per_token = 2 * n_params
        params_flops_per_seq = params_flops_per_token * self.model.cfg.max_seq_len
        # there are 2 FLOPS per mac; there is A=Q*K^T and out=A*V ops (ie mult by 2)
        attn_flops_per_seq = self.model.cfg.n_layers * 2 * 2 * (
            self.model.cfg.d_model * (self.model.cfg.max_seq_len**2))
        return params_flops_per_seq + attn_flops_per_seq

    def flops_per_batch(self, batch):
        # Note: this computation does not take into account padding, and assumes
        # that the dataset has been constructed without padding. Additionally, we
        # assume the backward pass is approximately 2x the forward pass
        return self.num_fwd_flops * 3 * batch['input_ids'].shape[0]

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> nn.Embedding:
        if new_num_tokens is not None:
            self.model._resize_token_embeddings(new_num_tokens)
    
    
class Qwen2Model(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        print(f'Tried to build Llama model with cfg.name={cfg.name}')
        self.cfg = cfg
        
        ### added ###
        self.l0_module = None
        if getattr(self.cfg, "l0_module", None) is not None:
            self.l0_module = L0Module(self.cfg, device=cfg.init_device)
        #############

        layernorm_class = LlamaRMSNorm # TODO: CoFiLayerNorm,RMSLayerNorm
        self.attn_impl = cfg.attn_impl

        self.embedding_fraction = cfg.get('embedding_fraction', 1)
        assert 0 < self.embedding_fraction <= 1, 'model.embedding_fraction must be between 0 (exclusive) and 1 (inclusive)!'

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(cfg.vocab_size, 
                                cfg.d_model,
                                device=cfg.init_device),
        })
        self.transformer.update({
            'blocks':
                nn.ModuleList([
                    Qwen2Block(cfg, device=cfg.init_device)
                    for _ in range(cfg.n_layers)
                ])
        })
        self.transformer.update({
            "ln_f": layernorm_class(cfg.d_model, cfg.get("rms_norm_eps", 1e-6), device=cfg.init_device),
        })
        self.transformer.update({
            "output": nn.Linear(cfg.d_model, cfg.vocab_size, device=cfg.init_device, bias=False),
        })
        
        self.is_causal = True 
        
        # define attn mask
        self._attn_bias_initialized = False
        self.attn_bias = None
        self.attn_bias_shape = None

        if cfg.get('verbose') and cfg.get('verbose') > 2:
            print(self)

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.transformer.wte
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens, nn.Embedding)
        self.transformer.wte = new_embeddings

        old_lm_head = self.transformer.output
        new_lm_head = self._get_resized_embeddings(old_lm_head, new_num_tokens, nn.Linear)
        self.transformer.output = new_lm_head

        self.cfg.vocab_size = new_num_tokens
    
    def _get_resized_embeddings(
        self, old_embeddings: nn.Embedding, new_num_tokens: Optional[int] = None, new_type=nn.Embedding
    ) -> nn.Embedding:
        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

        if old_num_tokens == new_num_tokens:
            return 

        # Build new embeddings
        if new_type == nn.Embedding:
            new_embeddings = new_type(new_num_tokens, old_embedding_dim)
        else:
            new_embeddings = new_type(old_embedding_dim, new_num_tokens, bias=False)
        new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)

        # numbers of tokens to copy
        n = min(old_num_tokens, new_num_tokens)

        new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]
        input_embeddings_avg = old_embeddings.weight.mean(dim=0, keepdim=True)
        new_embeddings.weight.data[n:] = input_embeddings_avg

        return new_embeddings
        
        
    def prune_params(self, zs=None):
        if zs is None:
            self.l0_module.eval()
            zs = self.l0_module(calculate_lagrangian=False)
        # wte as well :) 
        # ln_f if hidden states are to be pruned
        if "hidden_z" in zs:
            hidden_z = zs["hidden_z"]
            remaining_index = torch.where(~hidden_z.eq(0))[0]
            self.transformer.ln_f.prune_params(hidden_z)
            self.transformer.wte.weight.data = self.transformer.wte.weight.data.mul(hidden_z)
            self.transformer.wte.weight = torch.nn.parameter.Parameter(
                self.transformer.wte.weight.index_select(1, remaining_index).clone())
            self.transformer.wte.embedding_dim = len(remaining_index)
            # This is def a bug in llama, but does not incur too much issue
            self.transformer.output.weight.data = self.transformer.output.weight.data.mul(hidden_z) 
            half = self.transformer.output.weight.data.dtype == torch.float16
            self.transformer.output = prune_linear_layer(self.transformer.output, remaining_index, dim=1)
            if half:
                self.transformer.output = self.transformer.output.half()
            
        for i, block in enumerate(self.transformer.blocks):
            zs_block = self.get_zs_block(zs, i)
            block.prune_params(zs_block)
        
    def get_zs_block(self, zs, block_idx):
        zs_block = {}
        if zs is not None:
            for key in zs:
                if key == "hidden_z": zs_block["hidden_z"] = zs["hidden_z"]
                else: zs_block[key] = zs[key][block_idx] 
        return zs_block

    def forward(
            self,
            input_ids: torch.LongTensor,
            key_padding_mask: Optional[torch.ByteTensor] = None,
            past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
            pruned_steps: int = 0,
            retain_grad: bool = False,
            **zs,):
        S = input_ids.size(1)
        assert S <= self.cfg.max_seq_len, f"Sequence length ({S}) exceeds model maximum sequence length ({self.cfg.max_seq_len})!"

        tok_emb = self.transformer.wte(input_ids)
        if "hidden_z" in zs:
            tok_emb = tok_emb.mul(zs["hidden_z"])
        
        x = tok_emb 
        
        attn_bias = None # only consider the flash attention case
        attention_mask = prepare_decoder_attention_mask((tok_emb.size(0), tok_emb.size(1)), tok_emb)
        
        l0_output = None
        if self.l0_module is not None:
            assert zs == {}, "zs should be empty when using L0Module"
            zs = self.l0_module(calculate_lagrangian=False, pruned_steps=pruned_steps)
            
        for b_idx, block in enumerate(self.transformer.blocks):
            zs_block = self.get_zs_block(zs, b_idx)
            past_key_value = past_key_values[
                b_idx] if past_key_values is not None else None

            x, past_key_value = block(
                x,
                past_key_value=past_key_value,
                attn_bias=attn_bias,
                key_padding_mask=key_padding_mask,
                is_causal=self.is_causal,
                attention_mask=attention_mask,
                retain_grad=retain_grad,
                **zs_block 
            )

            if past_key_values is not None:
                past_key_values[b_idx] = past_key_value

        x = self.transformer.ln_f(x, hidden_z=zs.get("hidden_z", None))
        logits = self.transformer.output(x)

        if self.l0_module is not None:
            l0_output = self.l0_module(calculate_lagrangian=True, pruned_steps=pruned_steps)

        return {"logits": logits, "l0_output": l0_output, "zs": zs}
        
    # Param Initialization, needed for device='meta' fast initialization
    def param_init_fn(self, module):
        pass
        
    def fsdp_wrap_fn(self, module):
        return isinstance(module, Qwen2Block)

    # Activation Checkpointing
    def activation_checkpointing_fn(self, module):
        return isinstance(module, Qwen2Block)

class Qwen2Block(nn.Module):
    def __init__(self, cfg: DictConfig, device: Optional[str] = None):
        super().__init__()

        layernorm_class = LlamaRMSNorm # TODO: CoFiLayerNorm,RMSLayerNorm
        
        self.ln_1 = layernorm_class(cfg.d_model, cfg.get("rms_norm_eps", 1e-6), device=device)
        self.attn = Qwen2Attention(cfg, device) 
        self.ln_2 = layernorm_class(cfg.d_model, cfg.get("rms_norm_eps", 1e-6), device=device)
        self.mlp = LlamaMLP(cfg, device)
    
    def prune_params(self, zs_block):
        self.attn.prune_params(zs_block)
        self.mlp.prune_params(zs_block)
        # ln_1, ln_2 later
        
        if self.attn.wq is None:
            self.ln_1 = None
        if self.mlp.gate_proj is None:
            self.ln_2 = None
        if "hidden_z" in zs_block:
            hidden_z = zs_block["hidden_z"]
            if self.ln_1 is not None: self.ln_1.prune_params(hidden_z)
            if self.ln_2 is not None: self.ln_2.prune_params(hidden_z) 
            
        
    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attn_bias: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.ByteTensor] = None,
        is_causal: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
        retain_grad: bool = False,
        head_z: Optional[torch.Tensor] = None,
        head_layer_z: Optional[torch.Tensor] = None,
        intermediate_z: Optional[torch.Tensor] = None,
        mlp_z: Optional[torch.Tensor] = None,
        hidden_z: Optional[torch.Tensor] = None,
        qk_head_dim_z: Optional[torch.Tensor] = None,
        vo_head_dim_z: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
       
        if self.ln_1 is not None:
            a = self.ln_1(x, hidden_z=hidden_z)
            b, _, past_key_value = self.attn(a, 
                                             past_key_value=past_key_value,
                                             attn_bias=attn_bias,
                                             key_padding_mask=key_padding_mask,
                                             is_causal=is_causal,
                                             attention_mask=attention_mask,
                                             retain_grad=retain_grad,
                                             head_z=head_z,
                                             head_layer_z=head_layer_z,
                                             hidden_z=hidden_z,
                                             qk_head_dim_z=qk_head_dim_z,
                                             vo_head_dim_z=vo_head_dim_z)
        else:
            b = 0
        x = x + b
        
        if self.ln_2 is not None:
            m = self.ln_2(x, hidden_z=hidden_z)
            n = self.mlp(m, retain_grad, intermediate_z, mlp_z, hidden_z)
        else:
            n = 0
             
        x = x + n        
        return x, past_key_value 
    
def turn_head_z(head_z, head_layer_z):
    head_z = head_z.squeeze().clone()
    if head_layer_z is not None:
        head_z *= head_layer_z
    to_prune_heads = torch.where(head_z == 0)[0].view(-1).tolist()
    return to_prune_heads

def turn_mlp_z(intermediate_z, mlp_z):
    intermediate_z_layer = intermediate_z.squeeze().clone()
    if mlp_z is not None:
        intermediate_z_layer *= mlp_z
    keep_intermediate_dims = torch.where(intermediate_z_layer != 0)[0].tolist()
    return keep_intermediate_dims 


class Qwen2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, cfg: DictConfig, device: Optional[str] = None):
        super().__init__()
        self.attn_impl = cfg.get('attn_impl')
        
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.get('head_dim', self.d_model // self.n_heads)
        self.n_kv_heads = cfg.get('n_kv_heads', self.n_heads)
        self.n_kv_groups = self.n_heads // self.n_kv_heads
        self.max_position_embeddings = cfg.get('max_pos_embed', 2048)
        self.rope_theta = cfg.get('rope_theta', 10000)
        self.q_pruned_heads = set()
        self.kv_pruned_heads = set()
        
        self.softmax_scale = cfg.get('softmax_scale')
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.d_model / self.n_heads)
        self.attn_dropout_p = cfg.get('attn_pdrop')
        
        self.wq = nn.Linear(self.d_model, self.n_heads * self.head_dim, device=device, bias=True)
        self.wk = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, device=device, bias=True)
        self.wv = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, device=device, bias=True)
        
        self.attn_fn = flash_attn_fn if self.attn_impl == 'flash' else normal_attn_fn

        self.out_proj = nn.Linear(self.n_heads * self.head_dim, self.d_model, device=device, bias=False)
        self.out_proj._is_residual = True  # type: ignore
        
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, self.max_position_embeddings, self.rope_theta)
    
    def prune_params(self, zs_block):
        head_z = None; head_layer_z = None; hidden_z = None; qk_head_dim_z = None; vo_head_dim_z = None
        if "head_z" in zs_block:
            head_z = zs_block["head_z"].squeeze()
        
        if "head_layer_z" in zs_block:
            head_layer_z = zs_block["head_layer_z"].squeeze()
        
        if "hidden_z" in zs_block:
            hidden_z = zs_block["hidden_z"].squeeze()
        
        if "qk_head_dim_z" in zs_block:
            qk_head_dim_z = zs_block["qk_head_dim_z"].squeeze() # qk_head_dim is the same as hidden_z
            vo_head_dim_z = zs_block["vo_head_dim_z"].squeeze() # vo_head_dim is the same as hidden_z
            
            
        # update params #
        if head_z is not None:
            head_z_for_update = torch.repeat_interleave(head_z, self.head_dim)
            self.wv.weight.data = self.wv.weight.data.transpose(0, 1).mul(head_z_for_update).transpose(0, 1)
        if head_layer_z is not None:
            self.out_proj.weight.data = self.out_proj.weight.data.transpose(0, 1).mul(head_layer_z).transpose(0, 1)
        if hidden_z is not None:
            self.out_proj.weight.data = self.out_proj.weight.data.transpose(0, 1).mul(hidden_z).transpose(0, 1)
        if qk_head_dim_z is not None:
            self.wq.weight.data = self.wq.weight.data.transpose(0, 1).mul(qk_head_dim_z).transpose(0, 1)
            self.wv.weight.data = self.wv.weight.data.transpose(0, 1).mul(vo_head_dim_z).transpose(0, 1)
        #################
        
        if hidden_z is not None:
            remaining_index = torch.where(~hidden_z.eq(0))[0]
            print(f"    Head hidden: {len(hidden_z)} -> {len(remaining_index)}") 
            half = next(self.wq.parameters()).dtype == torch.float16
            self.wk = prune_linear_layer(self.wk, remaining_index, dim=1)
            self.wq= prune_linear_layer(self.wq, remaining_index, dim=1)
            self.wv = prune_linear_layer(self.wv, remaining_index, dim=1)
            self.out_proj = prune_linear_layer(self.out_proj, remaining_index)
            if half:
                self.wq.half()
                self.wk.half()
                self.wv.half()
                self.out_proj.half()
         
        kv_prune_heads = turn_head_z(head_z, head_layer_z)
        len_kv_prune_heads = len(kv_prune_heads)
        if len_kv_prune_heads == 0:
            print(f"    Heads: {self.n_heads} -> {self.n_heads}")
            return

        kv_heads, kv_index = find_pruneable_heads_and_indices(
            kv_prune_heads, self.n_kv_heads, self.head_dim, self.kv_pruned_heads
        )
        
        head_z = torch.repeat_interleave(head_z, repeats=self.n_kv_groups)
        to_prune_heads = turn_head_z(head_z, head_layer_z)
        len_to_prune_heads = len(to_prune_heads)
        if len_to_prune_heads == 0:
            print(f"    Heads: {self.n_heads} -> {self.n_heads}")
            return

        q_heads, q_index = find_pruneable_heads_and_indices(
            to_prune_heads, self.n_heads, self.head_dim, self.q_pruned_heads
        )

        # Prune linear layers
        # setting layers to be None if all the heads are pruned
        if len(kv_index) == 0:
            self.wq = None
            self.wk = None
            self.wv = None
            self.out_proj = None
        else:
            half = next(self.wq.parameters()).dtype == torch.float16
            self.wq = prune_linear_layer(self.wq, q_index)
            self.wk = prune_linear_layer(self.wk, kv_index)
            self.wv = prune_linear_layer(self.wv, kv_index)
            self.out_proj = prune_linear_layer(self.out_proj, q_index, dim=1)
            if half:
                self.wq.half()
                self.wk.half()
                self.wv.half()
                self.out_proj.half()

        print(f"    Heads: {self.n_heads} -> {self.n_heads - len(q_heads)}")
        print(f" KV Heads: {self.n_kv_heads} -> {self.n_kv_heads - len(kv_heads)}")

        # Update hyper params and store pruned heads
        self.n_heads = self.n_heads - len(q_heads)
        self.n_kv_heads = self.n_kv_heads - len(kv_heads)
        self.q_pruned_heads = self.q_pruned_heads.union(q_heads)
        self.kv_pruned_heads = self.kv_pruned_heads.union(kv_heads)
            
    def forward(
        self,
        x,
        past_key_value=None,
        attn_bias=None,
        key_padding_mask=None,
        is_causal=True,
        needs_weights=False,
        attention_mask=None,
        retain_grad=False,
        head_z=None,
        head_layer_z=None,
        hidden_z=None,
        qk_head_dim_z=None,
        vo_head_dim_z=None):

        # qkv = self.Wqkv(x)
        # query, key, value = qkv.chunk(3, dim=2)
        if self.wq is None:
            return None, None, past_key_value

        query = self.wq(x)
        key = self.wk(x)
        value = self.wv(x)

        if qk_head_dim_z is not None:
            query = query.mul(qk_head_dim_z)
            value = value.mul(vo_head_dim_z)
        
        query_padding_mask = None
        if key_padding_mask is not None:
            query_padding_mask = key_padding_mask[:, -query.size(1):]
        
        if attn_bias is not None:
            attn_bias = attn_bias[:, :, -query.size(1):, -key.size(1):]

        # b, s, d = query.shape
        query = rearrange(query, 'b s (h d) -> b h s d', h=self.n_heads)
        key = rearrange(key, 'b s (h d) -> b h s d', h=self.n_kv_heads)
        value = rearrange(value, 'b s (h d) -> b h s d', h=self.n_kv_heads)
        
        kv_seq_len = key.size(2)
        offset = 0
        if past_key_value is not None:
            offset = past_key_value[0].shape[-2]
            kv_seq_len += offset
        cos, sin = self.rotary_emb(value, seq_len=kv_seq_len)
        query, key = apply_rotary_pos_emb(query, key, cos, sin, offset=offset)

        offset = 0
        if past_key_value is not None:
            if len(past_key_value) != 0:
                offset = past_key_value[0].shape[-2]
                key = torch.cat([past_key_value[0], key], dim=1)
                value = torch.cat([past_key_value[1], value], dim=1)
                past_key_value = (key, value)

        key = torch.repeat_interleave(key, dim=1, repeats=self.n_kv_groups)
        value = torch.repeat_interleave(value, dim=1, repeats=self.n_kv_groups)
        if head_z is not None:
            head_z = torch.repeat_interleave(head_z, dim=1, repeats=self.n_kv_groups)

        if self.attn_fn == flash_attn_fn:
            query = rearrange(query, 'b h s d -> b s h d')
            key = rearrange(key, 'b h s d -> b s h d')
            value = rearrange(value, 'b h s d -> b s h d')
            context, attn_weights = self.attn_fn(
                query,
                key,
                value,
                softmax_scale=self.softmax_scale,
                attn_bias=attn_bias,
                query_padding_mask=query_padding_mask,
                key_padding_mask=key_padding_mask,
                is_causal=is_causal,
                dropout_p=self.attn_dropout_p,
                training=self.training,
                needs_weights=needs_weights,
                head_z=head_z
            )
        else:
            context = self.attn_fn(
                query=query,
                key=key,
                value=value,
                attention_mask=attention_mask,
                head_z=head_z
            )
            attn_weights = None

        if retain_grad:
            self.context = context
            if self.context.requires_grad:
                self.context.retain_grad()
                
        output = self.out_proj(context)
        
        if head_layer_z is not None:
            output *= head_layer_z
        
        if hidden_z is not None:
            output *= hidden_z
            
        if retain_grad: 
            self.output = output 
            if self.output.requires_grad:
                self.output.retain_grad()
        
        return output, attn_weights, past_key_value
