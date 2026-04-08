"""
Experimental models for algorithmic research.
Implements:
1. Ouroboros-VCoT (Recursive blocks with LoRA modulation)
2. Attention Residuals (Learnable depth-wise aggregation)
3. Latent Subspace Partitioning (Logic vs Probabilistic zones)
4. Dependency Tensor (Emergent parallelism)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
from torch import nn

@dataclass
class ExperimentalConfig:
    model_dim: int
    num_heads: int
    seq_len: int
    num_layers: int
    vocab_size: int
    expanded_model_dim: int
    multiple_of: int = 256
    rmsnorm_epsilon: float = 1e-6
    qknorm_epsilon: float = 1e-6
    use_residual_scaling: bool = True
    tie_embeddings: bool = True
    
    # Experimental Flags
    recursion_steps: int = 1  
    use_attn_residuals: bool = False  
    use_latent_partitioning: bool = False  
    use_ouroboros: bool = False  
    use_dependency_tensor: bool = False  # Ativa a matriz de dependência 0/1
    lora_rank: int = 32
    probabilistic_dim: int = 128 
    internal_attn_dim: Optional[int] = None # Para testes de interpretabilidade

class OuroborosController(nn.Module):
    """Plan 1: Controller that generates per-step modulation."""
    def __init__(self, dim: int, lora_rank: int):
        super().__init__()
        self.controller = nn.Sequential(
            nn.Linear(dim, lora_rank),
            nn.SiLU(),
            nn.Linear(lora_rank, lora_rank)
        )
    
    def forward(self, x):
        return self.controller(x.mean(dim=1))

class BayesianProbabilisticGate(nn.Module):
    """Plan 3: Subspace that uses variational inference for probability."""
    def __init__(self, dim: int):
        super().__init__()
        self.mu_gen = nn.Linear(dim, dim)
        self.logvar_gen = nn.Linear(dim, dim)
    
    def forward(self, x):
        mu = self.mu_gen(x)
        logvar = self.logvar_gen(x)
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256):
        super().__init__()
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.fc1 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)
        self.glu = nn.GLU(dim=-1)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)

    def forward(self, x):
        return self.fc2(self.glu(self.fc1(x)))

class ExperimentalAttention(nn.Module):
    def __init__(self, cfg: ExperimentalConfig):
        super().__init__()
        self.dim = cfg.model_dim
        self.n_heads = cfg.num_heads
        # Se internal_attn_dim for definido, reduzimos a dimensão interna
        self.internal_dim = cfg.internal_attn_dim if cfg.internal_attn_dim else cfg.model_dim
        self.head_dim = self.internal_dim // cfg.num_heads

        self.w_qkv = nn.Linear(cfg.model_dim, 3 * self.internal_dim, bias=False)
        self.w_out = nn.Linear(self.internal_dim, cfg.model_dim, bias=False)
        
        if cfg.use_ouroboros:
            self.lora_a = nn.Parameter(torch.randn(3 * self.internal_dim, cfg.lora_rank) * 0.01)
            self.lora_b = nn.Parameter(torch.randn(cfg.lora_rank, cfg.model_dim) * 0.01)

        self.eps = cfg.qknorm_epsilon
        self.attn_scale = nn.Parameter(torch.tensor(math.log2(cfg.seq_len**2 - cfg.seq_len)))

    def forward(self, x, freqs_cis, modulation: Optional[torch.Tensor] = None):
        bsz, seqlen, _ = x.shape
        
        if modulation is not None:
            lora_x = x @ self.lora_b.T 
            lora_x = lora_x * modulation.unsqueeze(1) 
            delta_qkv = lora_x @ self.lora_a.T 
            qkv = self.w_qkv(x) + delta_qkv
        else:
            qkv = self.w_qkv(x)

        q, k, v = qkv.split(self.internal_dim, dim=2)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_heads, self.head_dim)

        from .models import apply_rotary_emb_complex_like
        q, k = apply_rotary_emb_complex_like(q, k, freqs_cis=freqs_cis)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = q / (torch.norm(q, dim=-1, keepdim=True) + self.eps)
        k = k / (torch.norm(k, dim=-1, keepdim=True) + self.eps)
        q *= self.attn_scale

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0)
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, self.internal_dim)
        return self.w_out(out)

class ExperimentalBlock(nn.Module):
    def __init__(self, layer_id: int, cfg: ExperimentalConfig):
        super().__init__()
        self.layer_id = layer_id
        self.cfg = cfg
        self.attn = ExperimentalAttention(cfg)
        self.attn_norm = nn.RMSNorm(cfg.model_dim, eps=cfg.rmsnorm_epsilon)
        self.mlp = MLP(cfg.model_dim, cfg.expanded_model_dim, cfg.multiple_of)
        self.mlp_norm = nn.RMSNorm(cfg.model_dim, eps=cfg.rmsnorm_epsilon)
        
        if cfg.use_ouroboros:
            self.controller = OuroborosController(cfg.model_dim, cfg.lora_rank)
            self.recursion_gate = nn.Parameter(torch.ones(1) * 2.0) 
            
        if cfg.use_latent_partitioning:
            self.prob_gate = BayesianProbabilisticGate(cfg.probabilistic_dim)
            
    def forward(self, x, freqs_cis, dependency_input: Optional[torch.Tensor] = None):
        h = x if dependency_input is None else (x + dependency_input)
        
        for _ in range(self.cfg.recursion_steps):
            modulation = self.controller(h) if self.cfg.use_ouroboros else None
            
            if self.cfg.use_latent_partitioning:
                logic_zone = h[:, :, :-self.cfg.probabilistic_dim]
                prob_zone = h[:, :, -self.cfg.probabilistic_dim:]
                prob_zone = self.prob_gate(prob_zone)
                h = torch.cat([logic_zone, prob_zone], dim=-1)

            h = h + self.attn(self.attn_norm(h), freqs_cis, modulation)
            h = h + self.mlp(self.mlp_norm(h))
            
            if self.cfg.use_ouroboros:
                gate = torch.sigmoid(self.recursion_gate)
                h = gate * x + (1 - gate) * h
        
        return h

class ExperimentalTransformer(nn.Module):
    def __init__(self, cfg: ExperimentalConfig):
        super().__init__()
        self.cfg = cfg
        self.n_layers = cfg.num_layers
        head_dim = (cfg.internal_attn_dim if cfg.internal_attn_dim else cfg.model_dim) // cfg.num_heads
        
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.model_dim)
        self.layers = nn.ModuleList([ExperimentalBlock(i, cfg) for i in range(cfg.num_layers)])
        self.out_norm = nn.RMSNorm(cfg.model_dim, eps=cfg.rmsnorm_epsilon)
        self.lm_head = nn.Linear(cfg.model_dim, cfg.vocab_size, bias=False)
        
        if cfg.use_dependency_tensor:
            dep_matrix = torch.tril(torch.ones(cfg.num_layers, cfg.num_layers), diagonal=-1)
            self.dependency_matrix = nn.Parameter(dep_matrix)
        
        from .models import precompute_freqs_cis
        self.register_buffer(
            'freqs_cis',
            precompute_freqs_cis(head_dim, cfg.seq_len, 500000)[0 : cfg.seq_len],
            persistent=False,
        )

        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(self, x, targets=None):
        bsz, seqlen = x.shape
        h = self.embed_tokens(x)
        freqs_cis = self.freqs_cis[:seqlen].to(h.device)
        
        layer_outputs = [h]
        for i, layer in enumerate(self.layers):
            dep_input = None
            if self.cfg.use_dependency_tensor:
                weights = self.dependency_matrix[i, :len(layer_outputs)]
                valid_weights = weights.view(-1, 1, 1)
                dep_input = sum(layer_outputs[j] * valid_weights[j] for j in range(len(layer_outputs)))
            
            h = layer(h, freqs_cis, dependency_input=dep_input)
            layer_outputs.append(h)
            
        logits = self.lm_head(self.out_norm(h))
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
            return logits, loss
        return logits
