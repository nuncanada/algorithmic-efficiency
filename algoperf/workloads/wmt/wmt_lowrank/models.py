import copy
import math
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import normal_
from torch.nn.init import xavier_uniform_

from algoperf.workloads.wmt.wmt_pytorch.models import (
    PositionalEncoding, 
    MultiheadAttention, 
    TransformerDecoder,
    make_src_mask,
    make_tgt_and_memory_mask,
    shift_right
)

DROPOUT_RATE = 0.1

class LowRankDynamicAdapter(nn.Module):
    def __init__(self, d_model: int, rank: int = 16, activation: Callable = F.silu):
        super().__init__()
        self.rank = rank
        self.activation = activation
        self.W_in_down = nn.Linear(d_model, rank, bias=False)
        self.W_out_down = nn.Linear(d_model, rank, bias=False)
        self.W_up = nn.Linear(rank, d_model, bias=False)
        nn.init.zeros_(self.W_up.weight)
        nn.init.kaiming_uniform_(self.W_in_down.weight, a=1)
        nn.init.kaiming_uniform_(self.W_out_down.weight, a=1)

    def forward(self, h_in: Tensor, x_out: Tensor) -> Tensor:
        z = self.activation(self.W_in_down(h_in) + self.W_out_down(x_out))
        q_depth = self.W_up(z)
        return q_depth

class DepthAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.scale = d_model ** -0.5

    def forward(self, q_depth: Tensor, history: list[Tensor]) -> Tensor:
        K = torch.stack(history, dim=0)
        logits = torch.einsum('bsd,lbsd->lbs', q_depth, K) * self.scale
        weights = F.softmax(logits, dim=0)
        h_res = torch.einsum('lbs,lbsd->bsd', weights, K)
        return h_res

class AttnResEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, rank: int = 16, 
                 dropout_rate: float = 0.1, activation: Callable = F.relu, attention_temp: float = 1.0):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, self_attn=True, attention_temp=attention_temp, bias=False)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation_ffn = activation
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.adapter = LowRankDynamicAdapter(d_model, rank=rank)
        self.depth_attn = DepthAttention(d_model)

    def forward(self, x: Tensor, history: list[Tensor], src_mask: Optional[Tensor] = None, dropout_rate: float = 0.0) -> Tensor:
        h_in = x
        # Attention block - passing norm1(x) as query, key, and value
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=src_mask, dropout_rate=dropout_rate)
        x = x + F.dropout(attn_out, p=dropout_rate, training=self.training)
        
        # FF block
        ff_in = self.norm2(x)
        x_ff = self.activation_ffn(self.linear1(ff_in))
        x_out = self.linear2(F.dropout(x_ff, p=dropout_rate, training=self.training))
        x_out = F.dropout(x_out, p=dropout_rate, training=self.training)
        
        # Dynamic Routing (Your proposal)
        q_depth = self.adapter(h_in, x_out)
        current_candidate = x + x_out
        updated_history = history + [current_candidate]
        h_new = self.depth_attn(q_depth, updated_history)
        return h_new

class AttnResEncoder(nn.Module):
    def __init__(self, d_model, nhead, d_hid, nlayers, activation, layer_norm_eps, attention_temp, rank=16):
        super().__init__()
        self.layers = nn.ModuleList([
            AttnResEncoderLayer(d_model, nhead, d_hid, rank=rank, activation=activation, attention_temp=attention_temp)
            for _ in range(nlayers)
        ])
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, dropout_rate: float = 0.0) -> Tensor:
        x = src
        history = [x]
        for layer in self.layers:
            x = layer(x, history, src_mask=mask, dropout_rate=dropout_rate)
            history.append(x)
        return self.norm(x)

class TransformerLowRank(nn.Module):
    def __init__(self, ntoken: int = 32000, d_model: int = 1024, nhead: int = 16, d_hid: int = 1024, nlayers: int = 6,
                 activation: Callable = F.relu, glu: bool = False, layer_norm_eps: float = 1e-6, 
                 attention_temp: float = 1.0, pre_ln: bool = True, rank: int = 16):
        super().__init__()
        self.nhead = nhead
        self.pos_encoder = PositionalEncoding(d_model)
        self.shared_embedding = nn.Embedding(ntoken, d_model)
        self.encoder_body = AttnResEncoder(d_model, nhead, d_hid, nlayers, activation, layer_norm_eps, attention_temp, rank=rank)
        self.decoder_body = TransformerDecoder(d_model, nhead, d_hid, activation, glu, layer_norm_eps, nlayers, attention_temp, pre_ln)
        self._reset_parameters()

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                xavier_uniform_(module.weight)
                if module.bias is not None:
                    normal_(module.bias, std=1e-6)

    def forward(self, src: Tensor, tgt: Tensor, inputs_positions=None, targets_positions=None, 
                inputs_segmentation=None, targets_segmentation=None, decode=False, 
                max_len=None, cache=None, dropout_rate=DROPOUT_RATE) -> Tensor:
        
        # 1. Encoder path
        src = src.to(torch.int)
        src_mask = make_src_mask(src, inputs_segmentation, self.nhead)
        src_emb = self.shared_embedding(src) * math.sqrt(self.shared_embedding.embedding_dim)
        src_pos = self.pos_encoder(src_emb, inputs_positions, dropout_rate=dropout_rate)
        memory = self.encoder_body(src_pos, mask=src_mask, dropout_rate=dropout_rate)
        
        # 2. Decoder path
        tgt = tgt.to(torch.int)
        tgt_mask, memory_mask = make_tgt_and_memory_mask(
            tgt, src, inputs_segmentation, targets_segmentation, decode, self.nhead
        )
        if not decode:
            tgt = shift_right(tgt)
        
        tgt_emb = self.shared_embedding(tgt)
        tgt_pos = self.pos_encoder(tgt_emb, targets_positions, decode=decode, cache=cache, dropout_rate=dropout_rate)
        
        if decode:
            tgt_pos, cache = tgt_pos
            
        output = self.decoder_body(tgt_pos, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                   decode=decode, max_len=max_len, cache=cache, dropout_rate=dropout_rate)
        
        if decode:
            output, cache = output
            
        normalize = math.sqrt(output.shape[-1])
        output = torch.matmul(output, self.shared_embedding.weight.T) / normalize
        
        if decode:
            return output, cache
        return output
