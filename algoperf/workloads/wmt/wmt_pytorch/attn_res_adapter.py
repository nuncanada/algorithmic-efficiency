import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union, Callable
from torch import Tensor

class LowRankDynamicAdapter(nn.Module):
    """
    Implementação do Adaptador Dinâmico de Baixo Posto inspirado em LoRA.
    Comprime a entrada inicial e a saída da FFN para gerar uma query de profundidade.
    """
    def __init__(self, d_model: int, rank: int = 16, activation: Callable = F.silu):
        super().__init__()
        self.rank = rank
        self.activation = activation
        
        # Matrizes de projeção descendente (Down-projections)
        self.W_in_down = nn.Linear(d_model, rank, bias=False)
        self.W_out_down = nn.Linear(d_model, rank, bias=False)
        
        # Matriz de projeção ascendente (Up-projection) para o espaço latente
        self.W_up = nn.Linear(rank, d_model, bias=False)
        
        # Inicialização estilo LoRA: W_up em zero para começar como identidade residual
        nn.init.zeros_(self.W_up.weight)
        nn.init.kaiming_uniform_(self.W_in_down.weight, a=1)
        nn.init.kaiming_uniform_(self.W_out_down.weight, a=1)

    def forward(self, h_in: Tensor, x_out: Tensor) -> Tensor:
        # h_in: [batch, seq, d_model] - Entrada do bloco
        # x_out: [batch, seq, d_model] - Saída da FFN
        
        # Projeção para o subespaço de baixo posto (Gargalo)
        z = self.activation(self.W_in_down(h_in) + self.W_out_down(x_out))
        
        # Projeção de volta para a dimensão do modelo (Query de Profundidade)
        q_depth = self.W_up(z)
        return q_depth

class DepthAttention(nn.Module):
    """
    Calcula a atenção softmax sobre o histórico de camadas anteriores.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.scale = d_model ** -0.5
        # Opcional: Projeção de chaves para profundidade se quisermos mais parâmetros
        # self.k_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q_depth: Tensor, history: List[Tensor]) -> Tensor:
        # q_depth: [batch, seq, d_model]
        # history: List of [batch, seq, d_model] tensors (L items)
        
        # Stack history: [L, batch, seq, d_model]
        K = torch.stack(history, dim=0)
        
        # Cálculo de afinidade: [L, batch, seq]
        # (batch, seq, d_model) x (L, batch, seq, d_model) -> (L, batch, seq)
        # Usamos einsum para clareza em tensores 4D
        logits = torch.einsum('bsd,lbsd->lbs', q_depth, K) * self.scale
        
        # Softmax sobre a dimensão da profundidade (L)
        weights = F.softmax(logits, dim=0) # [L, batch, seq]
        
        # Soma ponderada do histórico: [batch, seq, d_model]
        h_res = torch.einsum('lbs,lbsd->bsd', weights, K)
        
        return h_res

class AttnResEncoderLayer(nn.Module):
    """
    Camada Transformer refatorada com AttnRes e Adaptador Dinâmico.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, rank: int = 16, 
                 dropout_rate: float = 0.1, activation: Callable = F.relu):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        # FFN Clássica
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation_ffn = activation
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Componentes Dinâmicos (Sua Proposta)
        self.adapter = LowRankDynamicAdapter(d_model, rank=rank)
        self.depth_attn = DepthAttention(d_model)

    def forward(self, x: Tensor, history: List[Tensor], src_mask: Optional[Tensor] = None) -> Tensor:
        h_in = x # Preserva a entrada inicial para o adaptador
        
        # 1. Self-Attention Block
        attn_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=src_mask)
        x = x + self.dropout(attn_out)
        
        # 2. Feed-Forward Block
        ff_in = self.norm2(x)
        x_ff = self.activation_ffn(self.linear1(ff_in))
        x_out = self.linear2(self.dropout(x_ff))
        x_out = self.dropout(x_out) # Saída "crua" da FFN antes da soma residual
        
        # 3. Low-Rank Dynamic Routing
        # Usa h_in e x_out para decidir como compor o novo estado a partir do passado
        q_depth = self.adapter(h_in, x_out)
        
        # 4. Attention Residuals sobre o Histórico
        # Incluímos o resultado atual (x + x_out) como o candidato mais recente
        current_candidate = x + x_out
        updated_history = history + [current_candidate]
        
        h_new = self.depth_attn(q_depth, updated_history)
        
        return h_new

class AttnResEncoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, nhead: int, dim_feedforward: int, rank: int = 16):
        super().__init__()
        self.layers = nn.ModuleList([
            AttnResEncoderLayer(d_model, nhead, dim_feedforward, rank=rank)
            for _ in range(num_layers)
        ])
        self.norm_final = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # history começa com o embedding inicial (v0)
        history = [x]
        
        for layer in self.layers:
            # Cada camada produz um novo estado baseado em todo o passado
            x = layer(x, history, src_mask=mask)
            history.append(x)
            
        return self.norm_final(x)

# Exemplo de teste rápido de dimensões
if __name__ == "__main__":
    batch, seq, d_model = 8, 32, 512
    rank = 16
    model = AttnResEncoder(num_layers=6, d_model=d_model, nhead=8, dim_feedforward=2048, rank=rank)
    
    sample_input = torch.randn(batch, seq, d_model)
    output = model(sample_input)
    
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Histórico de camadas gerenciado internamente com sucesso.")
