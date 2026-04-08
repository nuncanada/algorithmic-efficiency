# Low-Rank Dynamic Attention Residuals (LRD-AttnRes)

Este documento descreve a implementação da arquitetura **LRD-AttnRes** no framework AlgoPerf, baseada na hipótese de que as principais relações no espaço latente de modelos Transformer operam em subespaços de dimensionalidade reduzida (posto menor que 32).

## 1. Fundamentação Teórica

A arquitetura padrão de Transformers utiliza conexões residuais fixas ($x_{l} = x_{l-1} + f(x_{l-1})$), o que pode levar à diluição da informação em redes muito profundas. O conceito de **Attention Residuals (AttnRes)** propõe que a entrada de uma camada seja uma soma ponderada (via atenção softmax) de todas as saídas das camadas anteriores.

### A Hipótese de Baixo Posto
Inspirado na técnica LoRA (Low-Rank Adaptation) e em estudos de interpretabilidade, este modelo introduz um **Adaptador Dinâmico de Baixo Posto**. Em vez de usar uma query estática para a atenção na profundidade, o modelo gera uma query dinâmica que depende do contexto atual através de um gargalo de dimensão reduzida $r < 32$.

## 2. Detalhes da Arquitetura

Para cada camada $l$ no Encoder, o processo segue os seguintes passos:

1.  **Fusão Contextual**: O adaptador recebe a entrada inicial do bloco ($h_{in}$) e o refinamento produzido pela Feed-Forward Network ($x_{out}$).
2.  **Compressão (Bottleneck)**: Projetamos ambos os vetores para um subespaço de posto $r$ via matrizes de projeção descendente:
    $$z_l = \sigma(h_{in} W_{in}^{down} + x_{out} W_{out}^{down})$$
    Onde $z_l \in \mathbb{R}^r$ e $\sigma$ é a ativação SiLU.
3.  **Expansão (Up-projection)**: Projetamos $z_l$ de volta para a dimensão do modelo $d$ para gerar a query de profundidade:
    $$q_l = z_l W_{up}$$
4.  **Atenção na Profundidade**: Utilizamos $q_l$ para calcular os pesos de atenção sobre o histórico de todas as saídas anteriores $V = [v_0, v_1, \dots, v_{l-1}]$:
    $$\alpha_i = \text{softmax}(q_l^\top \text{RMSNorm}(v_i))$$
5.  **Recomposição**: O novo estado $h_l$ é a soma ponderada do histórico baseada nos pesos dinâmicos.

## 3. Implementação no AlgoPerf

A implementação foi isolada no workload `wmt_lowrank` para permitir benchmarks comparativos.

### Estrutura de Arquivos
- `algoperf/workloads/wmt/wmt_lowrank_pytorch/models.py`: Contém as classes `LowRankDynamicAdapter`, `DepthAttention` e o novo `AttnResEncoder`.
- `algoperf/workloads/wmt/wmt_lowrank_pytorch/workload.py`: Wrapper do workload que injeta o novo modelo no pipeline do WMT.
- `algorithms/wmt_lowrank_test/`: Contém o script de submissão (AdamW + Cosine Warmup) para testar a convergência.

### Hiperparâmetros
- **Rank ($r$)**: Definido por padrão como **16**. Pode ser ajustado no `init_model_fn` do workload ou via tuning externo.
- **Initialization**: A matriz $W_{up}$ é inicializada em zero (estilo LoRA) para que, no início do treinamento, a atenção de profundidade se comporte de forma estável antes de aprender o roteamento dinâmico.

## 4. Como Executar

Para rodar o benchmark contra este modelo em um ambiente com **GPU**:

```bash
# Instalação de dependências necessárias para WMT e PyTorch (GPU)
pip install -e ".[wmt,pytorch_gpu]"

# Execução do script de verificação experimental (compara Baseline vs Low-Rank)
# Este script utiliza dummy data para medir o overhead real da arquitetura com torch.compile em GPU.
python3 tests/verify_lowrank.py

# Execução do runner oficial (requer dataset WMT configurado)
python3 submission_runner.py \
    --framework=pytorch \
    --workload=wmt_lowrank \
    --experiment_dir=./experiments/wmt_lowrank \
    --experiment_name=lrd_attnres_v1 \
    --submission_path=algorithms/wmt_lowrank_test/submission.py \
    --tuning_search_space=algorithms/wmt_lowrank_test/tuning_search_space.json
```

## 5. Resultados Experimentais (Benchmarking em NVIDIA A100-SXM4-80GB)

Foram realizados testes de performance comparando a arquitetura base com a nova arquitetura **LRD-AttnRes** utilizando `torch.compile` (backend Inductor). O teste consistiu em execuções de 5 minutos para cada modelo com um batch size de 32 e sequência de 256.

### Métricas de Performance Comparativa (Training & Inference)
Os testes abaixo foram realizados com `torch.compile` e dados sintéticos para garantir paridade estatística em um tempo de execução controlado (~10 min total).

| Métrica | Baseline | Low-Rank (LRD-AttnRes) | Diferença/Overhead |
| :--- | :--- | :--- | :--- |
| **Training Throughput (step/s)** | 4.95 | 4.57 | -7.63% |
| **Training Peak Mem (GB)** | 13.12 | 14.44 | +1.32 GB |
| **Inference Latency (ms)** | 38.04 | 41.21 | +8.33% |
| **Inference Throughput (step/s)** | 25.87 | 23.94 | -7.47% |
| **Final Loss (Eval)** | 1.4212 | 1.4204 | -0.00 |
| **Final Accuracy (Eval)** | 1.0000 | 1.0000 | +0.00 |
| **Final Perplexity (Eval)** | 4.14 | 4.14 | -0.00 |

### Estatísticas do Framework (Low-Rank)
Abaixo, o top 10 de kernels e operações (via `torch.profiler`):
- **Tempo de Execução CUDA Total**: ~471ms (amostra de 3 steps).
- **Operações Dominantes**: `aten::mm` (Multiplicações de Matrizes) representa ~36% do tempo CUDA.
- **Compilação**: O overhead de tempo CPU é mitigado pelo `torch.compile`, mas a lógica de roteamento dinâmico e atenção de profundidade introduz kernels adicionais de leitura/escrita na memória global (devido ao histórico de estados).

### Observações de Implementação
Durante a validação experimental, foram realizados os seguintes ajustes:
1.  **Workload Import**: Adicionado o import do pacote `torch` em `algoperf/workloads/wmt/wmt_lowrank_pytorch/workload.py`.
2.  **Self-Attention API**: Corrigida a chamada do `MultiheadAttention` em `models.py` para evitar argumentos posicionais incorretos.
3.  **Scaling Inconsistency**: Removido o escalonamento extra por `sqrt(d_model)` na entrada do Encoder para garantir paridade com o baseline original.
4.  **Estrutura de Diretórios**: Renomeado o diretório para `wmt_lowrank_pytorch` para seguir o padrão `[workload]_[framework]` do AlgoPerf.
5.  **Validação de Métricas**: O script `tests/evaluate_architectures.py` foi utilizado para confirmar que Accuracy e Perplexity são calculados corretamente em ambos os modelos.

## 6. Vantagens Esperadas
- **Eficiência de Parâmetros**: Adição insignificante de parâmetros devido ao baixo posto.
- **Roteamento Inteligente**: O modelo pode "decidir" ignorar camadas ruidosas e buscar informação diretamente do embedding inicial ou de camadas específicas.
- **Estabilidade em Profundidade**: Mitiga o problema do crescimento descontrolado da magnitude do estado oculto em arquiteturas muito profundas.
