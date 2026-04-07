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
- `algoperf/workloads/wmt/wmt_lowrank/models.py`: Contém as classes `LowRankDynamicAdapter`, `DepthAttention` e o novo `AttnResEncoder`.
- `algoperf/workloads/wmt/wmt_lowrank/workload.py`: Wrapper do workload que injeta o novo modelo no pipeline do WMT.
- `algorithms/wmt_lowrank_test/`: Contém o script de submissão (AdamW + Cosine Warmup) para testar a convergência.

### Hiperparâmetros
- **Rank ($r$)**: Definido por padrão como **16**. Pode ser ajustado no `init_model_fn` do workload ou via tuning externo.
- **Initialization**: A matriz $W_{up}$ é inicializada em zero (estilo LoRA) para que, no início do treinamento, a atenção de profundidade se comporte de forma estável antes de aprender o roteamento dinâmico.

## 4. Como Executar

Para rodar o benchmark contra este modelo:

```bash
python3 submission_runner.py \
    --framework=pytorch \
    --workload=wmt_lowrank \
    --experiment_dir=./experiments/wmt_lowrank \
    --experiment_name=lrd_attnres_v1 \
    --submission_path=algorithms/wmt_lowrank_test/submission.py \
    --tuning_search_space=algorithms/wmt_lowrank_test/tuning_search_space.json
```

## 5. Vantagens Esperadas
- **Eficiência de Parâmetros**: Adição insignificante de parâmetros devido ao baixo posto.
- **Roteamento Inteligente**: O modelo pode "decidir" ignorar camadas ruidosas e buscar informação diretamente do embedding inicial ou de camadas específicas.
- **Estabilidade em Profundidade**: Mitiga o problema do crescimento descontrolado da magnitude do estado oculto em arquiteturas muito profundas.
