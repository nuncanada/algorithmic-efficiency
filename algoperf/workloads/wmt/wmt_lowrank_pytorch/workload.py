"""WMT Low-Rank Dynamic AttnRes workload implemented in PyTorch."""

import torch
import torch.nn.functional as F
from algoperf import param_utils, pytorch_utils, spec
from algoperf.workloads.wmt.wmt_pytorch.workload import WmtWorkload
from algoperf.workloads.wmt.wmt_lowrank_pytorch.models import TransformerLowRank

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_utils.pytorch_setup()

class WmtLowRankWorkload(WmtWorkload):
  """WMT Low-Rank Dynamic AttnRes PyTorch workload."""

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    torch.random.manual_seed(rng[0])

    if self.activation == 'relu':
      activation = F.relu
    elif self.activation == 'tanh':
      activation = F.tanh
    elif self.activation == 'silu':
        activation = F.silu
    else:
      raise ValueError(f'Unknown activation function {self.activation}.')

    # Rank fixo em 16 por padrão para este experimento inicial
    model = TransformerLowRank(
      pre_ln=self.pre_ln,
      attention_temp=self.attention_temp,
      activation=activation,
      glu=self.glu,
      rank=16 
    )
    
    self._param_shapes = param_utils.pytorch_param_shapes(model)
    self._param_types = param_utils.pytorch_param_types(self._param_shapes)
    model.to(DEVICE)
    
    if N_GPUS > 1:
      from torch.nn.parallel import DistributedDataParallel as DDP
      from torch.nn import DataParallel as DP
      if USE_PYTORCH_DDP:
        model = DDP(model, device_ids=[RANK], output_device=RANK)
      else:
        model = DP(model)
        
    return model, None

  @property
  def validation_target_value(self) -> float:
    # Mantemos o mesmo target do WMT original para comparação justa
    return 30.8491

  @property
  def test_target_value(self) -> float:
    return 30.7219
