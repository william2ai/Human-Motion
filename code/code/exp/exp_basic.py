import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SCINet, PAttn, TimeXer, \
    WPMixer, MultiPatchFormer, GraphNet


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            # 'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'MambaSimple': MambaSimple,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN,
            'TemporalFusionTransformer': TemporalFusionTransformer,
            "SCINet": SCINet,
            'PAttn': PAttn,
            'TimeXer': TimeXer,
            'WPMixer': WPMixer,
            'MultiPatchFormer': MultiPatchFormer,
            'GraphNet': GraphNet
        }
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

        self.device = self._acquire_device()
        model = self._build_model().to(self.device)
        if getattr(args, 'distributed', False):
            model = DDP(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True,
            )
        self.model = model

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if getattr(self.args, 'distributed', False):
            device = torch.device('cuda:{}'.format(self.args.local_rank))
            print('Use DDP GPU: cuda:{} rank {}/{}'.format(
                self.args.local_rank, self.args.rank, self.args.world_size
            ))
        elif self.args.use_gpu and self.args.gpu_type == 'cuda':
            if not self.args.use_multi_gpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    @property
    def is_main_process(self):
        return getattr(self.args, 'rank', 0) == 0

    def unwrap_model(self):
        return self.model.module if hasattr(self.model, 'module') else self.model

    def barrier(self):
        if getattr(self.args, 'distributed', False) and dist.is_initialized():
            dist.barrier()

    def reduce_mean(self, value):
        if not getattr(self.args, 'distributed', False):
            return float(value)
        tensor = torch.tensor(float(value), device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= self.args.world_size
        return tensor.item()

    def broadcast_bool(self, value, src=0):
        if not getattr(self.args, 'distributed', False):
            return bool(value)
        tensor = torch.tensor([1 if value else 0], device=self.device, dtype=torch.int)
        dist.broadcast(tensor, src=src)
        return bool(tensor.item())

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
