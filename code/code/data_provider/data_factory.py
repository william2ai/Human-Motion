from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader, RealdispLoader,DaphnetLoader
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import pdb
import torch

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    'Realdisp': RealdispLoader,
    'Daphnet': DaphnetLoader
}

# def custom_collate_fn(batch):
#     # Filter out entries where the first element in the tuple has no valid size
#     pdb.set_trace()
#     filtered_batch = [item for item in batch if item[0].size(1) > 0]
#     return torch.utils.data.dataloader.default_collate(filtered_batch)

def _build_loader(args, data_set, batch_size, shuffle_flag, drop_last, collate_fn=None, distributed=None):
    if distributed is None:
        distributed = getattr(args, 'distributed', False)

    sampler = None
    if distributed:
        sampler = DistributedSampler(
            data_set,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=shuffle_flag,
            drop_last=drop_last,
        )
        shuffle_flag = False

    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle_flag,
        'num_workers': args.num_workers,
        'drop_last': drop_last,
        'sampler': sampler,
        'pin_memory': bool(args.use_gpu),
    }
    if args.num_workers > 0:
        loader_kwargs['persistent_workers'] = True
    if collate_fn is not None:
        loader_kwargs['collate_fn'] = collate_fn

    return DataLoader(data_set, **loader_kwargs)


def data_provider(args, flag, distributed=None):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    if distributed is None:
        distributed = getattr(args, 'distributed', False)
    if distributed:
        batch_size = max(1, args.batch_size // args.world_size)
    freq = args.freq


    if args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = _build_loader(
            args,
            data_set,
            batch_size,
            shuffle_flag,
            drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len),
            distributed=distributed,
        )
        return data_set, data_loader
    elif args.task_name == 'imputation_graph':
        drop_last = True
        data_set = Data(
            args = args,
            size=[args.seq_len, args.label_len, args.pred_len],
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = _build_loader(
            args,
            data_set,
            batch_size,
            shuffle_flag,
            drop_last,
            distributed=distributed,
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        data_loader = _build_loader(
            args,
            data_set,
            batch_size,
            shuffle_flag,
            drop_last,
            distributed=distributed,
        )
        return data_set, data_loader
