import json
import os
import warnings
import argparse
# from torchstat import stat
if 'p' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['p']
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
warnings.filterwarnings('ignore')

import numpy as np
import torch
import random
# from thop import profile
from fastNLP.core.callbacks.topk_saver import TopkSaver
from fastNLP import cache_results, prepare_torch_dataloader
from fastNLP import print
from fastNLP import Trainer
from fastNLP import TorchGradClipCallback
from fastNLP import FitlogCallback, CheckpointCallback, TorchGradClipCallback
from fastNLP import SortedSampler, BucketedBatchSampler
from fastNLP import TorchWarmupCallback
import fitlog

# fitlog.debug()

from model.model import CNNNer
from model.metrics_length import NERMetric
from data.ner_pipe import SpanNerPipe
from data.padder import Torch3DMatrixPadder

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=2e-5, type=float)
parser.add_argument('--encoder_lr', default=2e-5, type=float)
parser.add_argument('-b', '--batch_size', default=24, type=int)
parser.add_argument('-n', '--n_epochs', default=50, type=int)
parser.add_argument('--warmup', default=0.1, type=float)
parser.add_argument('-d', '--dataset_name', default='ace2004', type=str) # ace2005 genia ace2004
parser.add_argument('--model_name', default=None, type=str)
parser.add_argument('--cnn_depth', default=1, type=int)
parser.add_argument('--cnn_dim', default=120, type=int)
parser.add_argument('--num', default=1, type=int)
parser.add_argument('--logit_drop', default=0, type=float)
parser.add_argument('--biaffine_size', default=200, type=int)
parser.add_argument('--n_head', default=5, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--n_layer', default=1, type=int)
parser.add_argument('--accumulation_steps', default=1, type=int)
parser.add_argument('--separateness_rate', default=5, type=int)
parser.add_argument('--theta', default=1, type=float)
parser.add_argument('--loss_theta', default=1, type=float)

args = parser.parse_args()
dataset_name = args.dataset_name
if args.model_name is None:
    if 'genia' in args.dataset_name:
        args.model_name = '/home/caiyuxiang/.cache/huggingface/hub/models--dmis-lab--biobert-v1.1/snapshots/551ca18efd7f052c8dfa0b01c94c2a8e68bc5488'
    elif args.dataset_name in ('conll03'):
        args.model_name = 'models--bert-large-cased'
    elif args.dataset_name in ('ace2004','ace2005'):
        args.model_name = '/home/caiyuxiang/.cache/huggingface/hub/models--roberta-base/snapshots/bc2764f8af2e92b6eb5679868df33e224075ca68'

model_name = args.model_name
n_head = args.n_head
######hyper
non_ptm_lr_ratio = 100
schedule = 'linear'
weight_decay = 1e-4
size_embed_dim = 25
ent_thres = 0.5
kernel_size = 3
######hyper

fitlog.set_log_dir('logs/')

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
def seed_torch(seed=43):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False


seed = fitlog.set_rng_seed(rng_seed=args.seed)
seed_torch(args.seed)
os.environ['FASTNLP_GLOBAL_SEED'] = str(seed)
fitlog.add_hyper(args)
fitlog.add_hyper_in_file(__file__)

@cache_results('caches/ner_caches.pkl', _refresh=False)
def get_data(dataset_name, model_name):
    # 以下是我们自己的数据
    if dataset_name == 'ace2004':
        paths = 'preprocess/outputs/ace2004'
    elif dataset_name == 'ace2005':
        paths = 'preprocess/outputs/ace2005'
    elif dataset_name == 'genia':
        paths = 'preprocess/outputs/genia'
    elif dataset_name == 'conll03':
        paths = 'preprocess/outputs/conll03'
    else:
        raise RuntimeError("Does not support.")
    pipe = SpanNerPipe(model_name=model_name)
    dl = pipe.process_from_file(paths)

    return dl, pipe.matrix_segs


dl, matrix_segs = get_data(dataset_name, model_name)


def densify(x):
    x = x.todense().astype(np.float32)
    return x


dl.apply_field(densify, field_name='matrix', new_field_name='matrix', progress_bar='Densify')

print(dl)
label2idx = getattr(dl, 'ner_vocab') if hasattr(dl, 'ner_vocab') else getattr(dl, 'label2idx')
print(f"{len(label2idx)} labels: {label2idx}, matrix_segs:{matrix_segs}")
dls = {}
for name, ds in dl.iter_datasets():
    ds.set_pad('matrix', pad_fn=Torch3DMatrixPadder(pad_val=ds.collator.input_fields['matrix']['pad_val'],
                                                    num_class=matrix_segs['ent'],
                                                    batch_size=args.batch_size))

    if name == 'train':
        _dl = prepare_torch_dataloader(ds, batch_size=args.batch_size, num_workers=0,
                                       batch_sampler=BucketedBatchSampler(ds, 'input_ids',
                                                                          batch_size=args.batch_size,
                                                                          num_batch_per_bucket=30),
                                       pin_memory=True, shuffle=True)

    else:
        _dl = prepare_torch_dataloader(ds, batch_size=args.batch_size, num_workers=0,
                                       sampler=SortedSampler(ds, 'input_ids'), pin_memory=True, shuffle=False)
    dls[name] = _dl

model = CNNNer(model_name, num_ner_tag=matrix_segs['ent'], cnn_dim=args.cnn_dim, biaffine_size=args.biaffine_size,
               size_embed_dim=size_embed_dim, logit_drop=args.logit_drop,n_layer=args.n_layer,
               kernel_size=kernel_size, n_head=n_head, cnn_depth=args.cnn_depth,separateness_rate=args.separateness_rate/100,theta=args.theta)

# optimizer
parameters = []
ln_params = []
non_ln_params = []
non_pretrain_params = []
non_pretrain_ln_params = []

import collections

counter = collections.Counter()
for name, param in model.named_parameters():
    counter[name.split('.')[0]] += torch.numel(param)
print(counter)
print("Total param ", sum(counter.values()))
fitlog.add_to_line(json.dumps(counter, indent=2))
fitlog.add_other(value=sum(counter.values()), name='total_param')
for name, param in model.named_parameters():
    name = name.lower()
    if param.requires_grad is False:
        continue
    if 'pretrain_model' in name:
        if 'norm' in name or 'bias' in name:
            ln_params.append(param)
        else:
            non_ln_params.append(param)
    else:
        if 'norm' in name or 'bias' in name:
            non_pretrain_ln_params.append(param)
        else:      
            non_pretrain_params.append(param)
optimizer = torch.optim.AdamW([{'params': non_ln_params, 'lr': args.lr, 'weight_decay': weight_decay},
                               {'params': ln_params, 'lr': args.lr, 'weight_decay': 0},
                               {'params': non_pretrain_ln_params, 'lr': args.lr * 
                                non_ptm_lr_ratio, 'weight_decay': 0},
                               {'params': non_pretrain_params, 'lr': args.lr * non_ptm_lr_ratio,
                                'weight_decay': weight_decay}])
# callbacks
callbacks = []
callbacks.append(FitlogCallback(log_loss_every=20))
callbacks.append(CheckpointCallback(monitor='f#f#test',save_evaluate_results=True, folder='_saved_models', topk=3))
callbacks.append(TorchGradClipCallback(clip_value=5))
callbacks.append(TorchWarmupCallback(warmup=args.warmup, schedule=schedule))
train_dls = {}
evaluate_dls = {}

if 'dev' in dls:
    evaluate_dls['dev'] = dls['dev']
if 'test' in dls:
    evaluate_dls['test'] = dls['test']
allow_nested = True
metrics = {'f': NERMetric(matrix_segs=matrix_segs, ent_thres=ent_thres, allow_nested=allow_nested)}

trainer = Trainer(model=model,
                  driver='torch',
                  train_dataloader=dls.get('train'),
                  evaluate_dataloaders=evaluate_dls,
                  optimizers=optimizer,
                  callbacks=callbacks,
                  overfit_batches=0,
                  device=0,
                  n_epochs=args.n_epochs,
                  metrics=metrics,
                  monitor='f#f#dev',
                  evaluate_every=-1,
                  evaluate_use_dist_sampler=True,
                  accumulation_steps=args.accumulation_steps,
                  fp16=False,
                  progress_bar='rich')

trainer.run(num_train_batch_per_epoch=-1, num_eval_batch_per_dl=-1, num_eval_sanity_batch=1)
fitlog.finish()
