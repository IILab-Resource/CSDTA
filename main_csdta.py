import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
# from apex import amp  # uncomment lines related with `amp` to use apex

from dataset import MyDataset,SeqDataset,SeqEmbDataset
from GaussianModel import GaussianNet, test, WOGaussianNet
import joblib


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
# from radam import *
print(sys.argv)

# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓
SHOW_PROCESS_BAR = True
data_path = '/home/wangchunyu/pdbbind_v2016_refined/data/'
seed = np.random.randint(33927, 33928) ##random 
#path = Path(f'../runs/DeepDTAF_{datetime.now().strftime("%Y%m%d%H%M%S")}_{seed}')
path = Path(f'/home/wangchunyu/pdbbind_v2016_refined/runs/pkt2_{datetime.now().strftime("%m%d%H")}_{seed}')
device = torch.device("cuda:1")  # or torch.device('cpu')
            
max_seq_len = 1000  
max_smi_len = 150

batch_size = 256#256#16
n_epoch = 100
interrupt = None
save_best_epoch = 1 #  when `save_best_epoch` is reached and the loss starts to decrease, save best model parameters
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑

# GPU uses cudnn as backend to ensure repeatable by setting the following (in turn, use advances function to speed up training)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True 

torch.manual_seed(seed)
np.random.seed(seed)

writer = SummaryWriter(path)
f_param = open(path / 'parameters.txt', 'w')

print(f'device={device}')
print(f'seed={seed}')
print(f'write to {path}')
f_param.write(f'device={device}\n'
          f'seed={seed}\n'
          f'write to {path}\n')
               

print(f'max_seq_len={max_seq_len}\n'
      f'max_smi_len={max_smi_len}')

f_param.write(f'max_seq_len={max_seq_len}\n'
      f'max_smi_len={max_smi_len}\n')

# model = VIBNet()
model = GaussianNet()
model = model.to(device)
f_param.write('model: \n')
f_param.write(str(model)+'\n')
f_param.close()

prots = joblib.load('data/prot_emb_all.job')
drugs = joblib.load('data/drug_emb_all.job')
# SeqDataset
# data_loaders = {phase_name:
#                     DataLoader(SeqDataset(data_path,   phase_name,
#                                          max_seq_len, max_smi_len),
#                                batch_size=batch_size,
#                                pin_memory=True,
#                                num_workers=8,
#                                shuffle=True )
#                 for phase_name in ['training', 'validation','test', 'test105', 'test71']}
data_loaders = {phase_name:
                    DataLoader(SeqEmbDataset(data_path, prots, drugs, phase_name,
                                         max_seq_len, max_smi_len),
                               batch_size=batch_size,
                               pin_memory=True,
                               num_workers=8,
                               shuffle=True #if phase_name=='training' else False
                                )
                for phase_name in ['training', 'validation','test', 'test105', 'test71']}
optimizer = optim.AdamW(model.parameters())


# optimizer = torchcontrib.optim.SWA(optimizer)
# normal_max_iters = int(self.configer.get('solver', 'max_iters') * 0.75)
# swa_step_max_iters = (self.configer.get('solver',
#                                                     'max_iters') - normal_max_iters) // 5 + 1  # we use 5 ensembles here

# def swa_lambda_poly(iters):
#     if iters < normal_max_iters:
#         return pow(1.0 - iters / normal_max_iters, 0.9)
#     else:  # set lr to half of initial lr and start swa
#         return 0.5 * pow(1.0 - ((iters - normal_max_iters) % swa_step_max_iters) / swa_step_max_iters, 0.9)

# scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=swa_lambda_poly)


scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-4, #5e-3, 
                                            epochs=n_epoch,
                                          steps_per_epoch=len(data_loaders['training']),
                                          )

# pct_start=args.pct_step_up,
#             div_factor=args.div_factor,
#             final_div_factor=args.final_div_factor,
#             cycle_momentum=args.cycle_momentum,
# 'pct_start': 0.1,               # OneCycleLR
#     'anneal_strategy': 'cos',       # OneCycleLR
#     'div_factor': 1e3,              # OneCycleLR
#     'final_div_factor': 1e3,        # OneCycleLR
loss_function = nn.MSELoss(reduction='mean')



# fp16
# model, optimizer = amp.initialize(model, optimizer, opt_level="O1") 
        
start = datetime.now()
print('start at ', start)

model_file_name = 'gaussian_best_model_dist'
early_stopping = EarlyStopping(patience=15, verbose=True, path=model_file_name)
    
best_epoch = -1
best_val_loss = 100000000
for epoch in range(1, n_epoch + 1):
    tbar = tqdm(enumerate(data_loaders['training']), disable=not SHOW_PROCESS_BAR, total=len(data_loaders['training']))
    _lambda = epoch/n_epoch
    for idx, (*x, y) in tbar:
        model.train()

        for i in range(len(x)):
            x[i] = x[i].to(device)
        y = y.to(device)

        optimizer.zero_grad()
        output  = model(*x) 
        loss = loss_function(output.view(-1), y.view(-1))  #  + 10 * nllloss
        loss.backward() 
        optimizer.step()
        scheduler.step()

        tbar.set_description(f' * Train Epoch {epoch} Loss={loss.item() :.3f}   ')

    performance = test(model, data_loaders['validation'], loss_function, device, False)
    val_loss = performance['loss']
    early_stopping(val_loss, model)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break

     
            
            
model.load_state_dict(torch.load(  model_file_name ))

with open(path / 'result.txt', 'w') as f:
    f.write(f'best model found at epoch NO.{best_epoch}\n')
    for _p in [  'validation', 'test', 'test105', 'test71']:
        performance = test(model, data_loaders[_p], loss_function, device, SHOW_PROCESS_BAR)
        f.write(f'{_p}:\n')
        print(f'{_p}:')
        for k, v in performance.items():
            f.write(f'{k}: {v}\n')
            print(f'{k}: {v}\n')
        f.write('\n')
        print()

print('training finished')

end = datetime.now()
print('end at:', end)
print('time used:', str(end - start))
