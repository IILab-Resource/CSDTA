import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Sequential, Linear, ReLU
from einops.layers.torch import Rearrange, Reduce
from torch import nn, einsum

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import metrics
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from entmax.activations import entmax15

from dataset import PT_FEATURE_SIZE


CHAR_SMI_SET_LEN = 64


class MultiHeadAttentionReciprocal(nn.Module):
    
    
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        
        super().__init__()
        
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        
        
        self.W_Q = nn.Linear(d_model, n_head*d_k)
        self.W_K = nn.Linear(d_model, n_head*d_k)
        self.W_V = nn.Linear(d_model, n_head*d_v)
        self.W_O = nn.Linear(n_head*d_v, d_model)
        self.W_V_2 = nn.Linear(d_model, n_head*d_v)
        self.W_O_2 = nn.Linear(n_head*d_v, d_model)

        self.W_R = nn.Parameter(torch.Tensor(d_model, d_model))
        self.W_R2 = nn.Parameter(torch.Tensor(d_model, d_model))

        self.layer_norm = nn.LayerNorm(d_model)  
        self.dropout = nn.Dropout(dropout)  
        self.layer_norm_2 = nn.LayerNorm(d_model) 
        self.dropout_2 = nn.Dropout(dropout)
    
    
 

    def forward(self, q, k, v, v_2):
        
        batch, len_q, _ = q.size()
        batch, len_k, _ = k.size()
        batch, len_v, _ = v.size()
        batch, len_v_2, _ = v_2.size()        
        
            
        Q = self.W_Q(q).view([batch, len_q, self.n_head, self.d_k])
        K = self.W_K(k).view([batch, len_k, self.n_head, self.d_k])
        V = self.W_V(v).view([batch, len_v, self.n_head, self.d_v])
        V_2 = self.W_V_2(v_2).view([batch, len_v_2, self.n_head, self.d_v])
        
        
           
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2).transpose(2, 3)
        V = V.transpose(1, 2)
        V_2 = V_2.transpose(1,2) 
        
        attention = torch.matmul(Q, K)       
        attention = attention /np.sqrt(self.d_k)  
        attention_2 = attention.transpose(-2, -1)
        attention = entmax15(attention, dim=-1)
        attention_2 = entmax15(attention_2, dim=-1)
        output = torch.matmul(attention, V)
        output_2 = torch.matmul(attention_2, V_2)  
        output = output.transpose(1, 2).reshape([batch, len_q, self.d_v*self.n_head])
        output_2 = output_2.transpose(1, 2).reshape([batch, len_k, self.d_v*self.n_head])  
        output = self.W_O(output) 
        output_2 = self.W_O_2(output_2)
        output = self.dropout(output) 
        # output = self.layer_norm(output + q)
        output = self.layer_norm(output+torch.tensordot(q, self.W_R, dims=([-1], [0])))  
        output_2 = self.dropout(output_2)  
        # output_2 = self.layer_norm(output_2 + k) 
        output_2 = self.layer_norm(output_2+torch.tensordot(k, self.W_R2, dims=([-1], [0]))) 
        output_2 = F.relu(output_2)
        output = F.relu(output)
        return output, output_2, attention, attention_2

class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1)
        return x * y.expand_as(x)


class ConcatELU(nn.Module):
    '''
    Activation function which applies ELU in both directions.
    '''
    def forward(self,x):
        return torch.cat([F.relu(x),F.relu(-x)],dim=1)

class LayerNormChannels(nn.Module):
    def __init__(self,in_channels):
        '''
        Applies normalization in accross the input channels.
        Inputs:
            Number of channels.
        '''
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_channels)

    def forward(self,x):
        x = x.permute(0,2,1)
        x = self.layer_norm(x)
        x = x.permute(0,2,1)
        return x

class GatedConv(nn.Module):
    def __init__(self,in_channels,hidden_channels,dilaSize=3):
        '''
        Create a two layer deep network for ResNet with input gate.
        Inputs:
            in_channels     - Number of input channels.
            hidden_channels - Number of hidden channels.
        '''
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels,kernel_size=3, padding=dilaSize,dilation=dilaSize),
            ConcatELU(),
            # nn.ReLU(),
            nn.Conv1d(2*hidden_channels, 2*in_channels, kernel_size=1),
        )

    def forward(self,x):
        out = self.net(x)
        val, gate = out.chunk(2,dim=1)
        x = x + val  * torch.sigmoid(gate)
        return x


class GatedResNet(nn.Module):
    def __init__(self,in_channels,hidden_channels,output_channels,num_layers=3,pool_size=2,dilaSize = 2):
        '''
        Creates GatedResNet using the previous modules.
        Inputs:
            in_channels     - Number of input channels.
            hidden_channels - Number of hidden channels.
            output_channels - Number of output channels (3K-1 * in_channels)
        '''
        super().__init__()
        
        layers = [nn.Conv1d(in_channels, hidden_channels,kernel_size=3,padding=dilaSize,dilation=dilaSize)]
        for _ in range(num_layers):
                layers += [
                    GatedConv(hidden_channels,hidden_channels,dilaSize),
                    # LayerNormChannels(hidden_channels),
                    nn.MaxPool1d(pool_size),
                     
                ]
             
        layers += [
            ConcatELU(),
            # nn.ReLU(),
            nn.Conv1d(2*hidden_channels,output_channels,kernel_size=3,padding=dilaSize,dilation=dilaSize),
            # LayerNormChannels(output_channels),
            # SE_Block(output_channels),
            # nn.ReLU(),
        ]

        self.net = nn.Sequential(*layers)
         
        

    def forward(self,x):
        x = self.net(x.transpose(1,2))
        x = Rearrange('b n d -> b d n')(x)
        
        return x
    
class MuSigma(nn.Module):
    def __init__(self, input_dim):
        super(MuSigma, self).__init__()
        
        self.layer1 = nn.Sequential(
            # nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.ELU(),
        )
        self.layer2 = nn.Sequential(
            # nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.ELU(),
        )
    def forward(self,  embedding):
        mu = self.layer1(embedding)
        sigma = self.layer2(embedding)     + 1e-6    # elu() + 1 ensures the covariance matrix is positive definite.
        # sigma = torch.nn.Softplus()( sigma  ) + 1e-6
        return mu, sigma    

 

class GaussianNet(nn.Module):
    
    def __init__(self):
        
        super(GaussianNet, self).__init__()

        embed_dim = 256
        out_dim = 256
        hidden_dim = 256
        # onehot smiles
        # self.embed_smile = nn.Embedding( 65, embed_dim)
        # self.embed_prot = nn.Embedding( 26, embed_dim)
        
        self.onehot_smi_net = GatedResNet( 384, hidden_dim, out_dim, num_layers=5, pool_size=2,dilaSize=1)
        self.onehot_prot_net = GatedResNet( 1024, hidden_dim, out_dim, num_layers=5, pool_size=2,dilaSize=1)
        
 

        self.musigma_prot = MuSigma(out_dim)
        self.musigma_drug = MuSigma(out_dim)
        
       
        
        self.transform = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, 1),
            # nn.ReLU(),
            # nn.Linear(256, 1),
            # nn.ReLU(),
            # nn.Linear(1024, 512),
            # nn.ReLU(),
            # nn.Linear(512, 1),
        )
        # self.final_predict = nn.Linear(256, 1)
         
    def forward(self, seq, smi):
        
        # smi = self.embed_smile(smi) 
        # seq  = self.embed_prot(seq) 
        proteinFeature_onehot = self.onehot_prot_net( seq) 
        compoundFeature_onehot = self.onehot_smi_net( smi )

        proteinFeature_onehot = Reduce('b n d -> b d', 'max')(proteinFeature_onehot)
        compoundFeature_onehot = Reduce('b n d -> b d', 'max')(compoundFeature_onehot)
 

        mu_drug, logvar_drug = self.musigma_drug(compoundFeature_onehot)
        mu_prot, logvar_prot = self.musigma_prot(proteinFeature_onehot)
         
        euclidean_dist = torch.square( mu_drug - mu_prot )

        w2_dist = euclidean_dist + torch.square(torch.sqrt( torch.exp(logvar_drug))-torch.sqrt( torch.exp(logvar_prot)))


        # all_features = torch.cat([mu_drug*mu_prot, w2_dist], dim=1)   
        out = self.transform(w2_dist)
        
        # nllloss1 = self.NLLLoss(mu_drug, logvar_drug, compoundFeature_onehot)
        # nllloss2 = self.NLLLoss(mu_prot, logvar_prot, proteinFeature_onehot)
        # print(nllloss1.item(), nllloss2.item())
        return out #,  (nllloss1+nllloss2)
 
    def NLLLoss(self, mu, sigma, y):
        # sigma = torch.nn.Softplus()( sigma  ) + 1e-6
        sigma = torch.exp(sigma)
        loss = torch.mean((torch.log(sigma) / 2) + (torch.pow((mu - y), 2) / (2 * sigma)))
        # loss = torch.mean(   (torch.pow((mu - y), 2) / (2 * sigma)))

        return loss

class GaussianNetVis(nn.Module):
    
    def __init__(self):
        
        super(GaussianNetVis, self).__init__()

        embed_dim = 256
        out_dim = 256
        hidden_dim = 256
         
        
        self.onehot_smi_net = GatedResNet( 384, hidden_dim, out_dim, num_layers=5, pool_size=2,dilaSize=1)
        self.onehot_prot_net = GatedResNet( 1024, hidden_dim, out_dim, num_layers=5, pool_size=2,dilaSize=1)
        
        self.musigma_prot = MuSigma(out_dim)
        self.musigma_drug = MuSigma(out_dim)
        
        self.transform = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, 1),
        )
         
    def forward(self, seq, smi):
        
         
        proteinFeature_onehot = self.onehot_prot_net( seq) 
        compoundFeature_onehot = self.onehot_smi_net( smi )

        proteinFeature_onehot = Reduce('b n d -> b d', 'max')(proteinFeature_onehot)
        compoundFeature_onehot = Reduce('b n d -> b d', 'max')(compoundFeature_onehot)
 

        mu_drug, logvar_drug = self.musigma_drug(compoundFeature_onehot)
        mu_prot, logvar_prot = self.musigma_prot(proteinFeature_onehot)
         
        euclidean_dist = torch.square( mu_drug - mu_prot )

        w2_dist = euclidean_dist + torch.square(torch.sqrt(torch.exp(logvar_drug))-torch.sqrt(torch.exp(logvar_prot)))


        out = self.transform(w2_dist)
        
        return out,  w2_dist

class WOPretrainedNet(nn.Module):
    
    def __init__(self):
        
        super(WOPretrainedNet, self).__init__()

        embed_dim = 256
        out_dim = 256
        hidden_dim = 256
        # onehot smiles
        self.embed_smile = nn.Embedding( 65, 384)
        self.embed_prot = nn.Embedding( 26, 1024)
        
        # onehot smiles
        self.onehot_smi_net = GatedResNet( 384, hidden_dim, out_dim, num_layers=5, pool_size=2,dilaSize=1)
        # onehot protein
        self.onehot_prot_net = GatedResNet( 1024, hidden_dim, out_dim, num_layers=5, pool_size=2,dilaSize=1)
         
        self.musigma_prot = MuSigma(out_dim)
        self.musigma_drug = MuSigma(out_dim)
        
        self.transform = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, 1),
            
        )
        # self.final_predict = nn.Linear(256, 1)
         
    def forward(self, seq, smi):
        
        smi = self.embed_smile(smi) 
        seq  = self.embed_prot(seq) 
        proteinFeature_onehot = self.onehot_prot_net( seq) 
        compoundFeature_onehot = self.onehot_smi_net( smi )

        proteinFeature_onehot = Reduce('b n d -> b d', 'max')(proteinFeature_onehot)
        compoundFeature_onehot = Reduce('b n d -> b d', 'max')(compoundFeature_onehot)
        mu_drug, logvar_drug = self.musigma_drug(compoundFeature_onehot)
        mu_prot, logvar_prot = self.musigma_prot(proteinFeature_onehot)
         
        euclidean_dist = torch.square( mu_drug - mu_prot )

        w2_dist = euclidean_dist + torch.square(torch.sqrt(torch.exp(logvar_drug))-torch.sqrt(torch.exp(logvar_prot)))

    
        out = self.transform(w2_dist)
        
        return out

class WOGaussianNet(nn.Module):
    
    def __init__(self):
        
        super(WOGaussianNet, self).__init__()

        embed_dim = 256
        out_dim = 256
        hidden_dim = 256
        
        self.onehot_smi_net = GatedResNet( 384, hidden_dim, out_dim, num_layers=5, pool_size=2,dilaSize=1)
        # onehot protein
        self.onehot_prot_net = GatedResNet( 1024, hidden_dim, out_dim, num_layers=5, pool_size=2,dilaSize=1)
      
       
        
        self.transform = nn.Sequential(
            nn.LayerNorm(out_dim*2),
            nn.Linear(out_dim*2, 1),
            
        )
         
    def forward(self, seq, smi):
        
        proteinFeature_onehot = self.onehot_prot_net( seq) 
        compoundFeature_onehot = self.onehot_smi_net( smi )

        proteinFeature_onehot = Reduce('b n d -> b d', 'max')(proteinFeature_onehot)
        compoundFeature_onehot = Reduce('b n d -> b d', 'max')(compoundFeature_onehot)

 
        all_features =   torch.cat([proteinFeature_onehot, compoundFeature_onehot], dim=1)    
        out = self.transform(all_features)
        
        return out
 
    
class VIBNet(nn.Module):
    
    def __init__(self):
        
        super(VIBNet, self).__init__()

        embed_dim = 256
        out_dim = 256
        hidden_dim = 256
        # onehot smiles
        self.embed_smile = nn.Embedding( 65, embed_dim)
        self.embed_prot = nn.Embedding( 26, embed_dim)
        
        # onehot smiles
        self.onehot_smi_net = GatedResNet( embed_dim, hidden_dim, out_dim)
        # onehot protein
        self.onehot_prot_net = GatedResNet( 1024, hidden_dim, out_dim)
        
    
        
        self.transform = nn.Sequential(
            # nn.LayerNorm(out_dim),
            nn.Linear(out_dim, 256),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128, 1),
        )

        self.enc_mean = nn.Linear(out_dim*2,out_dim)
        self.enc_std = nn.Linear(out_dim*2,out_dim)

    def reparametrize_n(self, mu, std, n=1):
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + eps * std

    def forward(self, seq, smi):
        

        smile_vectors_onehot = self.embed_smile(smi) 

        proteinFeature_onehot = self.onehot_prot_net( seq)#prot_vectors_onehot )
        compoundFeature_onehot = self.onehot_smi_net( smile_vectors_onehot )
        proteinFeature_onehot = Reduce('b n d -> b d', 'max')(proteinFeature_onehot)
        compoundFeature_onehot = Reduce('b n d -> b d', 'max')(compoundFeature_onehot)
        all_features = torch.cat([proteinFeature_onehot, compoundFeature_onehot], dim=1)    
        mu = self.enc_mean(all_features)
        std = F.softplus( self.enc_std(all_features) - 5  , beta=1) # TODO: why -5?
        new_embs = self.reparametrize_n(mu, std)


        out = self.transform(new_embs)

         
        return out, (mu, std) 

def test(model: nn.Module, test_loader, loss_function, device, show):
    model.eval()
    test_loss = 0
    outputs = []
    targets = []
    with torch.no_grad():
        for idx, (*x, y) in tqdm(enumerate(test_loader), disable=not show, total=len(test_loader)):
            for i in range(len(x)):
                x[i] = x[i].to(device)
            y = y.to(device)

            y_hat  = model(*x)

            test_loss += loss_function(y_hat.view(-1), y.view(-1)).item()
            outputs.append(y_hat.cpu().numpy().reshape(-1))
            targets.append(y.cpu().numpy().reshape(-1))

    targets = np.concatenate(targets).reshape(-1)
    outputs = np.concatenate(outputs).reshape(-1)

    # test_loss /= len(test_loader.dataset)

    evaluation = {
        'loss': test_loss,
        'c_index': metrics.c_index(targets, outputs),
        'RMSE': metrics.RMSE(targets, outputs),
        'MAE': metrics.MAE(targets, outputs),
        'SD': metrics.SD(targets, outputs),
        'CORR': metrics.CORR(targets, outputs),
    }

    return evaluation


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.01, emb_name='embed'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embed'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}