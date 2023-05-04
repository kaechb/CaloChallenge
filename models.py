import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weight_norm
from torch.nn import Parameter

from helpers import TPReLU, WeightNormalizedLinear

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden,dropout,weightnorm=True):
        super().__init__()
        self.fc0 = WeightNormalizedLinear(embed_dim, hidden) if weightnorm else nn.Linear(embed_dim, hidden)
        self.fc0_cls = (WeightNormalizedLinear(embed_dim+2, hidden)) if weightnorm else nn.Linear(embed_dim+2, hidden)
        self.fc1 = (WeightNormalizedLinear(hidden+embed_dim, embed_dim)) if weightnorm else nn.Linear(hidden+embed_dim, embed_dim)
        self.fc1_cls = (WeightNormalizedLinear(hidden+1, embed_dim)) if weightnorm else nn.Linear(hidden+1, embed_dim)
        self.cond_cls = nn.Linear(1, embed_dim)
        self.attn = weight_norm(nn.MultiheadAttention(hidden, num_heads, batch_first=True, dropout=dropout),"in_proj_weight")
        self.act = nn.LeakyReLU()
        self.ln = nn.LayerNorm(hidden)

    def forward(self, x, x_cls,cond, mask,weight=False):
        res = x.clone()
        res_cls = x_cls.clone()
        x = self.act(self.fc0(x))
        x_cls = self.act(self.ln(self.fc0_cls(torch.cat((x_cls,cond),dim=-1))))
        if weight:
            x_cls,w = self.attn(x_cls, x, x, key_padding_mask=mask)
        else:
            x_cls,w = self.attn(x_cls, x, x, key_padding_mask=mask,need_weights=False)
        x_cls = self.act(self.fc1_cls(torch.cat((x_cls,mask.float().sum(1).unsqueeze(1).unsqueeze(1)/4000),dim=-1)))#+x.mean(dim=1).
        x_cls = self.act(F.glu(torch.cat((x_cls,self.cond_cls(cond[:,:,:1])),dim=-1)))
        x=self.act(self.fc1(torch.cat((x,x_cls.expand(-1,x.shape[1],-1)),dim=-1))+res)
        x_cls =x_cls+res_cls
        #x=F.glu(torch.cat((x,self.cond(cond[:,:,:1]).expand(-1,x.shape[1],-1)),dim=-1))
        return x,x_cls,w




class Gen(nn.Module):
    def __init__(self, n_dim, l_dim_gen, hidden_gen, num_layers_gen, heads_gen, **kwargs):
        super().__init__()
        self.embbed = nn.Linear(n_dim, l_dim_gen)
        self.encoder = nn.ModuleList([Block(embed_dim=l_dim_gen, num_heads=heads_gen,hidden=hidden_gen,weightnorm=False,dropout=0) for i in range(num_layers_gen)])
        self.out = WeightNormalizedLinear(l_dim_gen, n_dim)
        self.act = nn.LeakyReLU()
        self.cond = nn.Linear(1, l_dim_gen)
    def forward(self, x,mask,cond,weight=False):
        x = self.act(self.embbed(x))
        cond=cond.unsqueeze(1)
        if weight:
            ws=[]
        x_cls = x.mean(1).unsqueeze(1).clone()
        for layer in self.encoder:
            x, x_cls, w = layer(x,x_cls=x_cls,mask=mask,cond=cond,weight=weight)
            if weight:
                ws.append(w)
            res=x_cls.clone()
        x=F.glu(torch.cat((x,self.cond(cond[:,:,:1]).expand(-1,x.shape[1],-1)),dim=-1))
        if weight:
            return self.out(x),ws
        else:
            return self.out(x)




class Disc(nn.Module):
    def __init__(self, n_dim, l_dim, hidden, num_layers, heads,dropout, **kwargs):
        super().__init__()
        self.embbed = WeightNormalizedLinear(n_dim, l_dim)
        self.encoder = nn.ModuleList([Block(embed_dim=l_dim, num_heads=heads, hidden=hidden,dropout=dropout,weightnorm=True) for i in range(num_layers)])
        self.out = WeightNormalizedLinear(l_dim, 1)
        self.embbed_cls = WeightNormalizedLinear(l_dim+2, l_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, l_dim), requires_grad=True)
        self.act = nn.LeakyReLU()
        self.fc1 = WeightNormalizedLinear(l_dim, hidden)
        self.fc2 = WeightNormalizedLinear(hidden, l_dim)
        self.ln = nn.LayerNorm(l_dim)


    def forward(self, x, mask,cond,weight=False):#mean_field=False
        ws=[]
        cond=cond.unsqueeze(1)
        x = self.act(self.embbed(x))
        x_cls = torch.cat(((x.sum(1)/4000).unsqueeze(1).clone(),cond),dim=-1)# self.cls_token.expand(x.size(0), 1, -1)
        x_cls = self.act(self.embbed_cls(x_cls))
        for layer in self.encoder:
            x,x_cls,w = layer(x, x_cls=x_cls, mask=mask,cond=cond,weight=weight)
            res=x_cls.clone()

            x_cls=(self.act(x_cls))
            if weight:
                ws.append(w)
        x_cls = self.act(self.ln(self.fc2(self.act(self.fc1(x_cls)))))
        if weight:
            return self.out(x_cls),res,ws
        else:
            return self.out(x_cls),res

if __name__ == "__main__":
    #def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

    z = torch.randn(256, 40, 4,device="cuda")
    mask = torch.zeros((256, 40),device="cuda").bool()
    cond=torch.cat(((~mask.bool()).float().sum(1).unsqueeze(-1),torch.randn(256,1,device="cuda")),dim=-1)
    model =Gen(n_dim=4, l_dim_gen=16, hidden_gen=32, num_layers_gen=8, heads_gen=8, dropout=0.0).cuda()
    x=model(z.cuda(),mask.cuda(),cond.cuda())
    print(x.std())
    model = Disc(4, 16, 64, 3, 8,0.1).cuda()
    s,s_cls,w=model(z.cuda(),mask.cuda(),cond=cond,weight=True)
    print(s.std(),s_cls.std())