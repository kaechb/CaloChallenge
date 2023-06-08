import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.utils import spectral_norm, weight_norm

from helpers import TPReLU, WeightNormalizedLinear, equal_lr


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden,dropout,cond_dim, weightnorm,noise ):
        super().__init__()
        self.fc0 =  nn.Linear(embed_dim, hidden) if not weightnorm else WeightNormalizedLinear(embed_dim, hidden)
        self.fc0_cls = nn.Linear(embed_dim+cond_dim, hidden) if not weightnorm else WeightNormalizedLinear(embed_dim+cond_dim, hidden)
        self.fc1 =nn.Linear(hidden+embed_dim, embed_dim) if not weightnorm else WeightNormalizedLinear(hidden+embed_dim, embed_dim)
        self.fc1_cls = nn.Linear(hidden+1, embed_dim) if not weightnorm else WeightNormalizedLinear(hidden+1, embed_dim)
        self.cond_cls = nn.Linear(1, embed_dim) if not weightnorm else WeightNormalizedLinear(1, embed_dim)
        self.attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True, dropout=dropout) if not weightnorm else weight_norm(nn.MultiheadAttention(hidden, num_heads, batch_first=True, dropout=dropout),"in_proj_weight")
        self.act = nn.LeakyReLU()
        self.ln = nn.LayerNorm(hidden)
        self.noise=noise
        self.alpha=nn.Parameter(torch.ones(1,1,hidden),requires_grad=True)
        #self=self.to("cuda")
        #self.ln_cls = nn.LayerNorm(hidden)

    def forward(self, x, x_cls,cond, mask,weight=False):
        res = x.clone()
        res_cls = x_cls.clone()
        x = self.act(self.fc0(x))
        if self.noise:
            x = x + torch.normal(0, 1, (x.size(0), x.size(1), x.size(2)),device=x.device) * self.alpha
        x_cls = self.act(self.ln(self.fc0_cls(torch.cat((x_cls,cond),dim=-1))))
       # x=self.ln(x)
        if weight:
            x_cls,w = self.attn(x_cls, x, x, key_padding_mask=mask)
        else:
            x_cls,w = self.attn(x_cls, x, x, key_padding_mask=mask,need_weights=False)
        x_cls = self.act(self.fc1_cls(torch.cat((x_cls,mask.float().sum(1).unsqueeze(1).unsqueeze(1)/4000),dim=-1)))#+x.mean(dim=1).
        x_cls = self.act(F.glu(torch.cat((x_cls,self.cond_cls(cond[:,:,:1])),dim=-1))+res_cls)/math.sqrt(2)
        x=self.act(self.fc1(torch.cat((x,x_cls.expand(-1,x.shape[1],-1)),dim=-1))+res)

        #x=F.glu(torch.cat((x,self.cond(cond[:,:,:1]).expand(-1,x.shape[1],-1)),dim=-1))
        return x,x_cls,w




class Gen(nn.Module):
    def __init__(self, n_dim, l_dim_gen, hidden_gen, num_layers_gen, cond_dim, heads_gen,equallr,noise, **kwargs):
        super().__init__()
        if noise:
            self.embbed=equal_lr(nn.Linear(cond_dim, hidden_gen)) if equallr else nn.Linear(cond_dim,hidden_gen)
            self.embbed2 =equal_lr(nn.Linear(hidden_gen, l_dim_gen)) if equallr else nn.Linear(hidden_gen,l_dim_gen)
        else:
            self.embbed=equal_lr(nn.Linear(n_dim, l_dim_gen)) if equallr else nn.Linear(n_dim, l_dim_gen)

        self.blocks=nn.ModuleList([Block(embed_dim=l_dim_gen, num_heads=heads_gen,hidden=hidden_gen,cond_dim=cond_dim,dropout=0,weightnorm=False,noise=noise) for i in range(num_layers_gen)])

        self.out = equal_lr(nn.Linear(l_dim_gen, n_dim)) if equallr else nn.Linear(l_dim_gen, n_dim)
        self.act = nn.ReLU()
        self.cond = nn.Linear(2, l_dim_gen)
        self.start=nn.Parameter(torch.randn(1, 1, l_dim_gen), requires_grad=True)
        self.alpha=nn.Parameter(torch.ones(1,1,l_dim_gen),requires_grad=True)
        self.use_cond=cond_dim>0
        self.noise=noise

        if equallr:
            for i, block in enumerate(self.blocks):


                for name,mod in block.named_parameters():
                    # Apply the equalized learning rate transformation
                    if isinstance(self.blocks, nn.Linear):
                        equalized_module = equal_lr(mod)
                        setattr(self.blocks[i], name, equalized_module)

    def forward(self, x,mask,cond,weight=False):
        #x = self.act(self.embbed(x))
        cond=cond.unsqueeze(1)
        # cond[:,:,1]=cond[:,:,1]/4000
        if self.noise:
             x=self.start.expand(x[0], x[1], -1).clone()
        else:
            x=self.embbed(x)
        if weight:
            ws=[]
        x_cls = (x.sum(1)/4000).unsqueeze(1).clone() if not self.noise  else self.act(self.embbed2(self.act(self.embbed(cond))))
        for layer in self.blocks:
            x, x_cls, w = layer(x,x_cls=x_cls,mask=mask,cond=cond,weight=weight)
            if weight:
                ws.append(w)
        #x=F.glu(torch.cat((x,self.cond(cond[:,:,:]).expand(-1,x.shape[1],-1)),dim=-1))
        if weight:
            return self.out(x),ws
        else:
            return self.out(x)




class Disc(nn.Module):
    def __init__(self, n_dim, l_dim, hidden, num_layers, heads,dropout,spectralnorm,equallr,cond_dim, **kwargs):
        super().__init__()
        self.embbed = spectral_norm(nn.Linear(n_dim, l_dim)) if spectralnorm else WeightNormalizedLinear(n_dim, l_dim)
        self.blocks = nn.ModuleList([Block(embed_dim=l_dim, num_heads=heads, hidden=hidden, dropout=dropout,weightnorm=not spectralnorm,cond_dim=cond_dim,noise=False) for i in range(num_layers)])
        self.out = nn.Linear(l_dim, 1)
        self.embbed_cls = WeightNormalizedLinear(l_dim+cond_dim, l_dim) if not spectralnorm else spectral_norm(nn.Linear(l_dim+cond_dim, l_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, l_dim), requires_grad=True)
        self.ln_in = nn.LayerNorm(l_dim)

        self.ln = nn.LayerNorm(hidden)
        self.act = nn.LeakyReLU(0.01)
        self.fc1 = WeightNormalizedLinear(l_dim*num_layers, hidden) if not spectralnorm  else spectral_norm(nn.Linear(l_dim*num_layers, hidden))
        self.fc2 = WeightNormalizedLinear(hidden, l_dim) if not spectralnorm else spectral_norm(nn.Linear(hidden, l_dim))
        # self.embbed_cls = equal_lr(self.embbed_cls,"weight_orig") if equallr else self.embbed_cls
        # self.fc1 = equal_lr(self.fc1,"weight_orig") if equallr else self.fc1
        # self.fc2 = equal_lr(self.fc2,"weight_orig") if equallr else self.fc2
        # self.embbed = equal_lr(self.embbed,"weight_orig") if equallr else self.embbed

        for i, block in enumerate(self.blocks):
            for name,mod in block.named_parameters():
                if spectralnorm:
                    # Apply the equalized learning rate transformation
                    if isinstance(self.blocks, nn.Linear):
                        equalized_module = spectral_norm(mod)
                        setattr(self.blocks[i], name, equalized_module)


                if equallr:
                    if isinstance(self.blocks, nn.Linear):
                        equalized_module = equal_lr(mod)
                        setattr(self.blocks[i], name, equalized_module)




    def forward(self, x, mask,cond,weight=False):#mean_field=False
        ws=[]
        self.meanfields=[]
        cond=cond.unsqueeze(1)
        #cond[:,:,1]=cond[:,:,1]/4000
        x = self.embbed(x)
        x_cls =self.act(torch.cat((self.ln_in(x.sum(1)/4000).unsqueeze(1).clone(),cond),dim=-1))# self.cls_token.expand(x.size(0), 1, -1)
        x_cls = self.act(self.embbed_cls(x_cls))
        for layer in self.blocks:
            x,x_cls,w = layer(x, x_cls=x_cls, mask=mask,cond=cond,weight=weight)

            self.meanfields.append(x_cls.clone())
            #x_cls=(self.act(x_cls))
            if weight:
                ws.append(w)
        x_cls = self.act(self.fc2(self.ln(self.fc1(self.act(torch.cat(self.meanfields,dim=-1).squeeze(1))))))


        if weight:
            return self.out(x_cls),self.meanfields[-1],ws
        else:
            return self.out(x_cls),self.meanfields[-1]
if __name__ == "__main__":
    #def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

    z = torch.randn(256, 4000, 4,device="cpu")
    mask = torch.zeros((256, 4000),device="cpu").bool()
    cond=torch.cat(((~mask.bool()).float().sum(1).unsqueeze(-1)/4000,torch.randn(256,1,device="cpu")),dim=-1)
    model =Gen(n_dim=4, l_dim_gen=16, hidden_gen=128, num_layers_gen=8, heads_gen=8, dropout=0.0,cond_dim=2,equallr=False,noise=False).cpu()
    #model=torch.compile(model)
    pytorch_total_params = sum(p.cpu().numel() for p in model.parameters())
    print(pytorch_total_params)
    x=model.to("cpu")(z.cpu(),mask.cpu(),cond.cpu())
    print(x.std(),x.mean())
    model = Disc( n_dim=4, l_dim=16, hidden=128, num_layers=8, heads=4,dropout=0,spectralnorm=False,equallr=True,cond_dim=2, ).cpu()
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    s,s_cls,w=model(z.cpu(),mask.cpu(),cond=cond,weight=True)
    print(s.std(),s_cls.std())