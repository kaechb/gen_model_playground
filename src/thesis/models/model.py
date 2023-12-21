import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

class Model(nn.Module):
    def __init__(self, in_features: int,out_features:int, num_blocks: int = 4, hidden_features: int = 64,cond_features=0,spectral=False,batch_norm=True,residual=True,time_features=0,bias=False,**kwargs):

        super(Model, self).__init__()
        self.time=time_features>0
        self.inblock = nn.Linear(in_features+int(cond_features)+int(time_features), hidden_features,bias=bias) if not spectral else spectral_norm(nn.Linear(in_features+int(cond_features)+int(time_features), hidden_features,bias=bias))
        self.midblocks = nn.ModuleList([Block(hidden_features,spectral,batch_norm,bias,time_features) for _ in range(num_blocks)])
        self.outblock = nn.Linear(hidden_features, out_features,bias=bias) if not spectral else spectral_norm(nn.Linear(hidden_features, out_features,bias=bias))
        self.act = lambda x: x*torch.nn.functional.sigmoid(x)
        self.residual=residual

    def forward(self, x: torch.Tensor, t=None,cond=None,feature_matching=False,) -> torch.Tensor:

        if self.time and t is not None:
            if len(t)==1:
                t=t.repeat(x.shape[0],1)
            elif len(t.shape)==1:
                t=t[:,None]

            x=torch.cat((x,t),dim=-1)
        if cond is not None:
            x=torch.cat((x,cond),dim=-1)
        val = self.inblock(x)
        for midblock in self.midblocks:
            if self.residual:
                val = val+midblock(val,t)
            else:
                val = midblock(val,t)
        x=self.act(val)
        x = self.outblock(x)
        if feature_matching:
            return x,val
        else:
            return x


class Block(nn.Module):
    def __init__(self, hidden_features,spectral,batch_norm,bias,time_features=0):
        super(Block, self).__init__()
        self.linear = spectral_norm(nn.Linear(hidden_features+time_features, hidden_features,bias=bias)) if spectral else nn.Linear(hidden_features+time_features, hidden_features,bias=bias)
        self.linear2 = spectral_norm(nn.Linear(hidden_features, hidden_features,bias=bias)) if spectral else nn.Linear(hidden_features, hidden_features,bias=bias)
        self.act=lambda x: x*torch.nn.functional.sigmoid(x)
        self.bn=nn.BatchNorm1d(hidden_features,track_running_stats=True) if batch_norm else nn.Identity()
        self.bn2=nn.BatchNorm1d(hidden_features,track_running_stats=True) if batch_norm else nn.Identity()
    def forward(self, x: torch.Tensor,t=None):


        x=self.linear(self.act(self.bn(x))) if t is None else self.linear(self.act(torch.cat((self.bn(x),t),dim=-1)))
        x = self.linear2(self.act(self.bn2(x)))
        return x

