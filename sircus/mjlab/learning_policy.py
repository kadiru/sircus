"""Scene/constraint-conditioned Gaussian actor critic for terminal-shot PPO."""

import copy
import hashlib
import numpy as np
import torch
from .learning_contract import LOW,HIGH,WARM,latents


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        def net(out):
            return torch.nn.Sequential(torch.nn.Linear(10,128),torch.nn.Tanh(),
                torch.nn.Linear(128,128),torch.nn.Tanh(),torch.nn.Linear(128,out))
        self.actor,self.critic = net(5),net(1)
        self.logstd = torch.nn.Parameter(torch.full((5,),-1.5))
        torch.nn.init.zeros_(self.actor[-1].weight)
        with torch.no_grad():
            self.actor[-1].bias.copy_(torch.tensor(latents(WARM),dtype=torch.float32))

    def distribution(self,x):
        return torch.distributions.Normal(self.actor(x),self.logstd.exp())


def weight_hash(policy):
    h=hashlib.sha256()
    for k,v in sorted(policy.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def update(policy,optimizer,obs,latent,old_logp,rewards,epochs=6):
    with torch.no_grad():
        advantage=rewards-policy.critic(obs).squeeze(-1)
        advantage=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-6)
    old_logp,latent=old_logp.detach(),latent.detach()
    before=weight_hash(policy)
    logs=[]
    for _ in range(epochs):
        dist=policy.distribution(obs)
        ratio=(dist.log_prob(latent).sum(-1)-old_logp).exp()
        actor=-torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean()
        value=((policy.critic(obs).squeeze(-1)-rewards)**2).mean()
        loss=actor+.5*value-.001*dist.entropy().sum(-1).mean()
        if not torch.isfinite(loss): raise RuntimeError("Nonfinite PPO loss")
        optimizer.zero_grad();loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),1.)
        optimizer.step()
        with torch.no_grad(): policy.logstd.clamp_(-3,-.7)
        logs.append(dict(loss=float(loss.detach()),actor_loss=float(actor.detach()),critic_loss=float(value.detach()),gradient_norm=float(norm)))
    return dict(before=before,after=weight_hash(policy),epochs=logs)


def warm_start(policy,obs,values,weights,steps=400):
    """Explicit return-weighted imitation bootstrap from TRAIN exploration only."""
    device=next(policy.parameters()).device
    x=torch.as_tensor(obs,device=device)
    y=torch.tensor(latents(values),device=device,dtype=torch.float32)
    w=torch.tensor(weights,device=device,dtype=torch.float32)
    opt=torch.optim.Adam(policy.actor.parameters(),lr=.001)
    for _ in range(steps):
        loss=(((policy.actor(x)-y)**2).mean(-1)*w).mean()
        opt.zero_grad();loss.backward();opt.step()
    return float(loss.detach())
