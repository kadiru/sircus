"""Deterministic TRAIN-only behavioral cloning before residual terminal PPO."""
import numpy as np
import torch
from .benchmark_policy import ResidualPolicy
from .learning_policy import weight_hash

def initialized(seed):
    torch.manual_seed(seed)
    model=ResidualPolicy()
    with torch.no_grad(): model.logstd.fill_(-3.)
    return model

def bootstrap(model,rows,teachers,*,actor_steps=800,critic_steps=400):
    if not rows or not teachers or any(r.get('role')!='train' for r in rows+teachers):
        raise ValueError('Only nonempty TRAIN data may bootstrap a policy')
    x=torch.tensor(np.asarray([r['observation'] for r in teachers]),dtype=torch.float32)
    y=torch.tensor(np.asarray([r['normalized_residual'] for r in teachers]),dtype=torch.float32)
    xx=torch.tensor(np.asarray([r['observation'] for r in rows]),dtype=torch.float32)
    yy=torch.tensor([r['learning_reward'] for r in rows],dtype=torch.float32)
    if (x.ndim!=2 or x.shape[1]!=9 or y.shape!=(len(teachers),5) or xx.shape!=(len(rows),9)
        or not all(torch.isfinite(t).all() for t in (x,y,xx,yy)) or bool((y.abs()>1).any())
        or any(type(n) is not int or n<0 for n in (actor_steps,critic_steps))):
        raise ValueError('Expected finite nine-feature residual demonstrations')
    actor=torch.optim.Adam(model.actor.parameters(),lr=.001)
    critic=torch.optim.Adam(model.critic.parameters(),lr=.001)
    before=weight_hash(model)
    aloss=lambda:((model.actor(x).tanh()-y)**2).mean()
    closs=lambda:((model.critic(xx).squeeze(-1)-yy)**2).mean()
    initial_actor=float(aloss().detach());initial_critic=float(closs().detach())
    for _ in range(actor_steps):
        loss=aloss();actor.zero_grad();loss.backward();actor.step()
    for _ in range(critic_steps):
        loss=closs();critic.zero_grad();loss.backward();critic.step()
    return dict(before=before,after=weight_hash(model),actor_steps=actor_steps,critic_steps=critic_steps,
        initial_actor_loss=initial_actor,final_actor_loss=float(aloss().detach()),
        initial_critic_loss=initial_critic,final_critic_loss=float(closs().detach()),
        actor_teacher_ids=[r['id'] for r in teachers],critic_row_count=len(rows),
        ordering='Literal teachers sorted by context; all TRAIN rows method/seed/train_index; full-batch, no minibatch sampling')
