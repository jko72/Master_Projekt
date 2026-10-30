"""Shared optimizer factory and update-based LMNet warmup/decay."""
import torch


def make_optimizer(params, train):
    name = str(train.get('optimizer', 'adamw')).lower()
    args = dict(lr=float(train['learning_rate']), weight_decay=float(train.get('weight_decay', 0)))
    if name == 'sgd':
        momentum = float(train.get('momentum', .9))
        if not 0 <= momentum < 1:
            raise ValueError('SGD momentum must be in [0,1)')
        return torch.optim.SGD(params, momentum=momentum, **args)
    if name == 'adamw':
        return torch.optim.AdamW(params, **args)
    if name == 'adam':
        return torch.optim.Adam(params, **args)
    raise ValueError(f'Unsupported optimizer: {name}')


class LMNetWarmup:
    """Explicit equivalent of reference warmupLR with modern PyTorch init.

    Reference initializes CyclicLR and advances it once during outer scheduler
    construction. First optimizer update therefore uses lr/warmup_steps.
    The peak is used twice at the transition before exponential decay starts.
    Count only successful optimizer updates; call step AFTER optimizer.step.
    """
    def __init__(self, optimizer, updates_per_epoch, warmup_epochs=1., decay=.99):
        if updates_per_epoch < 1 or warmup_epochs <= 0 or not 0 < decay <= 1:
            raise ValueError('Invalid warmup/decay or updates_per_epoch')
        self.optimizer = optimizer
        self.updates_per_epoch = int(updates_per_epoch)
        self.warmup_steps = max(1, int(warmup_epochs * updates_per_epoch))
        self.decay = float(decay) ** (1 / updates_per_epoch)
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]
        self.completed_updates = 0
        self._apply()

    def _apply(self):
        n = self.completed_updates
        factor = (n+1)/self.warmup_steps if n < self.warmup_steps else self.decay**(n-self.warmup_steps)
        for group, lr in zip(self.optimizer.param_groups, self.base_lrs):
            group['lr'] = lr * factor

    def step(self):
        self.completed_updates += 1
        self._apply()

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k != 'optimizer'}

    def load_state_dict(self, state):
        if state['updates_per_epoch'] != self.updates_per_epoch:
            raise ValueError('Resume changes updates_per_epoch; cannot preserve LR trace')
        self.__dict__.update(state)
        self._apply()

    def get_last_lr(self):
        return [g['lr'] for g in self.optimizer.param_groups]


def make_scheduler(optimizer, train, updates_per_epoch):
    name = str(train.get('scheduler', 'cosine')).lower()
    if name == 'lmnet':
        return LMNetWarmup(optimizer, updates_per_epoch, float(train.get('warmup_epochs', 1)),
                           float(train.get('lr_decay_per_epoch', .99)))
    if name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(train['epochs']),
                                                         eta_min=float(train.get('learning_rate_min', 0)))
    raise ValueError(f'Unsupported scheduler: {name}')
