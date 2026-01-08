from torch.optim import SGD
from .dot import DistillationOrientedTrainer

def get_optimizer(
    net,
    opt_name='sam',
    opt_hyperpara={}):
    if opt_name == 'sgd':
        return SGD(net.parameters(), **opt_hyperpara)
    elif opt_name == 'dot':
        return DistillationOrientedTrainer(net.parameters(), **opt_hyperpara)
    else:
        raise ValueError("Invalid optimizer!!!")