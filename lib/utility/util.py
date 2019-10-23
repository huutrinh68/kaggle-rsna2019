from src.common import *


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr(optim):
    if optim:
        return optim.param_groups[0]['lr']
    else:
        return 0


# evaluate meters
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_model(model, optim, detail, fold, dirname, log):
    path = os.path.join(dirname, 'fold%d_ep%d.pt' % (fold, detail['epoch']))
    torch.save({
        'model': model.state_dict(),
        'optim': optim.state_dict(),
        'detail': detail,
    }, path)
    log.write('saved model to %s\n' % path)


def load_model(path, model, optim=None):

    # remap everthing onto CPU 
    state = torch.load(str(path), map_location=lambda storage, location: storage)

    model.load_state_dict(state['model'])
    if optim:
        log('loading optim too')
        optim.load_state_dict(state['optim'])
    else:
        log('not loading optim')

    model.cuda()

    detail = state['detail']
    log('loaded model from %s' % path)

    return detail
