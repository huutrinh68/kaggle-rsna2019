from src.common import *
from src.cnn.dataset.custom_dataset import CustomDataset

def get_model(cfg, log):

    log.write(f'\t model:      {cfg.model.name}\n')
    log.write(f'\t pretrained: {cfg.model.pretrained}\n')
    log.write(f'\t n_output:   {cfg.model.n_output}\n')

    if cfg.model.name in ['resnext101_32x8d_wsl']:
        model = torch.hub.load('facebookresearch/WSL-Images', cfg.model.name)
        model.fc = torch.nn.Linear(2048, cfg.model.n_output)
        return model
    try:
        model_func = pretrainedmodels.__dict__[cfg.model.name]
    except KeyError as e:
        model_func = eval(cfg.model.name)

    model = model_func(num_classes=1000, pretrained=cfg.model.pretrained)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Linear(
        model.last_linear.in_features,
        cfg.model.n_output,
    )

    return model



def get_loss(cfg, log):

    #loss = getattr(nn, cfg.loss.name)(**cfg.loss.params)
    loss = getattr(nn, cfg.loss.name)(weight=torch.FloatTensor([2,1,1,1,1,1]).cuda(), **cfg.loss.params)
    log.write('\t loss: %s\n' % cfg.loss.name)

    return loss


def get_optim(cfg, parameters, log):

    optim = getattr(torch.optim, cfg.optim.name)(parameters, **cfg.optim.params)
    log.write('\t optim: %s\n' % cfg.optim.name)

    return optim


def get_dataloader(cfg, folds=None, log=None):

    dataset = CustomDataset(cfg, folds)
    log.write('use default(random) sampler\n')
    loader = DataLoader(dataset, **cfg.loader)

    return loader



def get_transforms(cfg):

    def get_object(transform):
        if hasattr(A, transform.name):
            return getattr(A, transform.name)
        else:
            return eval(transform.name)
    transforms = [get_object(transform)(**transform.params) for transform in cfg.transforms]

    return A.Compose(transforms)
    


def get_scheduler(cfg, optim, last_epoch, log):

    if cfg.scheduler.name == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optim,
            **cfg.scheduler.params,
        )
        scheduler.last_epoch = last_epoch
    else:
        scheduler = getattr(lr_scheduler, cfg.scheduler.name)(
            optim,
            last_epoch=last_epoch,
            **cfg.scheduler.params,
        )
    log.write(f'last_epoch: {last_epoch}\n')

    return scheduler