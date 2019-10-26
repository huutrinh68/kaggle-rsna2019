# common library --------------------------------
from src.utils.logger import log
from src.utils.include import *

# import other ----------------------------------
from src.dataset.dataset import CustomDataset
from src.dataset.transforms import RandomResizedCrop

def get_model(cfg):

    log.info(f'model: {cfg.model.name}')
    log.info(f'pretrained: {cfg.model.pretrained}')
    
    if cfg.model.name in ['resnext101_32x8d_wsl']:
        log.info('backbone get from: facebookresearch/WSL-Images')
        model = torch.hub.load('facebookresearch/WSL-Images', cfg.model.name)
        model.fc = torch.nn.Linear(2048, cfg.model.n_output)
        return model

    try:
        model_func = pretrainedmodels.__dict__[cfg.model.name]
        log.info('pretrained weight from: pretrainedmodels')
    except KeyError as e:
        model_func = eval(cfg.model.name)

    model = model_func(num_classes=1000, pretrained=cfg.model.pretrained)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Linear(
        model.last_linear.in_features,
        cfg.model.n_output,
    )
    log.info(f'last layer: {cfg.model.n_output}')

    return model



def get_loss(cfg):

    loss = getattr(nn, cfg.loss.name)(weight=torch.FloatTensor([2,1,1,1,1,1]).to(cfg.device), **cfg.loss.params)
    log.info(f'criterion: {cfg.loss}')

    return loss



def get_optim(cfg, parameters):

    optim = getattr(torch.optim, cfg.optim.name)(parameters, **cfg.optim.params)
    log.info(f'optim: {cfg.optim}')

    return optim



def get_dataloader(cfg, folds=None):

    dataset = CustomDataset(cfg, folds)
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



def get_scheduler(cfg, optim, last_epoch):

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
    log.info(f'last_epoch: {last_epoch}')

    return scheduler
