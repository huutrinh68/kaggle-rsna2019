
from src.utils.logger import log
from src.utils.include import *

def get_model(cfg):
    log.info('\n')
    log.info('---load model---')
    log.info(f'model:       {cfg.model.name}')
    log.info(f'pretrained:  {cfg.model.pretrained}')
    
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