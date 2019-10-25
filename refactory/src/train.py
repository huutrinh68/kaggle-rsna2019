# common library --------------------------------
from src.utils.logger import logger, log
logger.setup('./logs', name='log')

from src.utils.include import *
from src.utils.common import *
from src.utils.config import *
from src.utils.util import *
from src.utils.file import *

# add library if you want -----------------------
import src.factory as factory
#  

# common setting --------------------------------
os.environ['CUDA_VISIBLE_DEVICES']='0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# get args from command line
def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'valid', 'test'])
    parser.add_argument('config')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--n_tta', default=1, type=int)
    
    return parser.parse_args()

def show_config(cfg):
    log.info('---[START %s] %s'%(IDENTIFIER, '-'*32))
    log.info('\n')
    log.info('** show config **')
    log.info(f'workdir:     {cfg.workdir}')
    log.info(f'logpath:     {logger.path}')
    log.info(f'seed:        {cfg.seed}')
    log.info(f'model:       {cfg.model}')
    log.info(f'optim:       {cfg.optim}')
    log.info(f'loss:        {cfg.loss}')
    log.info(f'scheduler:   {cfg.scheduler}')
    log.info(f'mode:        {cfg.mode}')
    log.info(f'fold:        {cfg.fold}')
    log.info(f'epoch:       {cfg.epoch}')
    log.info(f'batch size:  {cfg.batch_size}')
    log.info(f'acc:         {cfg.data.train.n_grad_acc}')
    log.info(f'n_workers:   {cfg.num_workers}')
    log.info(f'apex:        {cfg.apex}')
    log.info(f'imgsize:     {cfg.imgsize}')
    log.info(f'normalize:   {cfg.normalize}')

    log.info(f'debug:       {cfg.debug}')
    log.info(f'n_tta:       {cfg.n_tta}')
    log.info(f'resume_from: {cfg.resume_from}')
    
    # device
    log.info(f'gpu:         {cfg.gpu}')
    log.info(f'device:      {cfg.device}')

# do train and do valid -------------------------
def run_train():
    args = get_args()
    cfg  = Config.fromfile(args.config)
    
    # copy command line args to cfg
    cfg.mode = args.mode
    cfg.debug = args.debug
    cfg.fold = args.fold
    cfg.n_tta = args.n_tta
    cfg.gpu = args.gpu
    cfg.device = device

    # print setting
    show_config(cfg)

    # torch.cuda.set_device(cfg.gpu)
    set_seed(cfg.seed)

    # setup -------------------------------------
    for f in ['checkpint', 'train', 'valid', 'test', 'backup']: os.makedirs(cfg.workdir+'/'+f, exist_ok=True)
    if 0: #not work perfect
        backup_project_as_zip(PROJECT_PATH, cfg.workdir+'/backup/code.train.%s.zip'%IDENTIFIER)

    ## model ------------------------------------
    log.info('\n')
    log.info('** model setting **')
    # model = factory.get_model(cfg)
    # model.to(device)

    ## ------------------------------------------
    model = torch.nn.Linear(3,5)
    if cfg.mode == 'train':
        do_train(cfg, model)
    elif cfg.mode == 'valid':
        do_valid(cfg, model)
    elif cfg.mode == 'test':
        do_test(cfg, model)
    else:
        log.error(f"mode '{cfg.mode}' is not in [train, valid, test]")
        exit(0)


# train model -----------------------------------
def do_train(cfg, model):
    log.info('\n')
    log.info('** start training **')

    # get criterion -----------------------------
    criterion = factory.get_loss(cfg)

    # get optimization --------------------------
    optim = factory.get_optim(cfg, model.parameters())

    # initial -----------------------------------
    best = {
        'loss': float('inf'),
        'score': 0.0,
        'epoch': -1,
    }

    # re-load model -----------------------------
    if cfg.resume_from:
        log.info('\n')
        log.info(f're-load model from {cfg.resume_from}')
        detail = load_model(cfg.resume_from, model, optim)
        best.update({
            'loss': detail['loss'],
            'score': detail['score'],
            'epoch': detail['epoch'],
        })


    # setting dataset ---------------------------
    log.info('\n')
    log.info('** dataset **')
    folds = [fold for fold in range(cfg.n_fold) if cfg.fold != fold]
    log.info(f'fold_train:    {folds}')
    log.info(f'fold_valid:    [{cfg.fold}]')

    loader_train = factory.get_dataloader(cfg.data.train, folds)
    loader_valid = factory.get_dataloader(cfg.data.valid, [cfg.fold])
    log.info(loader_train)
    log.info(loader_valid)







if __name__ == "__main__":
    try:
        run_train()
    except KeyboardInterrupt:
        print('Keyboard Interrupted')
