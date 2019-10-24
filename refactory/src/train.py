# common library --------------------------------
from src.utils.logger import logger, log
logger.setup('./workdir', name='log')
from src.utils.include import *
from src.utils.common import *
from src.utils.config import *
from src.utils.util import *

# add library if you want -----------------------
import src.factory as factory
#  

# common setting --------------------------------
os.environ['CUDA_VISIBLE_DEVICES']='0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# get args from command line
def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'valid', 'test'])
    parser.add_argument('config')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--snapshot')
    parser.add_argument('--output', type=str, default='./se_resnext50_32x4d') 
    parser.add_argument('--n_tta', default=1, type=int)
    
    return parser.parse_args()

def show_config(cfg):
    log.info('\n')
    log.info('---show config---')
    log.info(f'workdir:     {cfg.workdir}')
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
    log.info(f'normalize:   {cfg.normalize}')

    log.info(f'debug:       {cfg.debug}')
    log.info(f'n_tta:       {cfg.n_tta}')
    log.info(f'snapshot:    {cfg.snapshot}')

# do train and do valid -------------------------
def run_train():
    log.info('---start run_train---')
    args = get_args()
    cfg  = Config.fromfile(args.config)
    
    # copy command line args to cfg
    cfg.mode = args.mode
    cfg.debug = args.debug
    cfg.fold = args.fold
    cfg.snapshot = args.snapshot
    cfg.output = args.output
    cfg.n_tta = args.n_tta
    cfg.gpu = args.gpu

    # print setting
    show_config(cfg)

    # torch.cuda.set_device(cfg.gpu)
    set_seed(cfg.seed)

    ## model ------------------------------------
    model = factory.get_model(cfg)
    model.to(device)



if __name__ == "__main__":
    try:
        run_train()
    except KeyboardInterrupt:
        print('Keyboard Interrupted')