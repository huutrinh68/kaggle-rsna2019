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
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# get args from command line
def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'valid', 'test'])
    parser.add_argument('config')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--snapshot')
    parser.add_argument('--output') 
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



# test ------------------------------------------
def do_test(cfg, model):
    assert cfg.output
    load_model(cfg.snapshot, model)
    loader_test = factory.get_dataloader(cfg.data.test)
    with torch.no_grad():
        results = [run_nn(cfg.data.test, 'test', model, loader_test) for i in range(cfg.n_tta)]
    with open(cfg.output, 'wb') as f:
        pickle.dump(results, f)
    log.info('saved to %s' % cfg.output)



# valid -----------------------------------------
def do_valid(cfg, model):
    assert cfg.output
    criterion = factory.get_loss(cfg)
    load_model(cfg.snapshot, model)
    loader_valid = factory.get_dataloader(cfg.data.valid, [cfg.fold])
    with torch.no_grad():
        results = [run_nn(cfg.data.valid, 'valid', model, loader_valid, criterion=criterion) for i in range(cfg.n_tta)]
    with open(cfg.output, 'wb') as f:
        pickle.dump(results, f)
    log.info('saved to %s' % cfg.output)



# do train and do valid -------------------------
def run_train():
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
    cfg.device = device

    # print setting
    show_config(cfg)

    # torch.cuda.set_device(cfg.gpu)
    set_seed(cfg.seed)

    # setup -------------------------------------
    for f in ['checkpoint', 'train', 'valid', 'test', 'backup']: os.makedirs(cfg.workdir+'/'+f, exist_ok=True)
    if 0: #not work perfect
        backup_project_as_zip(PROJECT_PATH, cfg.workdir+'/backup/code.train.%s.zip'%IDENTIFIER)

    ## model ------------------------------------
    log.info('\n')
    log.info('** model setting **')
    model = factory.get_model(cfg)

    # multi-gpu----------------------------------
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    ## ------------------------------------------
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

    # scheduler ---------------------------------
    scheduler = factory.get_scheduler(cfg, optim, best['epoch'])

    if cfg.apex:
        amp.initialize(model, optim, opt_level='O1')

    for epoch in range(best['epoch']+1, cfg.epoch):
        log.info(f'---epoch {epoch}---')
        set_seed(epoch)

        ## train model --------------------------
        run_nn(cfg.data.train, 'train', model, loader_train, criterion=criterion, optim=optim, apex=cfg.apex)
        
        ## valid model --------------------------
        with torch.no_grad():
            val = run_nn(cfg.data.valid, 'valid', model, loader_valid, criterion=criterion)

        detail = {
            'score': val['score'],
            'loss': val['loss'],
            'epoch': epoch,
        }
        if val['loss'] <= best['loss']:
            best.update(detail)

        save_model(model, optim, detail, cfg.fold, os.path.join(cfg.workdir, 'checkpoint'))
            
        log.info('[best] ep:%d loss:%.4f score:%.4f' % (best['epoch'], best['loss'], best['score']))
            
        scheduler.step(val['loss']) # reducelronplateau
        # scheduler.step()



## train model
def run_nn(cfg, mode, model, loader, criterion=None, optim=None, scheduler=None, apex=None):
    if mode in ['train']:
        model.train()
    elif mode in ['valid', 'test']:
        model.eval()
    else:
        raise 

    t1 = time.time()
    losses = []
    ids_all = []
    targets_all = []
    outputs_all = []

    for i, (inputs, targets, ids) in enumerate(loader):

        batch_size = len(inputs)

        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)

        if mode in ['train', 'valid']:
            loss = criterion(outputs, targets)
            with torch.no_grad():
                losses.append(loss.item())

        if mode in ['train']:
            if apex:
                with amp.scale_loss(loss, optim) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward() # accumulate loss
            if (i+1) % cfg.n_grad_acc == 0:
                optim.step() # update
                optim.zero_grad() # flush
            
        with torch.no_grad():
            ids_all.extend(ids)
            targets_all.extend(targets.cpu().numpy())
            outputs_all.extend(torch.sigmoid(outputs).cpu().numpy())
            #outputs_all.append(torch.softmax(outputs, dim=1).cpu().numpy())

        elapsed = int(time.time() - t1)
        eta = int(elapsed / (i+1) * (len(loader)-(i+1)))
        progress = f'\r[{mode}] {i+1}/{len(loader)} {elapsed}(s) eta:{eta}(s) loss:{(np.sum(losses)/(i+1)):.6f} loss200:{(np.sum(losses[-200:])/(min(i+1,200))):.6f} lr:{get_lr(optim):.2e}'
        print(progress, end='')
        sys.stdout.flush()

    result = {
        'ids': ids_all,
        'targets': np.array(targets_all),
        'outputs': np.array(outputs_all),
        'loss': np.sum(losses) / (i+1),
    }

    if mode in ['train', 'valid']:
        result.update(calc_auc(result['targets'], result['outputs']))
        result.update(calc_logloss(result['targets'], result['outputs']))
        result['score'] = result['logloss']

        log.info(progress + ' auc:%.4f micro:%.4f macro:%.4f' % (result['auc'], result['auc_micro'], result['auc_macro']))
        log.info('%.6f %s' % (result['logloss'], np.round(result['logloss_classes'], 6)))
    else:
        log.info('')

    return result



##
def calc_logloss(targets, outputs, eps=1e-5):
    # for RSNA
    try:
        logloss_classes = [log_loss(np.floor(targets[:,i]), np.clip(outputs[:,i], eps, 1-eps)) for i in range(6)]
    except ValueError as e: 
        logloss_classes = [1, 1, 1, 1, 1, 1]

    return {
        'logloss_classes': logloss_classes,
        'logloss': np.average(logloss_classes, weights=[2,1,1,1,1,1]),
    }



##
def calc_auc(targets, outputs):
    macro = roc_auc_score(np.floor(targets), outputs, average='macro')
    micro = roc_auc_score(np.floor(targets), outputs, average='micro')
    return {
        'auc': (macro + micro) / 2,
        'auc_macro': macro,
        'auc_micro': micro,
    }



##
if __name__ == "__main__":
    try:
        run_train()
    except KeyboardInterrupt:
        print('Keyboard Interrupted')
