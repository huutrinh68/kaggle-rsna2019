import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from src.common import *
import src.cnn.factory as factory


def run_train():
    args = get_args()
    cfg  = Config.fromfile(args.config)

    ## copy command line args to cfg
    cfg.mode    = args.mode
    cfg.debug   = args.debug
    cfg.fold    = args.fold
    cfg.snapshot= args.snapshot
    cfg.out_dir = args.out_dir
    cfg.n_tta   = args.n_tta
    cfg.gpu     = args.gpu


    ## 
    torch.cuda.set_device(cfg.gpu)
    set_seed(cfg.seed)


    ## setup --------------------------------------------------------------------------------------
    initial_chckpoint = cfg.out_dir+'/checkpoint/top1.pth'
    for f in ['checkpoint', 'train', 'valid', 'backup']: os.makedirs(cfg.out_dir+'/'+f, exist_ok=True)
    if 0:
        backup_project_as_zip(PROJECT_PATH, cfg.out_dir+'/backup/code.train.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(cfg.out_dir+f'/log.train{cfg.fold}.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')

    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % cfg.out_dir)
    log.write('\n')


    ## dataset ------------------------------------------------------------------------------------
    log.write('** dataset setting **\n')


    ## model --------------------------------------------------------------------------------------
    log.write('** model setting **\n')
    model = factory.get_model(cfg, log)
    model.cuda()


    if cfg.mode == 'train':
        do_train(cfg, model, log)
    elif cfg.mode == 'valid':
        do_valid(cfg, model, log)
    elif cfg.mode == 'test':
        do_test(cfg, model, log)



def do_train(cfg, model, log):
    criterion = factory.get_loss(cfg, log)
    optim = factory.get_optim(cfg, model.parameters(), log)

    best = {
        'loss': float('inf'),
        'score': 0.0,
        'epoch': -1,
        'fold': None,
    }

    ##TODO: load model from checkpoint path
    if cfg.resume_from:
        detail = load_model(cfg.resume_from, model, optim=optim)
        best.update({
            'loss': detail['loss'],
            'score': detail['score'],
            'epoch': detail['epoch'],
            'fold': detail['fold'],
        })
    
    folds = [fold for fold in range(cfg.n_fold) if cfg.fold != fold]
    loader_train = factory.get_dataloader(cfg.data.train, folds, log)
    loader_valid = factory.get_dataloader(cfg.data.valid, [cfg.fold], log)

    log.write(f'\t train data: loaded %d records\n' % len(loader_train.dataset))
    log.write(f'\t valid data: loaded %d records\n' % len(loader_valid.dataset))

    scheduler = factory.get_scheduler(cfg, optim, best['epoch'], log)

    log.write('apex %s\n' % cfg.apex)
    if cfg.apex:
        amp.initialize(model, optim, opt_level='O1')
    
    for epoch in range(best['epoch']+1, cfg.epoch):
        log.write(f'---------epoch {epoch}---------\n')
        set_seed(epoch)

        run_nn(cfg.data.train, 'train', model, loader_train, criterion=criterion, optim=optim, apex=cfg.apex, log=log)
        with torch.no_grad():
            val = run_nn(cfg.data.valid, 'valid', model, loader_valid, criterion=criterion, log=log)

        detail = {
            'score': val['score'],
            'loss': val['loss'],
            'epoch': epoch,
            'fold': cfg.fold,
        }
        if val['loss'] <= best['loss']:
            best.update(detail)

        save_model(model, optim, detail, cfg.fold, cfg.out_dir+'/checkpoint', log)

        log.write('[best] ep:%d loss:%.4f score:%.4f\n' % (best['epoch'], best['loss'], best['score']))

        #scheduler.step(val['loss']) # reducelronplateau
        scheduler.step()

def run_nn(cfg, mode, model, loader, criterion=None, optim=None, scheduler=None, apex=None, log=None):
    if mode in ['train']:
        model.train()
    elif mode in ['valid', 'test']:
        model.eval()
    else:
        raise 

    t1 = time.time()
    # losses = []
    ids_all = []
    targets_all = []
    outputs_all = []
    losses = AverageMeter()

    for i, (inputs, targets, ids) in enumerate(loader):
        # zero out gradients so we can accumulate new ones over batches
        optimizer.zero_grad()

        batch_size = len(inputs)

        inputs = inputs.cuda()
        targets = targets.cuda()
        outputs = model(inputs)

        if mode in ['train', 'valid']:
            loss = criterion(outputs, targets)
            with torch.no_grad():
                losses.update(loss.item())

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
        progress = f'\r[{mode}] {i+1}/{len(loader)} {elapsed}(s) eta:{eta}(s) loss:{losses.avg):.6f} lr:{get_lr(optim):.2e}'
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

        log.write(progress + ' auc:%.4f micro:%.4f macro:%.4f\n' % (result['auc'], result['auc_micro'], result['auc_macro']))
        log.write('%.6f %s\n' % (result['logloss'], np.round(result['logloss_classes'], 6)))
    else:
        log.write('\n')

    return result


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


def calc_auc(targets, outputs):
    macro = roc_auc_score(np.floor(targets), outputs, average='macro')
    micro = roc_auc_score(np.floor(targets), outputs, average='micro')
    return {
        'auc': (macro + micro) / 2,
        'auc_macro': macro,
        'auc_micro': micro,
    }


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'valid', 'test'])
    parser.add_argument('config')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--snapshot')
    parser.add_argument('--out_dir', type=str, default='./se_resnext50_32x4d') 
    parser.add_argument('--n-tta', default=1, type=int)
    
    return parser.parse_args()

if __name__ == "__main__":
    print('%s: calling main function ...'% os.path.basename(__file__))

    try:
        run_train()
    except KeyboardInterrupt:
        print('Keyboard Interrupted')
