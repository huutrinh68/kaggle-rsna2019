import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from common  import *
# from dataset import *
# from model   import *



def run_train():
    out_dir = 'resnet152'

    initial_chckpoint = 'resnet152/checkpoint/top1.pth'



    ## setup --------------------------------------------------------------------------------------
    for f in ['checkpoint', 'train', 'valid', 'backup']: os.makedirs(out_dir+'/'+f, exist_ok=True)
    # backup_project_as_zip(PROJECT_PATH, out_dir+'/backup/code.train.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')

    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')


    ## dataset ------------------------------------------------------------------------------------
    log.write('** dataset setting **\n')


if __name__ == "__main__":
    print('%s: calling main function ...'% os.path.basename(__file__))

    run_train()