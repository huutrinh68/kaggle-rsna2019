# common library --------------------------------
from src.utils.logger import log
from src.utils.include import *

# import other ----------------------------------
import src.factory as factory
import src.dataset.util as util



def apply_dataset_policy(df, policy):

    if policy == 'all':
        pass
    elif policy == 'pos==neg':
        df_positive = df[df.labels != '']
        df_negative = df[df.labels != '']
        df_sampled  = df_negative.sample(len(df_positive))
        df          = pd.concat([df_positive, df_sampled], sort=False)

    else:
        raise
    log.info(f'apply_dataset_policy: {policy}')  

    return df




##
class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, folds):
        self.cfg = cfg
        self.folds = folds

        self.transforms = factory.get_transforms(self.cfg)

    #     with open(cfg.annotations, 'rb') as f:
    #         self.df = pickle.load(f)

    #     if folds:
    #         self.df = self.df[self.df.fold.isin(folds)]

    #     self.df = apply_dataset_policy(self.df, self.cfg.apply_dataset_policy)
    
    # def __len__(self):
    #     return len(self.df)

    # def __getitem__(self, idx):
    #     row = self.df.iloc[idx]
        
    #     path = '%s/%s.dcm'%(self.cfg.imgdir, row.ID)
        
    #     dicom = pydicom.dcmread(path)
    #     image = dicom.pixel_array
    #     image = 


    # def __str__(self):
    def __repr__(self):
        str = ''
        str += f'use default(random) sampler\t'
        str += f'folds: {self.folds}\t'
        # str += f'len: {len(self.df)}'
        return str

    


__all__ = ['CustomDataset']