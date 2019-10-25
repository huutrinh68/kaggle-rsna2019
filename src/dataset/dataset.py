# common library --------------------------------
from src.utils.logger import log
from src.utils.include import *

# import other ----------------------------------
import src.factory as factory
import src.dataset.misc as misc
import src.dataset.mappings as mappings


## 
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
def apply_window_policy(image, row, policy):
    if policy == 1:
        image1 = misc.apply_window(image, 40, 80) # brain
        image2 = misc.apply_window(image, 80, 200) # subdural
        image3 = misc.apply_window(image, row.WindowCenter, row.WindowWidth)
        image1 = (image1 - 0) / 80
        image2 = (image2 - (-20)) / 200
        image3 = (image3 - image3.min()) / (image3.max()-image3.min())
        image = np.array([
            image1 - image1.mean(),
            image2 - image2.mean(),
            image3 - image3.mean(),
        ]).transpose(1,2,0)
    elif policy == 2:
        image1 = misc.apply_window(image, 40, 80) # brain
        image2 = misc.apply_window(image, 80, 200) # subdural
        image3 = misc.apply_window(image, 40, 380) # bone
        image1 = (image1 - 0) / 80
        image2 = (image2 - (-20)) / 200
        image3 = (image3 - (-150)) / 380
        # image = np.array([
        #     image1 - image1.mean(),
        #     image2 - image2.mean(),
        #     image3 - image3.mean(),
        # ]).transpose(1,2,0)
        image = np.array([
            image1,
            image2,
            image3,
        ]).transpose(1,2,0)
    else:
        raise

    return image



##
class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, folds):
        self.cfg = cfg
        self.folds = folds

        self.transforms = factory.get_transforms(self.cfg)

        with open(cfg.annotations, 'rb') as f:
            log.info(f'loaded file: {cfg.annotations}')
            self.df = pickle.load(f)

        if folds:
            self.df = self.df[self.df.fold.isin(folds)]

        self.df = apply_dataset_policy(self.df, self.cfg.dataset_policy)
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        path = '%s/%s.dcm' % (self.cfg.imgdir, row.ID)

        dicom = pydicom.dcmread(path)
        image = dicom.pixel_array
        image = misc.rescale_image(image, row.RescaleSlope, row.RescaleIntercept)
        image = apply_window_policy(image, row, self.cfg.window_policy)

        image = self.transforms(image=image)['image']

        target = np.array([0.0] * len(mappings.label_to_num))
        for label in row.labels.split():
            cls = mappings.label_to_num[label]
            target[cls] = 1.0

        if hasattr(self.cfg, 'spread_diagnosis'):
            for label in row.LeftLabel.split() + row.RightLabel.split():
                cls = mappings.label_to_num[label]
                target[cls] += self.cfg.propagate_diagnosis
        target = np.clip(target, 0.0, 1.0)

        return image, torch.FloatTensor(target), row.ID

    # def __str__(self):
    def __repr__(self):
        str = ''
        str += f'use default(random) sampler\t'
        str += f'folds: {self.folds}\t'
        str += f'len: {len(self.df)}'
        return str

    


__all__ = ['CustomDataset']