import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common import *
from kaggle import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='provided by kaggle, stage_1_train.csv for stage1')
    parser.add_argument('--output')
    parser.add_argument('--imgdir')
    parser.add_argument('--n-pool', default=10, type=int)
    parser.add_argument('--nrows', default=None, type=int)
    return parser.parse_args()


def group_id_by_label(df):
    ids = {}
    for row in tqdm(df.itertuples(), total=len(df)):
        prefix, id, label = row.ID.split('_')
        id = '%s_%s' % (prefix, id)
        if id not in ids:
            ids[id] = []
        if row.Label == 1: 
            ids[id].append(label)
    return ids


def remove_corrupted_images(ids):
    ids = ids.copy()
    for id in ['ID_6431af929']:
        try:
            ids.pop(id) 
        except KeyError as e:
            print('%s not found' % id)
        else:
            print('removed %s' % id)

    return ids


def create_record(item, dirname):

    id, labels = item

    path = '%s/%s.dcm' % (dirname, id)
    dicom = pydicom.dcmread(path)
    
    record = {
        'ID': id,
        'labels': ' '.join(labels),
        'n_label': len(labels),
    }
    record.update(get_dicom_raw(dicom))

    raw = dicom.pixel_array
    slope = float(record['RescaleSlope'])
    intercept = float(record['RescaleIntercept'])
    center = get_dicom_value(record['WindowCenter'])
    width = get_dicom_value(record['WindowWidth'])

    image = rescale_image(raw, slope, intercept)
    doctor = apply_window(image, center, width)
    custom = apply_window(image, 40, 80)

    record.update({
        'raw_max': raw.max(),
        'raw_min': raw.min(),
        'raw_mean': raw.mean(),
        'raw_diff': raw.max() - raw.min(),
        'doctor_max': doctor.max(),
        'doctor_min': doctor.min(),
        'doctor_mean': doctor.mean(),
        'doctor_diff': doctor.max() - doctor.min(),
        'custom_max': custom.max(),
        'custom_min': custom.min(),
        'custom_mean': custom.mean(),
        'custom_diff': custom.max() - custom.min(),
    })
    return record


def create_df(ids, args):
    print('making records...')
    with Pool(args.n_pool) as pool:
        records = list(tqdm(
            iterable=pool.imap_unordered(
                functools.partial(create_record, dirname=args.imgdir),
                ids.items()
            ),
            total=len(ids),
        ))
    return pd.DataFrame(records).sort_values('ID').reset_index(drop=True)


def main():
    args = get_args()
    df_input = pd.read_csv(args.input, nrows=args.nrows)
    print('read %s (%d records)' % (args.input, len(df_input)))

    ids = group_id_by_label(df_input)
    ids = remove_corrupted_images(ids)

    df_output = create_df(ids, args)

    with open(args.output, 'wb') as f:
        pickle.dump(df_output, f)

    print('converted dicom to dataframe (%d records)' % len(df_output))
    print('saved to %s' % args.output)
    


if __name__ == '__main__':
    print('%s: calling main function from ...'%os.path.basename(__file__))
    print(sys.argv)
    main()