mkdir -p cache

# train
python ../preprocess/dicom_to_dataframe.py --input ../data/stage_1_train.csv --output ../cache/train_raw.pkl --imgdir ../data/stage_1_train_images
python ../preprocess/create_dataset --input ../cache/train_raw.pkl --output ../cache/train.pkl
python ../preprocess/make_folds --input ../cache/train.pkl --output ../cache/train_folds.pkl --n-fold 5 --seed 42

# test
python ../preprocess/dicom_to_dataframe --input ../data/stage_1_sample_submission.csv --output ../cache/test_raw.pkl --imgdir ../data/stage_1_test_images
python ../preprocess/create_dataset --input ../cache/test_raw.pkl --output ../cache/test.pkl