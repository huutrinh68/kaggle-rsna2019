model=model001
gpu=0
fold=0
ep=7
tta=5
clip=1e-6
conf=./conf/${model}.py

# snapshot=./model/${model}/fold${fold}_ep${ep}.pt
snapshot=./se_resnext50_32x4d/checkpoint/fold0_ep7.pt
valid=./model/${model}/fold${fold}_ep${ep}_valid_tta${tta}.pkl
test=./model/${model}/fold${fold}_ep${ep}_test_tta${tta}.pkl
sub=./data/submission/${model}_fold${fold}_ep${ep}_test_tta${tta}.csv

python -m src.train test ${conf} --snapshot ${snapshot} --output ${test} --n_tta ${tta} --fold ${fold} --gpu ${gpu}
python -m src.postprocess.make_submission --input ${test} --output ${sub} --clip ${clip}
#kaggle competitions submit rsna-intracranial-hemorrhage-detection -m "" -f ./data/submission/${sub}

