model=model001
mode=train
gpu=0
fold=0
conf=./conf/${model}.py
n_tta=1

python -m src.train ${mode} ${conf} --fold ${fold} --gpu ${gpu} --n_tta ${n_tta}
