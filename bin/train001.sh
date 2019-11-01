model=model001
mode=train
gpu=1
fold=0
conf=./conf/${model}.py
n_tta=1

export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m src.train ${mode} ${conf} --fold ${fold} --gpu ${gpu} --n_tta ${n_tta}
