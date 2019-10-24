model=model001
mode=train
gpu=0
fold=0
conf=./conf/${model}.py
n_tta=1
output=workdir/${model}001

python -m src.train ${mode} ${conf} --fold ${fold} --gpu ${gpu} --output ${output} --n_tta ${n_tta}
