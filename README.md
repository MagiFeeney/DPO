# Distillation Policy Optimization
Pytorch implementation of Distillation Policy Optimization (DPO), a general learning framework for a set of on-policy algorithms, with off-policy data fully engaged.

### How To Use
#### PPO
```
python main.py --env-name "Walker2d-v3" --learner PPO --clip-param 0.2 --log-dir "logs" --seed 0 --log-interval 2 --eval-interval 2 --num-steps 2048 --num-processes 1 --lr 3e-4 --dpo-epoch 10 --num-mini-batch 8 --gamma 0.99 --uae-lambda 0.95 --num-env-steps 1000000
```	  	     

#### A2C
```
python main.py --env-name "Walker2d-v3" --learner A2C --log-dir "logs" --seed 0 --log-interval 16 --eval-interval 16 --num-steps 256 --num-processes 1 --lr 3e-4 --dpo-epoch 1 --num-mini-batch 1 --gamma 0.99 --uae-lambda 0.95 --num-env-steps 1000000 --baseline-updates 4
```

#### TRPO
```
python main.py --env-name "Walker2d-v3" --learner TRPO --max-kl 0.1 --damping 0.1 --log-dir "logs" --seed 0 --log-interval 1 --eval-interval 1 --num-steps 4096 --num-processes 1 --lr 3e-4 --dpo-epoch 1 --num-mini-batch 1 --gamma 0.99 --uae-lambda 0.95 --num-env-steps 1000000
```
### Requirements
It requires MuJoCo pre-installed, see more [instructions](https://github.com/openai/mujoco-py#install-mujoco). A suggested approach to build the environment is
```
sudo apt-get update; sudo apt-get install make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
# mujoco dependencies
apt-get -y install wget unzip software-properties-common \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev patchelf
# mujoco installation
curl -OL https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
mkdir ~/.mujoco
tar -zxf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco
rm mujoco210-linux-x86_64.tar.gz
```
And other dependencies
```
pip install -r requirements.txt
```

### Citation
```
@misc{ma2023distillation,
      title={Distillation Policy Optimization}, 
      author={Jianfei Ma},
      journal={arXiv preprint arXiv:2302.00533},
      year={2023}
}
```