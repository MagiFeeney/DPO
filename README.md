# Distillation Policy Optimization
Pytorch implementation of Distillation Policy Optimization (DPO) for **discrete domains**, a general learning framework for a set of on-policy algorithms, with off-policy data fully engaged.

### How To Use
We reference the hyperparameters from original PPO paper and repository [Pytorch-PPO](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail):
#### PPO
```
python main.py --env-name "Breakout-v4" --learner PPO --clip-param 0.2 --log-dir "logs" --seed 0 --log-interval 2 --eval-interval 2 --num-steps 128 --num-processes 8 --lr 3e-4 --dpo-epoch 4 --num-mini-batch 4 --gamma 0.99 --uae-lambda 0.95 --num-env-steps 100000
```
### Requirements
```
pip install -r requirements.txt
```