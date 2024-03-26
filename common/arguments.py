import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser(description='Distillation-Policy-Optimization')
    parser.add_argument(
        '--learner',
        default='PPO',
        help='on policy agent: (default: PPO)')
    parser.add_argument(
        '--lr',
        type=float,
        default=3e-4,
        help='learning rate (default: 3e-4)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--uae-lambda',
        type=float,
        default=0.95,
        help='uae lambda parameter (default: 0.95)')    
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=1,
        help='how many training CPU processes to use (default: 1)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=2048,
        help='number steps per update (default: 2048)')
    parser.add_argument(
        '--critic-updates',
        type=int,
        default=10,
        help='number of updates of critic for DPO(TRPO) (default: 10)')
    parser.add_argument(
        '--baseline-updates',
        type=int,
        default=12,
        help='number of updates of baseline (default: 12)')
    parser.add_argument(
        '--dpo-epoch',
        type=int,
        default=10,
        help='number of dpo epochs (default: 10)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=8,
        help='number of batches for ppo (default: 8)')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='size of batch sampled from replay buffer (default: 256)')    
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--tau',
        type=float,
        default=0.005,
        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument(
        '--nu',
        type=float,
        default=0.3,
        help='Interpolating parameter for advantage estimation (default: 0.3)')
    parser.add_argument(
        '--omega',
        type=float,
        default=0.7,
        help='Interpolating parameter for policy gradients (default: 0.7)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.03,
        help='Temperature parameter α of the entropy augmentation (default: 0.03)') 
    parser.add_argument(
        '--num-samples',
        type=int,
        default=30,
        help='number of samples for approximating baseline (default: 30)')
    parser.add_argument(
        '--critic-samples',
        type=int,
        default=25,
        help='number of samples for approximating baseline (default: 25)')
    parser.add_argument(
        '--update-critic-interval',
        type=int,
        default=1,
        help='log interval, one log per n updates (default: 2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=2,
        help='log interval, one log per n updates (default: 2)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=2,
        help='eval interval, one eval per n updates (default: 2)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=1e6,
        help='number of environment steps to train (default: 1e6)')
    parser.add_argument(
        '--env-name',
        default='Walker2d-v3',
        help='environment to train on (default: Walker2d-v3)')
    parser.add_argument(
        '--log-dir',
        default='/tmp/gym/',
        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--max-kl',
        type=float,
        default=0.1,
        help='max kl divergence radius (default: 0.1)')
    parser.add_argument(
        '--damping',
        type=float,
        default=1e-1,
        help='damping (default: 1e-1)')
    parser.add_argument(
        '--UTD',
        type=int,
        default=4,
        help='UTD ratio (default: 4)')    
        
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.learner in ['PPO', 'A2C', 'TRPO']
    
    return args
