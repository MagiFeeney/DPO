import os
import time
import numpy as np
from collections import deque
from tqdm import tqdm

import algo
from dpo import DPO
from common import utils
from common.arguments import get_args
from common.envs import make_vec_envs
from common.model import Policy
from common.storage import RolloutStorage
from common.ReplayBuffer import ReplayBuffer
from evaluation import Evaluator

import torch
from torch.utils.tensorboard import SummaryWriter

def main():
    args = get_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    utils.cleanup_log_dir(log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, device)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        hidden_size=256,
        latent_size=256,        
        activation='tanh'
    )
    actor_critic.to(device)

    if args.eval_interval is not None:
        evaluator = Evaluator(actor_critic, args.env_name, args.seed,
                              1, device)

    writer_dir = os.path.join(log_dir, 'runs', 'DPO-test-reward-{}-{}'.format(args.env_name, args.seed))
    writer = SummaryWriter(writer_dir)

    if args.learner == 'PPO':
        learner = algo.PPO(
            actor_critic,
            args.clip_param,
            args.num_mini_batch)
    elif args.learner == 'A2C':
        learner = algo.A2C(
            actor_critic,
            args.num_mini_batch)
    elif args.learner == 'TRPO':
        learner = algo.TRPO(
            actor_critic,
            args.num_mini_batch,
            args.omega,
            args.max_kl,
            args.damping)

    agent = DPO(
        learner,
        actor_critic,
        args.dpo_epoch,
        args.num_samples,
        args.gamma,
        args.tau,
        args.alpha,
        args.omega,
        args.baseline_updates,
        args.critic_updates,
        args.UTD,
        lr=args.lr)
     
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space, args.critic_samples)

    memory = ReplayBuffer(envs.observation_space.shape, envs.action_space, args.num_processes, device)

    episode_rewards = deque(maxlen=10)
        
    state = envs.reset()
    rollouts.states[0].copy_(state)
    rollouts.to(device)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    
    for j in tqdm(range(num_updates)):            
        transform_reward = False
        for step in range(args.num_steps):
            # Sample according to policy
            with torch.no_grad():
                action, log_prob = actor_critic.act(rollouts.states[step])
                
            # state, reward and next state
            state, reward, done, infos = envs.step(action)
                
            # If done then stop bootstrapping
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # Handling timeout due to TimeLimit if exist
            truncated = torch.as_tensor(done &
                                        [info.get("terminal_observation") is not None for info in infos] &
                                        [info.get("TimeLimit.truncated", False) for info in infos])

            if truncated.any():
                indexes = torch.where(truncated)
                terminal_state = state.clone()
                terminal_state[indexes].copy_(torch.cat([torch.as_tensor(infos[index]["terminal_observation"]) for index in indexes]))
                done[indexes] = False # to allow timeout state to bootstrap the real terminal state in off-policy learning
                transform_reward = True
                
                memory.add(rollouts.states[step], action, reward, terminal_state, done)
            else:
                memory.add(rollouts.states[step], action, reward, state, done)

            rollouts.insert(state, action, log_prob, reward, masks, truncated)

            # policy evaluation
            if step % args.update_critic_interval == 0 and memory.size >= args.batch_size:
                agent.update_critic(memory, args.batch_size)

        with torch.no_grad():
            next_action, next_log_prob = actor_critic.sample_action(rollouts.states[-1])
            next_q = actor_critic.sample_from_critic(rollouts.states[-1], next_action, args.critic_samples)
            rollouts.qvalues[-1]   = next_q
            rollouts.log_probs[-1] = next_log_prob

        # baseline fitting
        agent.update_baseline(memory, args.batch_size)

        with torch.no_grad():
            if transform_reward:
                indexes = torch.where(rollouts.truncated)
                next_states = memory.extract_backward("next state", args.num_steps).unsqueeze(1)
                terminal_states = next_states[indexes]
                terminal_actions, _ = actor_critic.sample_action(terminal_states)
                terminal_qvalues = actor_critic.get_q_value(terminal_states, terminal_actions)
                rollouts.rewards[indexes] += args.gamma * terminal_qvalues.unsqueeze(-1) # bootstrap with unseen next state due to timeout

            states = rollouts.states[:-1].squeeze(1)
            actions = rollouts.actions.squeeze(1)
            q = actor_critic.sample_from_critic(states, actions, args.critic_samples)
            b = actor_critic.get_baseline(states, args.num_samples)
            assert rollouts.qvalues[:-1].shape == q.shape, f"{rollouts.qvalues[:-1].shape} != {q.shape}"
            rollouts.qvalues[:-1].copy_(q)
            assert rollouts.baselines.shape == b.view(*b.shape, 1).shape, f"{rollouts.baselines.shape} != {b.view(*b.shape, 1).shape}"
            rollouts.baselines.copy_(b.view(*b.shape, 1))

        rollouts.compute_advantages(args.gamma,
                                    args.uae_lambda,
                                    normalize=True,
                                    nu=args.nu)

        # policy improvement
        agent.update(rollouts, memory, args.batch_size)
        rollouts.after_update()

        if (j + 1) % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j + 1, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards)))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and (j + 1) % args.eval_interval == 0):
            avg_reward = evaluator.eval()
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            writer.add_scalar('eval rewards', avg_reward, total_num_steps)


if __name__ == "__main__":
    main()
