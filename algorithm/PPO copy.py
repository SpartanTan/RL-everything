import argparse
import os
from distutils.util import strtobool
import time
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

import gymnasium as gym
from cartpole import CartPoleEnv

import sys

def make_env(gym_id, seed, idx, capture_video, run_name):
        def thunk():
            # env = CartPoleEnv()
            env = gym.make(gym_id, render_mode="rgb_array")
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if capture_video:
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            # env.seed(args.seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std) # fills the input tensor with a (semi) orthogonal matrix; std is the standard deviation or gain
    nn.init.constant_(layer.bias, bias_const) # fills the input tensor with the value bias_const
    return layer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=os.path.basename(__file__).rstrip(".py"), help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="CartPole-v1", help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4, help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1, help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=25000, help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True, help='if toggled, cuda will not be enabled by default')
    # parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True, help='if toggled, the experiment will be tracked with Weights and Biases')
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="weather to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument('--num-envs', type=int, default=4, help="the number of parallel game environments to run")
    parser.add_argument('--num-steps', type=int, default=128, help="the number of steps per game environment to run")
    parser.add_argument('--anneal-lr', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True, help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--gae', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True, help="Use GAE for advantage estimation")
    parser.add_argument('--gamma', type=float, default=0.99, help="discount factor gamma")
    parser.add_argument('--gae-lambda', type=float, default=0.95, help="lambda for GAE")
    parser.add_argument('--num-minibatches', type=int, default=4, help="the number of mini-batches per update")
    parser.add_argument('--update-epochs', type=int, default=4, help="the K epochs to update the policy")
    parser.add_argument('--norm-adv', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True, help="Toggle advantage normalization")
    parser.add_argument('--clip-coef', type=float, default=0.2, help="the surrogate clipping coefficient")
    parser.add_argument('--clip-vloss', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True, help="Toggle clipping of the value function loss")
    parser.add_argument('--ent-coef', type=float, default=0.01, help="coefficient of the entropy loss") # c2 in the paper
    parser.add_argument('--vf-coef', type=float, default=0.5, help="coefficient of the value function loss") # c1 in the paper
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help="the maximum norm for the gradient clipping")
    parser.add_argument('--target-kl', type=float, default=None, help="the target KL divergence threashold")
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps) # 4 * 128 = 512
    args.minibatch_size = int(args.batch_size // args.num_minibatches) # 512 // 4 = 128

    return args

class Agent(nn.Module):
    def __init__(self, envs:gym.vector.SyncVectorEnv):
          super().__init__()
          self.critic = nn.Sequential(
               layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)), # input feature: 4, output feature: 64
               nn.Tanh(),
               layer_init(nn.Linear(64, 64)),
               nn.Tanh(),
               layer_init(nn.Linear(64, 1), std=1.0)
          )
          self.actor = nn.Sequential(
               layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
               nn.Tanh(),
               layer_init(nn.Linear(64, 64)),
               nn.Tanh(),
               layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01) # 0.01 make the probability of taking each action be similar
          )

    def get_value(self, x)->torch.Tensor:
        """
        forward the observation through the critic network
        Torch only support mini-batches. [nsamples x observation_dim]
        ### Parameters:
        - `x`: the observation, [env_num, obs_dim]

        ### Returns:
        - `value`: value of the value function V(x), [env_num, 1]
        """
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """
        ### Parameters: 
        - `x`: the observation, [env_num, obs_dim]
        
        ### Returns:
        - (action, log_prob, entropy, value) \\
        action: [env_num] \\ 
        log_prob: [env_num]\\
        entropy: [env_num] \\
        value: [env_num, 1]  
        """
        logits = self.actor(x) # unnormalized action probabilities. [env_num, action_dim]
        probs = Categorical(logits=logits) # probability distribution over actions
        if action is None:
            action = probs.sample() # [env_num]
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
        



if __name__ == "__main__":
    args = parse_args()
    
    run_name = f"{args.gym_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" %("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic=args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env = CartPoleEnv(render_mode="rgb_array")
    # # env = gym.make("CartPole-v1", render_mode="rgb_array")
    # env = gym.wrappers.RecordEpisodeStatistics(env, 500)
    # # env = gym.wrappers.RecordVideo(env, "videos", episode_trigger=lambda x: x % 100 == 0)

    # observation, info = env.reset() # np.array(self.state), {}
    # # print(info)
    # # action = env.action_space.sample()
    # # print(action)
    # # observation, reward, done, truncated, info = env.step(action)
    # # print(observation, reward, done, truncated, info)
    # episodic_return = 0
    # for _ in range(200):
    #     action = env.action_space.sample()
    #     observation, reward, done, truncated, info = env.step(action)
    #     episodic_return += reward
    #     if done:
    #         print(f"Episodic return: {info['episode']['r']}")
    #         observation, info = env.reset()
    #         episodic_return = 0
    # env.close()


    
    
    # envs = gym.vector.SyncVectorEnv([make_env(args.gym_id, args.seed, 1, False, run_name) for _ in range(4)])
    # observation, info = envs.reset() # np.array(self.state), {}
    # for _ in range(200):
    #     action = envs.action_space.sample()
    #     observation, reward, done, truncated, info = envs.step(action)
    #     if True in done:
    #         for idx, element in enumerate(info['final_info']):
    #             if isinstance(element, dict):
    #                 print(f"Episodic return: {element['episode']['r']}")

    envs = gym.vector.SyncVectorEnv([make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only supports discrete action space"
    print("envs.single_observation_space.shape", envs.single_observation_space.shape) # (4,)
    print("envs.single_action_space.n", envs.single_action_space.n) # 2
    print("envs.single_action_space.shape", envs.single_action_space.shape) #

    
    agent = Agent(envs).to(device)
    # print(agent)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # storge steup
    # Rollouts data: 4*128=512, it mens 4 envs and 128 steps per envs, so 512 steps in one rollout
    # init -> rollout -> collect batch -> policy training -> update agent -> rollout
    # batch size: 512
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device) # [num_steps x num_envs x single_observation_space] -> [128 x 4 x 4]
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device) # [128 x 4]
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device) # [128 x 4]
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device) # [128 x 4]
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device) # [128 x 4]
    values = torch.zeros((args.num_steps, args.num_envs)).to(device) # [128 x 4]

    # global
    global_step = 0
    start_time = time.time()
    # cartpole reset() function returns `self.state.T`
    # `self.state` is of shape(4, num_envs), then transformed to (num_envs, 4)
    #  env1|| obs1 | obs2 | obs3 | obs4
    #  env2|| obs1 | obs2 | obs3 | obs4
    #  env3|| obs1 | obs2 | obs3 | obs4 
    #  env4|| obs1 | obs2 | obs3 | obs4 
    next_obs, info = envs.reset() # notice that `info` is {}
    next_obs = torch.tensor(next_obs).to(device) # [env_num x obs_number] -> [4 x 4] 
    next_done = torch.zeros(args.num_envs).to(device) # [4]
    num_updates = args.total_timesteps // args.batch_size
    print(num_updates)
    # print("next_obs.shape", next_obs.shape)
    # print("agent.get_value(next_obs)", agent.get_value(next_obs))
    # print("agent.get_value(next_obs).shape", agent.get_value(next_obs).shape)

    # print()
    # print("agent.get_action_and_value(next_obs)", agent.get_action_and_value(next_obs))


    for update in range(1, num_updates + 1):
        
        # decresing learning rate, change the parameter of optimizer
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates # update increasing, frac decreasing
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        
        # Policy rollout
        # 4 envs, 128 steps in one rollout
        for step in range(0, args.num_steps):
            # store the initial observation
            global_step += 1 * args.num_envs # 4 steps in global in one step
            obs[step] = next_obs #[1 x 4]
            dones[step] = next_done #[1 x 4]
            
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten() # since the values is of shape [num_steps, num_envs]
            actions[step] = action
            logprobs[step] = logprob
            
            next_obs, reward, done, truncated, info = envs.step(action.cpu().numpy())
            
            rewards[step] = torch.tensor(reward).to(device).view(-1) # store the reward from each step in each env
            next_obs, next_done = torch.tensor(next_obs).to(device), torch.tensor(done).to(device) # [1 x 4]

            # print out and save the episodic return
            if any(done):
                for idx, element in enumerate(info['final_info']):
                    if isinstance(element, dict):
                        print(f"gloal_step={global_step}, Episodic return: {element['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", element['episode']['r'], global_step)
                        writer.add_scalar("charts/episodic_length", element['episode']['l'], global_step)
                        break

        # boostrap values if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1) # from [n_envs x 1]-> [1 x n_envs]
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device) # rewards: [n_steps, n_envs] -> [128 x 4]
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    # td_error = (rewards)
                    if t == args.num_steps - 1: # the last step TD error cannot be calulated via bootstrap
                        # since it is the last step, the "done" signal is from the last step
                        # after this operaion, 1 will be on the position where the last step is not done
                        # The not done episode can be learned
                        next_done = next_done.to(torch.int)
                        nextnonterminal = 1 - next_done
                        # the next value is also from the last step. e.g. 
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1 - dones[t+1]
                        nextvalues = values[t+1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t+1]
                        next_return = returns[t+1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return # regular return: sum of discounted rewards
                advantages = returns - values  # Q-V
        
        # flatten the batch
        # Requirede the second dimension the same as the observation dimension, 4(or might be 4 x someint in other case when observation is larger)
        b_obs = obs.reshape(-1, *obs.shape[2:]) # [512 x 4]. 
        b_logprobs = logprobs.reshape(-1) # compress to 1 dimension [512]
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape) # [512]
        b_advantages = advantages.reshape(-1) # [512]
        b_returns = returns.reshape(-1) # [512]
        b_values = values.reshape(-1) # [512]
        
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        # each update do 4 epochs
        for epoch in range(args.update_epochs):
            # each epoch do 4 minibatches, each minibatch do 128 steps
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                
                _, newlogproba, entropy, newvalues = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                # newvalues: [128 x 1]
                # log(P1/P2) = log(P1) - log(P2)
                logratio = newlogproba - b_logprobs[mb_inds] # new/old
                ratio = logratio.exp() # e^(log(P1/P2)) = P1/P2
                
                with torch.no_grad():
                    # joscha.net/blog/kl-approx.html
                    # KL[old,new]=log(old/new)
                    # but now we have r = new/old
                    # thhus log(old/new) = -log(r)
                    old_approx_kl = (-logratio).mean()
                    # but joscha suggests a better estimator
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean()]
                    
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                # L = E[min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)]
                # Here we use negative loss, so use max() instead of min() in clip
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean() 
                
                # Value loss
                # unclipped loss: L^VF = E[(V - V_target)^2]
                # clipped V: clip(V-V_target, -coef, coef) + V_target
                # clipped loss: (v_clipped - V_target)^2
                newvalues = newvalues.view(-1) # size: [minibatch_size]->[128] e.g. [1.1 -2.3 -1.3 ...]
                if args.clip_vloss:
                    v_loss_unclipped = (newvalues - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalues - b_returns[mb_inds], -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalues - b_returns[mb_inds]) ** 2).mean() # scalar
                    
                
                entropy_loss = entropy.mean()
                # idea: minimize the policy loss and value loss but maximize the entropy
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
                
                # otimization
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
            
            # batch-level early stopping
            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break
            
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y # tells if value function is a good indicator of the returns
        
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]['lr'], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step) # .item() take the scalar value out of the tensor
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("debug/explained_variance", explained_var, global_step)
        writer.add_scalar("debug/advantage_mean", advantages.mean().item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        
    envs.close()
    writer.close()