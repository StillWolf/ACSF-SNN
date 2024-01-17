import argparse
import numpy as np
import torch
import gym
from algorithms import SpikingBCQ
from algorithms import BCQ_AEAD
from algorithms import DDPG
from algorithms import TD3
from algorithms import OriBCQ


def eval_policy(policy, env_name, seed, eval_episodes=10):
    reward_all = torch.zeros(10)
    for i in range(10):
        eval_env = gym.make(env_name)
        eval_env.reset(seed=(seed+i*10))
        avg_reward = 0.
        for _ in range(eval_episodes):
            state, done = eval_env.reset(), False
            while not done:
                action = policy.select_action(np.array(state))
                state, reward, done, _ = eval_env.step(action)
                avg_reward += reward

        avg_reward /= eval_episodes
        reward_all[i] = avg_reward

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {torch.mean(reward_all):.3f} ± {torch.std(reward_all)}")
    print("---------------------------------------")
    print(reward_all)
    return torch.mean(reward_all)


if __name__ == "__main__":
    # Load parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Ant-v3")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--lmbda", default=0.75)  # Weighting for clipped double Q-learning in BCQ
    parser.add_argument("--phi", default=0.05)  # Max perturbation hyper-parameter for BCQ
    parser.add_argument("--mode", default="DDPG", type=str)
    args = parser.parse_args()
    env = gym.make(args.env)

    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    device = torch.device("cuda:" + args.gpu.__str__() if torch.cuda.is_available() else "cpu")

    if args.mode == "Spiking":
        print("Test Spiking")
        policy = SpikingBCQ.BCQ(state_dim, action_dim, max_action, device, args.discount, args.tau, args.lmbda, args.phi)
        filename = "./models/BCQ_Spiking_"+args.env+"_"+args.seed.__str__()+"_Accum"
        policy.load(filename)
        eval_policy(policy, args.env, args.seed)
    elif args.mode == "AEAD":
        print("Test AutoEncoderAutoDecoder")
        policy = BCQ_AEAD.BCQ(state_dim, action_dim, max_action, device, args.discount, args.tau, args.lmbda,
                                args.phi)
        filename = "./models/BCQ_AEAD_" + args.env + "_" + args.seed.__str__() + "_Robust_" + args.env + "_" + args.seed.__str__()
        policy.load(filename)
        eval_policy(policy, args.env, args.seed)
    elif args.mode == "DDPG":
        print("Test DDPG")
        policy = DDPG.DDPG(state_dim, action_dim, max_action, device, args.discount, args.tau)
        filename = "./models/behavioral_" + args.env + "_" + args.seed.__str__()
        policy.load(filename)
        eval_policy(policy, args.env, args.seed + 10)
    elif args.mode == "TD3":
        print("Test TD3")
        kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_action": max_action,
            "device": device,
            "discount": args.discount,
            "tau": args.tau,
            "policy_noise": 0.2 * max_action,
            "noise_clip": 0.5 * max_action,
            "policy_freq": 2,
        }
        policy = TD3.TD3(**kwargs)
        filename = "./models/TD3_" + args.env + "_" + args.seed.__str__()
        policy.load(filename)
        eval_policy(policy, args.env, args.seed)
    elif args.mode == "BCQ":
        print("Test BCQ")
        policy = OriBCQ.BCQ(state_dim, action_dim, max_action, device, args.discount, args.tau, args.lmbda, args.phi)
        filename = "./models/BCQ_BCQ_" + args.env + "_" + args.seed.__str__() + "_Robust_" + args.env + "_" + args.seed.__str__()
        policy.load(filename)
        eval_policy(policy, args.env, args.seed)
