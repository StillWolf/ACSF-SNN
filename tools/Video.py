import gym
import sys
from gym import wrappers
from time import time
import os
import argparse
import numpy as np
import torch

sys.path.append("..")

from algorithms import BCQ_AEAD

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Hopper-v3")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--buffer", default="Robust", type=str)
    parser.add_argument("--gpu", default=2, type=int)
    args = parser.parse_args()

    if not os.path.exists("./videos"):
        os.makedirs("./videos")

    env = gym.make(args.env)
    env = wrappers.Monitor(env, './videos/' + f"BCQ_AEAD_{args.env}_{args.seed}_{args.buffer}" + '/', force=True)

    env.seed(args.seed)
    env.action_space.seed(args.seed)
    np.random.seed(args.seed)

    model_path = f"./models/BCQ_AEAD_{args.env}_{args.seed}_{args.buffer}"
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    device = torch.device("cuda:" + args.gpu.__str__() if torch.cuda.is_available() else "cpu")

    policy = BCQ_AEAD.BCQ(state_dim, action_dim, max_action, device, T=4)
    policy.load(model_path)
    state, done = env.reset(), False
    tot_reward = 0
    while not done:
        env.render()
        action = policy.select_action(np.array(state))
        state, reward, done, _ = env.step(action)
        tot_reward += reward

    print(tot_reward)
