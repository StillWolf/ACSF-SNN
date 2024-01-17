import argparse
import gym
import numpy as np
import os
import torch
from tools import utils

from algorithms import DDPG
from algorithms import TD3
from algorithms import OriBCQ

from algorithms import BCQ_AEAD
from algorithms import AC_BCQ_ANN
from algorithms import RateBCQ
from algorithms import SpikingBCQ
import warnings

warnings.filterwarnings("ignore")


# Handles interactions with the environment, i.e. train behavioral or generate buffer
def interact_with_environment(env, state_dim, action_dim, max_action, device, args):
	# For saving files
	setting = f'{args.env}_{args.seed}'
	buffer_name = f'{args.buffer_name}_{setting}'

	# Initialize and load policy
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
	if args.mode == 'TD3':
		policy = TD3.TD3(**kwargs)
		print("TD3")
	elif args.mode == 'DDPG':
		policy = DDPG.DDPG(state_dim, action_dim, max_action, device)
		print("DDPG")
	if args.generate_buffer:
		if args.mode == 'TD3':
			policy.load(f"./models/TD3_{setting}")
			print(f'load:TD3_{setting}')
		else:
			policy.load(f"./models/{args.mode}_{setting}")
			print(f'load:{args.mode}_{setting}')

	# Initialize buffer
	replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
	
	evaluations = []

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	max_reward = -1e6
	# Interact with the environment for max_timesteps
	for t in range(int(args.max_timesteps)):

		episode_timesteps += 1

		# Select action with noise
		if (
			(args.generate_buffer and np.random.uniform(0, 1) < args.rand_action_p) or 
			(args.train_behavioral and t < args.start_timesteps)
		):
			action = env.action_space.sample()
		else: 
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.gaussian_std, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, _ = env.step(action) 
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if args.train_behavioral and t >= args.start_timesteps:
			policy.train(replay_buffer, args.batch_size)

		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			# Reset environment
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1

		# Evaluate episode
		if args.train_behavioral and (t + 1) % args.eval_freq == 0:
			eval_reward, _ = eval_policy(policy, args.env, args.seed)
			evaluations.append(eval_reward)
			np.save(f"./results/{args.mode}_{setting}", evaluations)
			if evaluations[-1] > max_reward:
				max_reward = evaluations[-1]
				policy.save(f"./models/{args.mode}_{setting}")

	if args.generate_buffer:
		eval_reward, _ = eval_policy(policy, args.env, args.seed)
		evaluations.append(eval_reward)
		np.save(f"./results/buffer_performance_{setting}", evaluations)
		replay_buffer.save(f"./buffers/{buffer_name}")


def train_SpikingBCQ(state_dim, action_dim, max_action, device, args):
	# For saving files
	setting = f"{args.env}_{args.seed}"
	buffer_name = f"{args.buffer_name}_{args.env}_9853"

	# Initialize policy
	if args.mode == "Spiking":  # Accum. Coding + SNN
		policy = SpikingBCQ.BCQ(state_dim, action_dim, max_action, device, args.discount, args.tau, args.lmbda,
								args.phi, args.T)
	if args.mode == "AEAD":  # ACSF
		policy = BCQ_AEAD.BCQ(state_dim, action_dim, max_action, device, args.discount, args.tau, args.lmbda,
								args.phi, args.T)
	if args.mode == "BCQ":  # Original BCQ
		policy = OriBCQ.BCQ(state_dim, action_dim, max_action, device, args.discount, args.tau, args.lmbda, args.phi)
	if args.mode == "AC-BCQ-ANN":  # Adapt. Coding + ANN
		policy = AC_BCQ_ANN.BCQ(state_dim, action_dim, max_action, device, args.discount, args.tau, args.lmbda, args.phi, args.T)
	if args.mode == "Rate":  # Rate Coding + SNN
		policy = RateBCQ.BCQ(state_dim, action_dim, max_action, device, args.discount, args.tau, args.lmbda,
								args.phi, args.T)
	# Load buffer
	replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
	if args.buffer == "DDPG_9853":
		replay_buffer.load(f"/data3/ql/buffers/DDPG_Buffer_9853/{buffer_name}")
	elif args.buffer == "DDPG_0":
		buffer_name = f"{args.buffer_name}_{args.env}_0"
		replay_buffer.load(f"/data3/ql/buffers/DDPG_Buffer_0/{buffer_name}")
	elif args.buffer == "TD3":
		buffer_name = f"{args.buffer_name}_TD3_{args.env}_9853"
		replay_buffer.load(f"/data3/ql/buffers/TD3_Buffer/{buffer_name}")
	mean, std = 0, 1
	print("load buffer:", buffer_name)
	evaluations = []
	episode_num = 0
	done = True
	training_iters = 0
	max_reward = -1e6
	while training_iters < args.max_timesteps:
		pol_vals = policy.train(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)
		eval_reward, reward_all = eval_policy(policy, args.env, args.seed, mean, std)
		evaluations.append(eval_reward)
		np.save(f"./results/BCQ_{args.mode}_{setting}_{args.buffer}", evaluations)
		training_iters += args.eval_freq
		print(f"Training iterations: {training_iters}")
		if eval_reward > max_reward:
			max_reward = eval_reward
			max_reward_all = reward_all
			policy.save(f"./models/BCQ_{args.mode}_{setting}_{args.buffer}")
	print(max_reward)
	print("---------------------------------------")
	print(f"Final Eval: {torch.mean(max_reward_all):.3f} ± {torch.std(max_reward_all)}")
	print("---------------------------------------")
	print(max_reward_all)

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, mean=0, std=1, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)
	reward_all = torch.zeros(eval_episodes)

	for index in range(eval_episodes):
		avg_reward = 0.
		state, done = eval_env.reset(), False
		while not done:
			# state = (np.array(state).reshape(1, -1) - mean) / std  # State Normalized
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward
		reward_all[index] = avg_reward

	mean_reward = torch.mean(reward_all)

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {mean_reward:.3f}")
	print("---------------------------------------")
	return mean_reward, reward_all


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", default="Ant-v3")               # OpenAI gym environment name
	parser.add_argument("--seed", default=9853, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--buffer_name", default="Robust")          # Prepends name to filename
	parser.add_argument("--eval_freq", default=5e3, type=float)     # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment or train for (this defines buffer size)
	parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used before training behavioral
	parser.add_argument("--rand_action_p", default=0.3, type=float) # Probability of selecting random action during batch generation
	parser.add_argument("--gaussian_std", default=0.3, type=float)  # Std of Gaussian exploration noise (Set to 0.1 if DDPG trains poorly)
	parser.add_argument("--batch_size", default=100, type=int)      # Mini batch size for networks
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--lmbda", default=0.75)                    # Weighting for clipped double Q-learning in BCQ
	parser.add_argument("--phi", default=0.05)                      # Max perturbation hyper-parameter for BCQ
	parser.add_argument("--train_behavioral", action="store_true")  # If true, train behavioral (DDPG)
	parser.add_argument("--generate_buffer", action="store_true")   # If true, generate buffer
	parser.add_argument("--gpu", default=0, type=int)
	parser.add_argument("--T", default=4, type=int)
	parser.add_argument("--mode", default="Spiking", type=str)
	parser.add_argument("--buffer", default="TD3", type=str)
	args = parser.parse_args()

	print("---------------------------------------")	
	if args.train_behavioral:
		print(f"Setting: Training behavioral, Env: {args.env}, Seed: {args.seed}")
	elif args.generate_buffer:
		print(f"Setting: Generating buffer, Env: {args.env}, Seed: {args.seed}")
	else:
		print(f"Setting: Training BCQ, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if args.train_behavioral and args.generate_buffer:
		print("Train_behavioral and generate_buffer cannot both be true.")
		exit()

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if not os.path.exists("./models"):
		os.makedirs("./models")

	if not os.path.exists("./buffers"):
		os.makedirs("./buffers")

	env = gym.make(args.env)

	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])
	device = torch.device("cuda:"+args.gpu.__str__() if torch.cuda.is_available() else "cpu")

	if args.train_behavioral or args.generate_buffer:
		interact_with_environment(env, state_dim, action_dim, max_action, device, args)
	else:
		train_SpikingBCQ(state_dim, action_dim, max_action, device, args)
