import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import layer, neuron, surrogate, functional


class NonSpikingLIFNode(neuron.MultiStepLIFNode):
    def forward(self, dv: torch.Tensor):
        self.neuronal_charge(dv)
        # self.neuronal_fire()
        # self.neuronal_reset()
        return self.v


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, phi=0.05, T=32, tau=2.0, alpha=2.0, v_threshold=1.0,
                 v_reset=0.0):
        super(Actor, self).__init__()
        self.T = T
        surrogate_func = surrogate.ATan(alpha)
        self.l1 = nn.Sequential(layer.SeqToANNContainer(nn.Linear(state_dim + action_dim, 400)),
                                neuron.MultiStepLIFNode(tau, v_threshold=v_threshold, v_reset=v_reset,
                                                        surrogate_function=surrogate_func))
        self.l2 = nn.Sequential(layer.SeqToANNContainer(nn.Linear(400, 300)),
                                neuron.MultiStepLIFNode(tau, v_threshold=v_threshold, v_reset=v_reset,
                                                        surrogate_function=surrogate_func))
        self.l3 = nn.Sequential(layer.SeqToANNContainer(nn.Linear(300, action_dim)),
                                NonSpikingLIFNode(tau, v_threshold=v_threshold, v_reset=v_reset,
                                                        surrogate_function=surrogate_func))

        self.max_action = max_action
        self.phi = phi

    def forward(self, state, action):
        input = torch.cat([state, action], 1)
        input = input.repeat(self.T, 1, 1)
        a = self.l1(input)
        a = self.l2(a)
        a = self.l3(a)
        a, _ = a.max(dim=0)
        a = self.phi * self.max_action * torch.tanh(a)
        return (a + action).clamp(-self.max_action, self.max_action)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, T=32, tau=2.0, alpha=2.0, v_threshold=1.0, v_reset=0.0):
        super(Critic, self).__init__()
        self.T = T
        surrogate_func = surrogate.ATan(alpha)
        self.l1 = nn.Sequential(layer.SeqToANNContainer(nn.Linear(state_dim + action_dim, 400)),
                                neuron.MultiStepLIFNode(tau, v_threshold=v_threshold, v_reset=v_reset,
                                                        surrogate_function=surrogate_func))
        self.l2 = nn.Sequential(layer.SeqToANNContainer(nn.Linear(400, 300)),
                                neuron.MultiStepLIFNode(tau, v_threshold=v_threshold, v_reset=v_reset,
                                                        surrogate_function=surrogate_func))
        self.l3 = nn.Sequential(layer.SeqToANNContainer(nn.Linear(300, 1)),
                                NonSpikingLIFNode(tau, v_threshold=v_threshold, v_reset=v_reset,
                                                        surrogate_function=surrogate_func))

        self.l4 = nn.Sequential(layer.SeqToANNContainer(nn.Linear(state_dim + action_dim, 400)),
                                neuron.MultiStepLIFNode(tau, v_threshold=v_threshold, v_reset=v_reset,
                                                        surrogate_function=surrogate_func))
        self.l5 = nn.Sequential(layer.SeqToANNContainer(nn.Linear(400, 300)),
                                neuron.MultiStepLIFNode(tau, v_threshold=v_threshold, v_reset=v_reset,
                                                        surrogate_function=surrogate_func))
        self.l6 = nn.Sequential(layer.SeqToANNContainer(nn.Linear(300, 1)),
                                NonSpikingLIFNode(tau, v_threshold=v_threshold, v_reset=v_reset,
                                                        surrogate_function=surrogate_func))

    def forward(self, state, action):
        input = torch.cat([state, action], 1)
        input = input.repeat(self.T, 1, 1)
        q1 = self.l1(input)
        q1 = self.l2(q1)
        q1 = self.l3(q1)
        q1, _ = q1.max(dim=0)

        q2 = self.l4(input)
        q2 = self.l5(q2)
        q2 = self.l6(q2)
        q2, _ = q2.max(dim=0)
        return q1, q2

    def q1(self, state, action):
        input = torch.cat([state, action], 1)
        input = input.repeat(self.T, 1, 1)
        q1 = self.l1(input)
        q1 = self.l2(q1)
        q1 = self.l3(q1)
        q1, _ = q1.max(dim=0)
        return q1


# Vanilla Variational Auto-Encoder
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, device, T=32, tau=2.0, alpha=2.0, v_threshold=1.0,
                 v_reset=0.0):
        super(VAE, self).__init__()
        self.T = T
        surrogate_func = surrogate.ATan(alpha)
        self.e1 = nn.Sequential(layer.SeqToANNContainer(nn.Linear(state_dim + action_dim, 750)),
                                neuron.MultiStepLIFNode(tau, v_threshold=v_threshold, v_reset=v_reset,
                                                        surrogate_function=surrogate_func))
        self.e2 = nn.Sequential(layer.SeqToANNContainer(nn.Linear(750, 750)),
                                neuron.MultiStepLIFNode(tau, v_threshold=v_threshold, v_reset=v_reset,
                                                        surrogate_function=surrogate_func))
        self.mean = nn.Sequential(layer.SeqToANNContainer(nn.Linear(750, latent_dim)),
                                NonSpikingLIFNode(tau, v_threshold=v_threshold, v_reset=v_reset,
                                                        surrogate_function=surrogate_func))
        self.log_std = nn.Sequential(layer.SeqToANNContainer(nn.Linear(750, latent_dim)),
                                NonSpikingLIFNode(tau, v_threshold=v_threshold, v_reset=v_reset,
                                                        surrogate_function=surrogate_func))

        self.d1 = nn.Sequential(layer.SeqToANNContainer(nn.Linear(state_dim + latent_dim, 750)),
                                neuron.MultiStepLIFNode(tau, v_threshold=v_threshold, v_reset=v_reset,
                                                        surrogate_function=surrogate_func))
        self.d2 = nn.Sequential(layer.SeqToANNContainer(nn.Linear(750, 750)),
                                neuron.MultiStepLIFNode(tau, v_threshold=v_threshold, v_reset=v_reset,
                                                        surrogate_function=surrogate_func))
        self.d3 = nn.Sequential(layer.SeqToANNContainer(nn.Linear(750, action_dim)),
                                NonSpikingLIFNode(tau, v_threshold=v_threshold, v_reset=v_reset,
                                                        surrogate_function=surrogate_func))

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, state, action):
        input = torch.cat([state, action], 1)
        input = input.repeat(self.T, 1, 1)
        z = self.e1(input)
        z = self.e2(z)
        mean, _ = self.mean(z).max(dim=0)

        # Clamped for numerical stability
        log_std, _ = self.log_std(z).max(dim=0)
        log_std = log_std.clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(state, z)

        return u, mean, std

    def decode(self, state, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5, 0.5)
        input = torch.cat([state, z], 1)
        input = input.repeat(self.T, 1, 1)
        # print(input.shape)
        a = self.d1(input)
        a = self.d2(a)
        a = self.d3(a)
        a, _ = a.max(dim=0)
        return self.max_action * torch.tanh(a)


class BCQ(object):
    def __init__(self, state_dim, action_dim, max_action, device, discount=0.99, tau=0.005, lmbda=0.75,
                 phi=0.05, T=32):
        latent_dim = action_dim * 2

        self.actor = Actor(state_dim, action_dim, max_action, phi, T).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(state_dim, action_dim, T).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.vae = VAE(state_dim, action_dim, latent_dim, max_action, device).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters())

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.lmbda = lmbda
        self.device = device
        self.total_it = 0

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).repeat(100, 1).to(self.device)
            action = self.actor(state, self.vae.decode(state))
            q1 = self.critic.q1(state, action)
            ind = q1.argmax(0)
            functional.reset_net(self.vae)
            functional.reset_net(self.actor)
            functional.reset_net(self.critic)

        return action[ind].cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100):
        self.total_it += 1
        for it in range(iterations):
            # Sample replay buffer / batch
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            # Variational Auto-Encoder Training
            recon, mean, std = self.vae(state, action)

            recon_loss = F.mse_loss(recon, action)
            KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + 0.5 * KL_loss

            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_optimizer.step()
            functional.reset_net(self.vae)

            # Critic Training
            with torch.no_grad():
                # Duplicate next state 10 times
                next_state = torch.repeat_interleave(next_state, 10, 0)

                # Compute value of perturbed actions sampled from the VAE
                target_Q1, target_Q2 = self.critic_target(next_state,
                                                          self.actor_target(next_state, self.vae.decode(next_state)))
                functional.reset_net(self.critic_target)
                functional.reset_net(self.vae)
                # Soft Clipped Double Q-learning
                target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(target_Q1,
                                                                                                        target_Q2)
                # Take max over each action sampled from the VAE
                target_Q = target_Q.reshape(batch_size, -1).max(1)[0].reshape(-1, 1)

                target_Q = reward + not_done * self.discount * target_Q

            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            functional.reset_net(self.critic)
            if self.total_it % 2 == 0:
                # Pertubation Model / Action Training
                sampled_actions = self.vae.decode(state)
                perturbed_actions = self.actor(state, sampled_actions)
                functional.reset_net(self.vae)

                # Update through DPG
                actor_loss = -self.critic.q1(state, perturbed_actions).mean()
                functional.reset_net(self.critic)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                functional.reset_net(self.actor)

                # Update Target Networks
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.vae.state_dict(), filename + "_vae")
        torch.save(self.vae_optimizer.state_dict(), filename + "_vae_optimizer")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))

        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))

        self.vae.load_state_dict(torch.load(filename + "_vae"))
        self.vae_optimizer.load_state_dict(torch.load(filename + "_vae_optimizer"))