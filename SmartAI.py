from networks import JointPolVal
from DumbAI import act as dumact
import torch
from torch.autograd import Variable
import numpy as np
import pickle

class PPOAgent:
    # PPO
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = JointPolVal(state_dim, action_dim)

    def get_log_prob(self, action, action_mean, action_log_std, action_std):
        log_prob = -0.5 * torch.sum(((action - action_mean) / (action_std + 1e-8)).pow(2.), dim=1) \
            - 0.5 * self.action_dim * np.log(2. * np.pi) \
            - torch.sum(action_log_std, dim=1)
        return log_prob.unsqueeze(1)

    def get_entropy(self, action_log_std):
        return torch.sum(action_log_std + 0.5 * np.log(2. * np.pi * np.e), dim=1)

    def sample_action(self, state, exploration_decay):
        state = torch.Tensor(state)
        action_mean, action_log_std, action_std, value = self.net.forward(Variable(state))
        action = action_mean + action_std * exploration_decay * Variable(torch.normal(torch.Tensor(np.zeros(4)), torch.Tensor(np.ones(4))))
        action_log_prob = self.get_log_prob(action, action_mean, action_log_std, action_std)
        return action, action_log_prob, value

    def train(self, env, discount=0.99, epochs=10000, mb_size=100, lr=2.5e-4, gae_decay=0.96, clip_ratio=0.1, entropy_coef=0, value_coef=1, max_grad_norm=0.5):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        for i in range(epochs):
            # collect data
            mb_states, mb_actions, mb_action_log_probs, mb_values, mb_advantages = [], [], [], [], []
            mb_reward = 0
            for episode in range(mb_size):
                episode_states, episode_actions, episode_action_log_probs, episode_rewards, episode_values = [], [], [], [], []

                # do rollout
                env.reset()
                reward = 0
                while reward == 0:
                    state = np.expand_dims(env.game_vec, axis=0)
                    action, action_log_prob, value = self.sample_action(state, exploration_decay=1-i/epochs)
                    action, action_log_prob, value = action.data.numpy(), action_log_prob.data.numpy(), value.data.numpy()
                    oppact = dumact(state[0])
                    reward, _ = env.step(action[0], oppact)

                    episode_states.extend(state)
                    episode_actions.extend(action)
                    episode_action_log_probs.extend(action_log_prob)
                    episode_rewards.extend(np.array([reward]))
                    episode_values.extend(value)

                # GAE
                episode_advantages = [0 for _ in range(len(episode_states))]
                for t in reversed(range(len(episode_states))):
                    if t == len(episode_states) - 1:
                        # adv[t] = reward[t] - value[t]
                        episode_advantages[t] = episode_rewards[t] - episode_values[t]
                    else:
                        # adv[t] = reward[t] + gamma * value[t+1] - value[t] + gamma * lambda * adv[t+1]
                        episode_advantages[t] = episode_rewards[t] + discount * episode_values[t+1] - episode_values[t] \
                            + discount * gae_decay * episode_advantages[t+1]

                # record episode
                mb_reward += reward
                mb_states.extend(episode_states)
                mb_actions.extend(episode_actions)
                mb_action_log_probs.extend(episode_action_log_probs)
                mb_values.extend(episode_values)
                mb_advantages.extend(episode_advantages)

            if i % 100 == 0:
                print('mb {} average reward: {}'.format(i, mb_reward / mb_size))
                env.reset()
                state = Variable(torch.Tensor(np.expand_dims(env.game_vec, axis=0)))
                print('starting state value: {}'.format(self.net.forward(state)[3].data.numpy()))

            # preprocessing
            mb_states = torch.Tensor(np.array(mb_states))
            mb_actions = torch.Tensor(np.array(mb_actions))
            mb_action_log_probs = torch.Tensor(np.array(mb_action_log_probs))
            mb_values = torch.Tensor(np.array(mb_values))
            mb_advantages = torch.Tensor(np.array(mb_advantages))

            mb_value_targets = mb_advantages + mb_values
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # setup losses, get gradients, update
            action_means, action_log_stds, action_stds, values = self.net.forward(Variable(mb_states))
            action_log_probs = self.get_log_prob(Variable(mb_actions), action_means, action_log_stds, action_stds)
            action_prob_ratio = torch.exp(action_log_probs - Variable(mb_action_log_probs))

            policy_loss1 = action_prob_ratio * Variable(mb_advantages)
            policy_loss2 = torch.clamp(action_prob_ratio, 1-clip_ratio, 1+clip_ratio) * Variable(mb_advantages)
            policy_loss = -torch.min(policy_loss1, policy_loss2).mean()

            #values_clipped = Variable(mb_values) + torch.clamp(values - Variable(mb_values), -clip_ratio, clip_ratio)
            #value_loss1 = 0.5 * (values - Variable(mb_value_targets)).pow(2.)
            #value_loss2 = 0.5 * (values_clipped - Variable(mb_value_targets)).pow(2.)
            #value_loss = torch.max(value_loss1, value_loss2).mean()
            value_loss = 0.5 * (values - Variable(mb_value_targets)).pow(2.).mean()

            entropy_loss = self.get_entropy(action_log_stds).mean()

            total_loss = policy_loss + value_coef * value_loss - (1 - i/epochs) * entropy_coef * entropy_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm(self.net.parameters(), max_grad_norm, norm_type='inf')
            optimizer.step()

            if i % 100 == 0:
                print('avg policy gradient l2 norm:', self.net.action_mean.weight.grad.pow(2.).mean().pow(0.5).data.numpy())
                print('entropy:', entropy_loss.data.numpy())
                print('value loss:', value_loss.data.numpy())
                torch.save(self.net, 'log/{}.pt'.format(i))
