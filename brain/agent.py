from .model import Actor, Critic
import numpy as np
import torch
from collections import namedtuple, deque
import random
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Config:
    def __init__(self):
        self.BUFFER_SIZE = int(1e5)  # replay buffer size
        self.BATCH_SIZE = 1024         # minibatch size
        self.GAMMA = 0.99            # discount factor
        self.TAU = 1e-3              # for soft update of target parameters
        self.LR_ACTOR = 5e-4         # learning rate of the actor
        self.LR_CRITIC = 6e-3        # learning rate of the critic
        self.UPDATE_EVERY = 4


class Agent:
    def __init__(self, state_size, action_size, config=Config(), random_seed=42):
        self.action_size = action_size
        self.noise = OUNoise(action_size, random_seed)
        self.config = config

        self.actor_local = Actor(
            state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(
            state_size, action_size, random_seed).to(device)

        self.critic_local = Critic(
            state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(
            state_size, action_size, random_seed).to(device)

        # HARD update
        self.soft_update(self.actor_target, self.actor_target, 1.0)
        self.soft_update(self.critic_local, self.critic_target, 1.0)

        self.memory = ReplayBuffer(
            action_size, config.BUFFER_SIZE, config.BATCH_SIZE, random_seed)

        self.actor_optimizer = torch.optim.Adam(
            self.actor_local.parameters(), lr=config.LR_ACTOR)
        self.critic_optimizer = torch.optim.Adam(
            self.critic_local.parameters(), lr=config.LR_CRITIC)
        self.t_step = 0

        self.episodes_passed = 1

        self.learning_started = False

    def act(self, state, add_noise):
        """Get action according to actor policy

        Arguments:
            state (List[float]): Current observation of environment
            add_noise (bool): Whether to add noise from Ornstein-Uhlenbeck process

        Returns:
            ndarray[np.float32] -- Estimated best action
        """

        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action_values = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            action_values += (self.noise.sample()-0.6) / \
                np.sqrt(self.episodes_passed)

        return np.clip(action_values, -1, 1)

    def step(self, states, actions, rewards, next_states, dones):
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)

        self.t_step += 1

        if self.t_step % self.config.UPDATE_EVERY == 0:
            if len(self.memory) > self.config.BATCH_SIZE:
                if not self.learning_started:
                    self.learning_started = True
                    print(f"Learning started")
                experiences = self.memory.sample()
                self.learn(experiences, self.config.GAMMA)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples

        Arguments:
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done)
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        max_Qhat = self.critic_target(
            next_states, self.actor_target(next_states))
        Q_target = rewards + (gamma * max_Qhat * (1 - dones))

        Q_expected = self.critic_local(states, actions)
        loss = torch.nn.functional.mse_loss(Q_expected, Q_target)
        self.critic_loss = loss.cpu().data.numpy()

        self.critic_optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        loss.backward()
        self.critic_optimizer.step()

        policy_loss = -self.critic_local(states, self.actor_local(states))
        policy_loss = policy_loss.mean()
        self.actor_loss = policy_loss.cpu().data.numpy()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.actor_local, self.actor_target, self.config.TAU)
        self.soft_update(self.critic_local, self.critic_target, self.config.TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Arguments:
            local_model (PyTorch model): model weights will be copied from
            target_model (PyTorch model): model weights will be copied to
            tau (float): interpolation parameter
        """
        iter_params = zip(target_model.parameters(), local_model.parameters())
        for target_param, local_param in iter_params:
            tensor_aux = tau*local_param.data + (1.0-tau)*target_param.data
            target_param.data.copy_(tensor_aux)

    def reset(self):
        self.noise.reset()
        self.episodes_passed += 1


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * \
            np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.buffer_size = buffer_size
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.filled = False

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

        if not self.filled and len(self.memory) == self.buffer_size:
            self.filled = True
            print("Memory filled")

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack(
            [e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
