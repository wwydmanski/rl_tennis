from .model_tf import TFActor, TFCritic
import numpy as np
from collections import namedtuple, deque
import random
import copy
import tensorflow as tf

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 512         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
UPDATE_EVERY = 4


class Agent:
    def __init__(self, state_size, action_size, random_seed=42):
        self.sess = tf.Session()

        self.action_size = action_size
        self.noise = OUNoise(action_size, random_seed)

        self.actor_local = TFActor(self.sess, state_size, action_size, 'local')
        self.actor_target = TFActor(self.sess, state_size, action_size, 'target')

        self.critic_local = TFCritic(self.sess, state_size, action_size, 'local', self.actor_local.output)
        self.critic_target = TFCritic(self.sess, state_size, action_size, 'target', None)
        self.agent_loss, self.agent_learn = self.get_agent_update_op()

        self.sess.run([tf.global_variables_initializer(),
                           tf.local_variables_initializer()])

        # # HARD update
        # self.soft_update(self.actor_target, self.actor_target, 1.0)
        # self.soft_update(self.critic_local, self.critic_target, 1.0)

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        self.t_step = 0

        self.episodes_passed = 1
        self.actor_update, self.critic_update = self._get_soft_update_ops()


    def act(self, state, add_noise):
        """Get action according to actor policy

        Arguments:
            state (List[float]): Current observation of environment
            add_noise (bool): Whether to add noise from Ornstein-Uhlenbeck process

        Returns:
            ndarray[np.float32] -- Estimated best action
        """
        
        action_values = self.actor_local.forward(state)

        if add_noise:
            action_values += (self.noise.sample()-0.6)/np.sqrt(self.episodes_passed)

        return np.clip(action_values, -1, 1)

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step += 1

        if self.t_step % UPDATE_EVERY == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples

        Arguments:
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done)
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        max_Qhat = self.critic_target.forward(
            next_states, self.actor_target.forward(next_states))
        Q_target = rewards + (gamma * max_Qhat * (1 - dones))

        self.critic_loss = self.critic_local.train(states, actions, Q_target)

        # Q_expected = self.critic_local(states, actions)
        # loss = torch.nn.functional.mse_loss(Q_expected, Q_target)
        # self.critic_loss = loss.cpu().data.numpy()

        # self.critic_optimizer.zero_grad()
        # torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        # loss.backward()
        # self.critic_optimizer.step()

        # critique = self.critic_local.forward(states, self.actor_local.forward(states))
        # self.actor_local.train(critique)
        # actor_input = self.sess.run(self.actor_local.output, feed_dict={self.actor_local.input: states})
        loss, _ = self.sess.run([self.agent_loss, self.agent_learn], feed_dict={self.critic_local.input: states, self.actor_local.input: states})
        self.actor_loss = loss
        # policy_loss = policy_loss.mean()
        # self.actor_loss = policy_loss.cpu().data.numpy()

        # self.actor_optimizer.zero_grad()
        # policy_loss.backward()
        # self.actor_optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update()

    def soft_update(self):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Arguments:
            local_model (PyTorch model): model weights will be copied from
            target_model (PyTorch model): model weights will be copied to
            tau (float): interpolation parameter
        """
        self.sess.run(self.actor_update)
        self.sess.run(self.critic_update)
        # iter_params = zip(target_model.parameters(), local_model.parameters())
        # for target_param, local_param in iter_params:
        #     tensor_aux = tau*local_param.data + (1.0-tau)*target_param.data
        #     target_param.data.copy_(tensor_aux)

    def _get_soft_update_ops(self):
        Qvars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_inference_local')
        target_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_inference_target')

        Qvars2 = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_inference_local')
        target_vars2 = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_inference_target')

        actors_op = [tvar.assign(TAU*qvar + (1.0-TAU)*tvar) for qvar, tvar in zip(Qvars, target_vars)]
        critic_op = [tvar.assign(TAU*qvar + (1.0-TAU)*tvar) for qvar, tvar in zip(Qvars2, target_vars2)]

        return actors_op, critic_op

    def reset(self):
        self.noise.reset()
        self.episodes_passed += 1

    def get_agent_update_op(self):
        critique = -self.critic_local.training_output
        critique = tf.reduce_mean(critique)
        return critique, self.actor_local.optimizer.minimize(critique)


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
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
