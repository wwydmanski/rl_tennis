import numpy as np
import tqdm
import matplotlib.pyplot as plt


class Teacher:
    """Class that handles training of RL model in Unity environment

    Attributes:
        agent (Pytorch model): agent which will be trained
        env (Unity environment): environment used for learning
        num_agents (int): number of agents present at once
        brain_name (str): name of used Unity brain
    """

    def __init__(self, agent, env, brain_name, num_agents):
        self.agent = agent
        self.env = env
        self.num_agents = num_agents
        self.brain_name = brain_name

    def train(self, epochs, mean_window, solution_threshold):
        scores_ep = []

        with tqdm.trange(epochs) as t:
            for i_episode in t:
                env_info = self.env.reset(train_mode=True)[self.brain_name]
                states = env_info.vector_observations
                scores = np.zeros(self.num_agents)

                while True:
                    actions = self.agent.act(states, True)
                    env_info = self.env.step(actions)[self.brain_name]
                    next_states, rewards, dones = self._get_info_from_env(
                        env_info)

                    self.agent.step(states, actions, rewards,
                                    next_states, dones)

                    scores += rewards
                    states = next_states
                    if any(dones):
                        break

                # After episode updates
                scores_ep.append(np.max(scores))
                self.agent.reset()

                # Updating postfix
                if i_episode > mean_window:
                    mn = np.mean(scores_ep[-mean_window:])
                else:
                    mn = np.mean(scores_ep[-i_episode:])

                if self.agent.learning_started:
                    agent_loss = self.agent.actor_loss 
                    critic_loss = self.agent.critic_loss
                else:
                    agent_loss = 0
                    critic_loss = 0

                t.set_postfix(scores=mn, actor_loss=agent_loss,
                              critic_loss=critic_loss)

                if mn >= solution_threshold:
                    print(
                        f"Environment solved in {i_episode-mean_window} episodes!")
                    return scores_ep
        return scores_ep

    def display(self):
        env_info = self.env.reset(train_mode=False)[self.brain_name]
        states = env_info.vector_observations
        scores = np.zeros(self.num_agents)

        while True:
            actions = self.agent.act(states, False)

            env_info = self.env.step(actions)[self.brain_name]
            next_states, rewards, dones = self._get_info_from_env(
                env_info)

            scores += rewards

            self.agent.step(
                states, actions, rewards, next_states, dones)

            states = next_states
            if np.any(dones):
                break
        return np.mean(scores)

    def _get_info_from_env(self, env_info):
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        return next_states, rewards, dones

    @classmethod
    def visualise_scores(cls, scores, mean_window, solution_threshold):
        means = np.convolve(scores, np.ones(
            (mean_window,))/mean_window, mode='valid')
        plt.style.use('seaborn')
        plt.plot(scores, label='scores')
        plt.plot(np.arange(len(scores)-len(means), len(scores)), means,
                 color='xkcd:light orange', label=f'Mean for last {mean_window} episodes')
        plt.plot(np.arange(len(scores)), solution_threshold*np.ones(len(scores)),
                 '--', color='xkcd:melon', label=f'{solution_threshold} score boundary')
        plt.xlabel("Iterations")
        plt.ylabel("Score")
        plt.legend()
        plt.show()
