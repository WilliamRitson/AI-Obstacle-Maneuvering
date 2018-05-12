import numpy as np
import gym
import _pickle as pickle
import random


# Input/output shapes of Bipedal Walker weights (24 inputs, 4 outputs)
# with 1 hidden layer of 16 nodes
# (Linear network)
class Model(object):

    def __init__(self):
        self.weights = [
            np.zeros(shape=(24, 16)),
            np.zeros(shape=(16, 16)),
            np.zeros(shape=(16, 4)),
        ]

    def predict(self, inp):
        out = np.expand_dims(inp.flatten(), 0)
        # Normalize vector
        out = out / np.linalg.norm(out)
        for layer in self.weights:
            out = np.dot(out, layer)
        return out[0]

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights


class Agent:

    ### Iterate the VERSION by 1 everytime you modify the algorithm
    VERSION = 1


    AGENT_HISTORY_LENGTH = 1 # Number of directly previous observations to use in our model predictions
    POPULATION_SIZE = 20 # Number of episodes to run simultaneously. We then use the best rewarded episodes to update our weights
    EPISODE_AGENTS = 1 # Number of agents to run per episode, we use the average reward
    SIGMA = 0.1 # Amount of mutation to perform on weights
    LEARNING_RATE = 0.01
    LEARNING_RATE_DECAY = 1.0 # Gradually lower learning rate if < 1.0
    # self.exploration: as the simulation progresses, we progressively reduce the amount of random actions, using our model to predict actions instead
    INITIAL_EXPLORATION = 1.0 # Max exploration
    FINAL_EXPLORATION = 0.0 # Min exploration
    EXPLORATION_STEPS = 1000000 # Number of iterations before reaching FINAL_EXPLORATION

    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.model = Model()
        self.es = EvolutionStrategy(self.model.get_weights(), self._get_reward, self.POPULATION_SIZE, self.SIGMA, self.LEARNING_RATE, self.LEARNING_RATE_DECAY)
        self.exploration = self.INITIAL_EXPLORATION

    def load(self, filename):
        with open(filename,'rb') as fp:
            weights = None
            try:
                weights = pickle.load(fp)
            except: # Compatability with python2 i.e. plain-text pickles
                fp.seek(0)
                weights = pickle.load(fp, encoding='latin1')

            self.model.set_weights(weights)
        self.es.weights = self.model.get_weights()


    def save(self, filename):
        with open(filename, 'wb') as fp:
            pickle.dump(self.es.get_weights(), fp)


    def play(self, episodes, render=True):
        self.model.set_weights(self.es.weights)
        for episode in range(episodes):
            total_reward = 0
            observation = self.env.reset()
            sequence = [observation]*self.AGENT_HISTORY_LENGTH
            done = False
            while not done:
                if render:
                    self.env.render()
                action = self._get_predicted_action(sequence)
                observation, reward, done, _ = self.env.step(action)
                total_reward += reward
                sequence = sequence[1:]
                sequence.append(observation)
            print('total reward:', total_reward)


    def train(self, iterations):
        self.es.run(iterations, print_step=1)


    def _get_predicted_action(self, sequence):
        prediction = self.model.predict(np.array(sequence))
        return prediction


    # Here we actually runs the simulation
    def _get_reward(self, weights):
        total_reward = 0.0
        self.model.set_weights(weights)
        EXPLORATION_DECAY = self.INITIAL_EXPLORATION / self.EXPLORATION_STEPS

        for episode in range(self.EPISODE_AGENTS):
            observation = self.env.reset()
            sequence = [observation]*self.AGENT_HISTORY_LENGTH
            done = False
            while not done:
                self.exploration = max(self.FINAL_EXPLORATION, self.exploration - EXPLORATION_DECAY)
                if random.random() < self.exploration:
                    action = self.env.action_space.sample()
                else:
                    action = self._get_predicted_action(sequence)
                observation, reward, done, _ = self.env.step(action)
                total_reward += reward
                sequence = sequence[1:]
                sequence.append(observation)

        return total_reward / self.EPISODE_AGENTS


# General evolution strategy algorithm
# (this is independent from any gym environment)
# Credits go to https://github.com/alirezamika/evostra
class EvolutionStrategy(object):

    def __init__(self, weights, get_reward_func, population_size=50, sigma=0.1, learning_rate=0.001, decay=1.0, seed=0):
        np.random.seed(seed)
        self.weights = weights
        self.get_reward = get_reward_func
        self.POPULATION_SIZE = population_size
        self.SIGMA = sigma
        self.DECAY = decay
        self.learning_rate = learning_rate

    def _get_mutated_weights(self, current_weights, sample):
        weights = []
        for index, i in enumerate(sample):
            jittered = self.SIGMA * i
            weights.append(current_weights[index] + jittered)
        return weights


    def get_weights(self):
        return self.weights


    def run(self, iterations, print_step=10):
        for iteration in range(iterations):

            population = []
            rewards = np.zeros(self.POPULATION_SIZE)
            for i in range(self.POPULATION_SIZE):
                x = []
                for w in self.weights:                 
                    x.append(np.random.randn(*w.shape))
                population.append(x)

            for i in range(self.POPULATION_SIZE):
                mutated_weights = self._get_mutated_weights(self.weights, population[i])
                rewards[i] = self.get_reward(mutated_weights)

            rewards = (rewards - np.mean(rewards)) / np.std(rewards)

            for index, w in enumerate(self.weights):
                A = np.array([p[index] for p in population])
                self.weights[index] = w + self.learning_rate / (self.POPULATION_SIZE * self.SIGMA) * np.dot(A.T, rewards).T

            self.learning_rate *= self.DECAY

            if (iteration+1) % print_step == 0:
                print('iter %d. reward: %f' % (iteration+1, self.get_reward(self.weights)))