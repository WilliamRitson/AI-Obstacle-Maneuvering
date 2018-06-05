import _pickle as pickle
import random
from copy import deepcopy, copy
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
# import multiprocessing as mp

import numpy as np
import gym

gym.logger.setLevel(gym.logger.ERROR)

AGENT_HISTORY_LENGTH = 1 # Number of directly previous observations to use in our model predictions
POPULATION_SIZE = 24 # Number of episodes to run simultaneously. We then use the best rewarded episodes to update our weights
EPISODE_AGENTS = 1 # Number of agents to run per episode, we use the average reward
SIGMA = 0.1 # Amount of mutation to perform on weights
LEARNING_RATE = 0.01
LEARNING_RATE_DECAY = 1.0 # Gradually lower learning rate if < 1.0
# self.exploration: as the simulation progresses, we progressively reduce the amount of random actions, using our model to predict actions instead
INITIAL_EXPLORATION = 1.0 # Max exploration
FINAL_EXPLORATION = 0.0 # Min exploration
EXPLORATION_STEPS = 10000000 # Number of iterations before reaching FINAL_EXPLORATION


# This function is seperate from a class for optimal threading
# Here we actually run the simulation
def _get_reward(args):
    model, envname, const = args

    env = gym.make(envname)

    total_reward = 0.0
    EXPLORATION_DECAY = (const['INITIAL_EXPLORATION'] - const['FINAL_EXPLORATION']) / const['EXPLORATION_STEPS']
    exploration = const['exploration']

    observation = env.reset()
    sequence = [observation] * const['AGENT_HISTORY_LENGTH']
    done = False
    while not done:
        exploration = max(const['FINAL_EXPLORATION'], exploration - EXPLORATION_DECAY)
        if random.random() < exploration:
            action = env.action_space.sample()
        else:
            action = model.predict(np.array(sequence))
        observation, reward, done, _ = env.step(action)
        total_reward += reward
        sequence = sequence[1:]
        sequence.append(observation)

    return total_reward, const['exploration'] - exploration


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
    VERSION = 5

    def __init__(self, env_name):
        self.model = Model()
        self.es = EvolutionStrategy(self.model.get_weights(), _get_reward, self._setup_envs)
        self.env_name = env_name

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
        env = gym.make(self.env_name)
        self.model.set_weights(self.es.weights)
        for episode in range(episodes):
            total_reward = 0
            observation = env.reset()
            sequence = [observation]*AGENT_HISTORY_LENGTH
            done = False
            while not done:
                if render:
                    env.render()
                action = self.model.predict(np.array(sequence))
                observation, reward, done, _ = env.step(action)
                total_reward += reward
                sequence = sequence[1:]
                sequence.append(observation)
            print('total reward:', total_reward)


    def train(self, iterations):
        self.es.run(iterations, print_step=1)

    def _setup_envs(self):
        pass_vars = dict(EXPLORATION_STEPS=EXPLORATION_STEPS, INITIAL_EXPLORATION=INITIAL_EXPLORATION, FINAL_EXPLORATION=FINAL_EXPLORATION, EPISODE_AGENTS=EPISODE_AGENTS, AGENT_HISTORY_LENGTH=AGENT_HISTORY_LENGTH)
        return [ (
            Model(), # model must be first element in tuple
            self.env_name,
            pass_vars.copy()
        ) for i in range(POPULATION_SIZE) ]


# General evolution strategy algorithm
# (this is independent from any gym environment)
# Credits go to https://github.com/alirezamika/evostra
class EvolutionStrategy(object):

    def __init__(self, weights, get_reward_func, setup_envs_func, population_size=50, sigma=0.1, learning_rate=0.001, decay=1.0, exploration=1.0, seed=0):
        np.random.seed(seed)
        self.weights = weights
        self.get_reward = get_reward_func
        self.setup_envs = setup_envs_func
        self.pool = ThreadPool(cpu_count())
        self.exploration = INITIAL_EXPLORATION
        self.learning_rate = LEARNING_RATE

    def _get_mutated_weights(self, current_weights, sample):
        weights = []
        for index, i in enumerate(sample):
            jittered = SIGMA * i
            weights.append(current_weights[index] + jittered)
        return weights


    def get_weights(self):
        return self.weights


    def run(self, iterations, print_step=10):

        vals = self.setup_envs()

        for iteration in range(iterations):

            population = []
            for i in range(POPULATION_SIZE):
                x = [ np.random.randn(*w.shape) for w in self.weights ]
                population.append(x)

                # Assign mutated weights to each environment
                vals[i][0].set_weights(self._get_mutated_weights(deepcopy(self.weights), x))
                # Update exploration
                vals[i][2]['exploration'] = self.exploration

            # Multithreaded strategy
            out = self.pool.imap(self.get_reward, vals) # chunk size of 2 seems to work ok

            rewards = []
            for reward, delta_exploration in out:
                rewards.append(reward)
                self.exploration -= delta_exploration

            rewards = (rewards - np.mean(rewards)) / np.std(rewards)

            for index, w in enumerate(self.weights):
                A = np.array([p[index] for p in population])
                # Calculate new weights
                self.weights[index] = w + self.learning_rate / (POPULATION_SIZE * SIGMA) * np.dot(A.T, rewards).T

            self.learning_rate *= LEARNING_RATE_DECAY

            if (iteration+1) % print_step == 0:
                model = vals[0][0]
                model.set_weights(self.weights)
                print('iter %d. reward: %f' % (iteration+1, self.get_reward(vals[0])[0]))