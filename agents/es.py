import numpy as np
import gym
import _pickle as pickle

class Model(object):

    def __init__(self):
        self.weights = [np.zeros(shape=(24, 16)), np.zeros(shape=(16, 16)), np.zeros(shape=(16, 4))]

    def predict(self, inp):
        out = np.expand_dims(inp.flatten(), 0)
        out = out / np.linalg.norm(out)
        for layer in self.weights:
            out = np.dot(out, layer)
        return out[0]

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights


class Agent:

    AGENT_HISTORY_LENGTH = 1
    POPULATION_SIZE = 20
    EPS_AVG = 1
    SIGMA = 0.1
    LEARNING_RATE = 0.01
    INITIAL_EXPLORATION = 1.0
    FINAL_EXPLORATION = 0.0
    EXPLORATION_DEC_STEPS = 1000000

    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.model = Model()
        self.es = EvolutionStrategy(self.model.get_weights(), self.get_reward, self.POPULATION_SIZE, self.SIGMA, self.LEARNING_RATE)
        self.exploration = self.INITIAL_EXPLORATION


    def get_predicted_action(self, sequence):
        prediction = self.model.predict(np.array(sequence))
        return prediction


    def load(self, filename='weights.pkl'):
        with open(filename,'rb') as fp:
            weights = None
            try:
                weights = pickle.load(fp)
            except:
                fp.seek(0)
                weights = pickle.load(fp, encoding='latin1')

            self.model.set_weights(weights)
        self.es.weights = self.model.get_weights()


    def save(self, filename='weights.pkl'):
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
                action = self.get_predicted_action(sequence)
                observation, reward, done, _ = self.env.step(action)
                total_reward += reward
                sequence = sequence[1:]
                sequence.append(observation)
            print('total reward:', total_reward)


    def train(self, iterations):
        self.es.run(iterations, print_step=1)


    def get_reward(self, weights):
        total_reward = 0.0
        self.model.set_weights(weights)

        for episode in range(self.EPS_AVG):
            observation = self.env.reset()
            sequence = [observation]*self.AGENT_HISTORY_LENGTH
            done = False
            while not done:
                self.exploration = max(self.FINAL_EXPLORATION, self.exploration - self.INITIAL_EXPLORATION/self.EXPLORATION_DEC_STEPS)
                if random.random() < self.exploration:
                    action = self.env.action_space.sample()
                else:
                    action = self.get_predicted_action(sequence)
                observation, reward, done, _ = self.env.step(action)
                total_reward += reward
                sequence = sequence[1:]
                sequence.append(observation)

        return total_reward/self.EPS_AVG


class EvolutionStrategy(object):

    def __init__(self, weights, get_reward_func, population_size=50, sigma=0.1, learning_rate=0.001, decay=1.0):
        np.random.seed(0)
        self.weights = weights
        self.get_reward = get_reward_func
        self.POPULATION_SIZE = population_size
        self.SIGMA = sigma
        self.DECAY = decay
        self.learning_rate = learning_rate

    def _get_weights_try(self, w, p):
        weights_try = []
        for index, i in enumerate(p):
            jittered = self.SIGMA*i
            weights_try.append(w[index] + jittered)
        return weights_try


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
                weights_try = self._get_weights_try(self.weights, population[i])
                rewards[i] = self.get_reward(weights_try)

            rewards = (rewards - np.mean(rewards)) / np.std(rewards)

            for index, w in enumerate(self.weights):
                A = np.array([p[index] for p in population])
                self.weights[index] = w + self.learning_rate/(self.POPULATION_SIZE*self.SIGMA) * np.dot(A.T, rewards).T

            self.learning_rate *= self.DECAY

            if (iteration+1) % print_step == 0:
                print('iter %d. reward: %f' % (iteration+1, self.get_reward(self.weights)))