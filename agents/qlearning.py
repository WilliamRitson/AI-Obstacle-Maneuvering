import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import sgd
import os
import random
from os.path import isfile
from collections import deque
import time

# Returns the product of an iterable (each item multiplied by the next)
def product(iterable):
    product_total = 1
    for x in iterable:
        product_total *= x
    return product_total

class QLearningAgent():
    # The name of the agent, used to generate the weight filename
    __name = "QLearningAgent"
    
    # Initlizes the agent on the named enviroment with given learning parameters
    def __init__(self, env_name, epsilon, future_discount):
        # Create an instance of the enviroment
        self.env = gym.make(env_name)
        # Store the number of inputs (observations) to the enviroment
        self.number_of_states = product(self.env.observation_space.shape)
        # Store the number of outputs (actions) from the enviroment
        self.number_of_actions = self.env.action_space.n 
        # Create a filename to store the weights based on the model and the enviroment
        self.filename = "weights/{}_{}.h5".format(QLearningAgent.__name, env_name)
        # Epsilon learning parameter
        self.epsilon = epsilon
        # Future discount (gamma) learning parameter
        self.future_discount = future_discount
        # Initilize the keras model
        self.__create_model()

    # Creates a keras model for the agent to use for learning/prediction
    # If there are already stored weights, those will be loaded
    def __create_model(self):
        # Defines a keras neural network
        # The initial layer has 8 nerons all desnly connected with a ReLu activation function
        # The second layer has 16 neursons all desnly connected with a ReLu activation function
        # The final (output layer) has a neuron for each possible action with a linear activation function
        self.model = Sequential([
            Dense(8, batch_input_shape = (None, self.number_of_states)),
            Activation('relu'),
            Dense(16),
            Activation('relu'),
            Dense(self.number_of_actions),
            Activation('linear')
        ])
        # User the "adam" optimizer and the "mean squared error" loss function
        self.model.compile('adam', loss = 'mse')
        # If there are already stored weights for the model, load them
        if isfile(self.filename):
            print("[+] Loaded weights from file:", self.filename)
            self.model.load_weights(self.filename)
    
    # Trains the agent's model on a given number of games
    # This should probably be broken into more functions if possible
    def train(self, number_of_games):
        replay = []
        MAX_REPLAY_STATES = 100
        JUMP_FPS = 2
        BATCH_SIZE= 20
        for number_game in range(number_of_games):
            new_state = self.env.reset()
            reward_game = 0
            done = False
            loss = 0
            index_train_per_game = 0
            print( '[+] Starting Game ' + str(number_game))
            while not done:
                self.env.render()
                index_train_per_game += 1
                if random.random() < self.epsilon:
                    action = np.random.randint(self.number_of_actions)
                else:
                    q = self.model.predict(new_state.reshape(1, self.number_of_states))[0]
                    action = np.argmax(q)
                old_state = new_state
                new_state, reward, done, info = self.env.step(action)
                reward_game += reward
                replay.append([new_state, reward, action, done, old_state])
                if len(replay) > MAX_REPLAY_STATES: 
                    replay.pop(np.random.randint(MAX_REPLAY_STATES) + 1)
                if JUMP_FPS != 1 and index_train_per_game % JUMP_FPS == 0:
                    continue # We skip this train, but already add data
                len_mini_batch = min(len(replay), BATCH_SIZE)
                mini_batch = random.sample(replay, len_mini_batch)
                X_train = np.zeros((len_mini_batch, self.number_of_states))
                Y_train = np.zeros((len_mini_batch, self.number_of_actions))
                for index_rep in range(len_mini_batch):
                    new_rep_state, reward_rep, action_rep, done_rep, old_rep_state = mini_batch[index_rep]
                    old_q = self.model.predict(old_rep_state.reshape(1, self.number_of_states))[0]
                    new_q = self.model.predict(new_rep_state.reshape(1, self.number_of_states))[0]
                    update_target = np.copy(old_q)
                    if done_rep:
                        update_target[action_rep] = -1
                    else:
                        update_target[action_rep] = reward_rep + (self.future_discount * np.max(new_q))
                    X_train[index_rep] = old_rep_state
                    Y_train[index_rep] = update_target
                loss += self.model.train_on_batch(X_train, Y_train)
                if reward_game > 200:
                    break
            print( "[+] End Game {} | Reward {} | self.epsilon {:.4f} | TrainPerGame {} | Loss {:.4f} "
                .format(number_game, reward_game, self.epsilon, index_train_per_game, loss / index_train_per_game * JUMP_FPS))
            if self.epsilon >= 0.1:
                self.epsilon -= (1 / (number_of_games))
            if isfile(self.filename):
                os.remove(self.filename)
            self.model.save_weights(self.filename)

    # Has the agent play the game (as currently trained)
    def play(self, repetitions):
        for index_game in range(repetitions):
            print( '[+] {} Playing Game {} of {}'.format(QLearningAgent.__name, index_game +1, repetitions))
            observation = self.env.reset()
            while True:
                self.env.render()
                q = self.model.predict(observation.reshape(1, self.number_of_states))
                action = np.argmax(q)
                observation, reward, done, info = self.env.step(action)
                time.sleep(0.05)
                if done:
                    break

