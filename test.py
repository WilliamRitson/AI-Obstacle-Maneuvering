import atexit
import importlib
import time


### Set the environment name: ###
# (`es`` only supports BipedalWalker* and `qlearning` only supports CartPole*)
ENV_NAME = 'BipedalWalkerHardcore-v2'


def main(agent, iterations, load=None, save=None, play=False):
    start_time = time.time()

    Agent = getattr(importlib.import_module('agents.%s' % agent), 'Agent')

    agent = Agent(ENV_NAME)
    if load:
        agent.load('weights/' + load)
    if save:
        atexit.register(agent.save, 'weights/' + save)
    if play:
        agent.play(iterations)
    else:
        agent.train(iterations)
        print('Training finished in {} seconds'.format(time.time() - start_time))


if __name__ == '__main__':
    ### Uncomment one of these lines at a time: ###

    # main('es', 1, load='es_1000_hardcore.pkl', play=True)
    main('es', 1000, save='es_1000_hardcore_scratch.pkl')
    # main('qlearning', 10, load='qlearning_500.h5', play=True)
    # main('qlearning', 500, save='qlearning_500.h5')

