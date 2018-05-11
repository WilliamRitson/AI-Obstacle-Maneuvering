import atexit
import importlib
import time
import sys


### Set the environment name: ###
# (`es`` only supports BipedalWalker* and `qlearning` only supports CartPole*)
# ENV_NAME = 'BipedalWalkerHardcore-v2'
ENV_NAME = 'BipedalWalker-v2'


def main(agent, iterations, load=None, save=None):
    start_time = time.time()

    Agent = getattr(importlib.import_module('agents.%s' % agent), 'Agent')

    agent = Agent(ENV_NAME)
    if load:
        agent.load('weights/' + load)
    if save:
        def on_exit():
            agent.save('weights/' + save)
            print('Training finished in {} seconds'.format(time.time() - start_time))
        atexit.register(on_exit)
        agent.train(iterations)
    else:
        agent.play(iterations)


if __name__ == '__main__':
    # Optionally support command line args
    args = sys.argv[1:]
    if len(args) > 0:
        if len(args) >= 2 and len(args) <= 4: main(*args)
        else: sys.exit('Invalid args! See function `main`')
    else:
        ### Uncomment one of these lines at a time: ###

        main('es', 1, load='es_1000.pkl')
        # main('es', 1, load='es_1000_hardcore_scratch.pkl')
        # main('es', 1000, save='es_1000_hardcore_scratch.pkl')
        # main('qlearning', 10, load='qlearning_500.h5')
        # main('qlearning', 500, save='qlearning_500.h5')

