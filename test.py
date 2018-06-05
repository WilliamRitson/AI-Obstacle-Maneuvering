import atexit
import importlib
import time
import sys


### Set the environment name: ###
# (`es`` only supports BipedalWalker* and `qlearning` only supports CartPole*)
ENV_NAME = 'BipedalWalkerHardcore-v2'
# ENV_NAME = 'BipedalWalker-v2'


def main(agent, iterations, play=False, filename=None, envname=ENV_NAME):
    start_time = time.time()

    Agent = getattr(importlib.import_module('agents.%s' % agent), 'Agent')

    if not filename:
        filename = '{}-v{}_{}.pkl'.format(agent, Agent.VERSION, envname)

    agent = Agent(envname)
    if play:
        agent.load('weights/' + filename)
        agent.play(iterations)
    else:
        def on_exit():
            agent.save('weights/' + filename)
            print('Training finished in {} seconds'.format(time.time() - start_time))
        atexit.register(on_exit)
        agent.load('weights/' + filename)
        agent.train(iterations)


if __name__ == '__main__':
    # Optionally support command line args
    args = sys.argv[1:]
    if len(args) > 0:
        if len(args) >= 2 and len(args) <= 5: main(*args)
        else: sys.exit('Invalid args! See function `main`')
    else:
        ### Uncomment one of these lines at a time: ###

        # main('es', 1, play=True)
        main('es', 2000)
        # main('qlearning', 10, play=True)
        # main('qlearning', 500)

