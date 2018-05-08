import atexit
import importlib

ENV_NAME = 'BipedalWalker-v2'

def main(agent, iterations, load=None, save=None, play=False):
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

if __name__ == '__main__':
    main('es', 1, load='es_1000.pkl', play=True)
    # main('es', 10, save='es_10.pkl')
    # main('qlearning', 10, load='qlearning_500.h5', play=True)
    # main('qlearning', 500, save='qlearning_500.h5')
