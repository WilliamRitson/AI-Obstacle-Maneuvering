from bipedal import Agent
import atexit

agent = Agent()

# load pre-trained weights
agent.load('1000.pkl')

# play
agent.play(1)

# train and save weights
# atexit.register(agent.save, '1000.pkl')
# agent.train(1000)
