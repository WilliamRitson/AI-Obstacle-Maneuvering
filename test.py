from agents.qlearning import QLearningAgent

agent = QLearningAgent(
    env_name = 'CartPole-v0',
    epsilon = 1, 
    future_discount = 0.99
)

agent.train(500)
agent.play(10)