from agents.qlearning import QLearningAgent

agent = QLearningAgent(
    env_name = 'CartPole-v0',
    epsilon = 1, 
    future_discount = 0.99,
    max_replay_states = 100,
    jump_fps = 2,
    bach_size= 20
)

agent.train(500)
agent.play(10)