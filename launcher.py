from SmartAI import PPOAgent
from ShipsEnv import ShipsEnv

env = ShipsEnv(False)
agent = PPOAgent(70, 4)
agent.train(env)
