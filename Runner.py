from ShipsEnv import ShipsEnv
import numpy as np
import torch
from DumbAI import act as dumact

m = torch.load('log/2000.pt')
env = ShipsEnv(True)
reward = 0

while not reward:
    action, _, _, value = m.forward(torch.autograd.Variable(torch.Tensor(np.expand_dims(env.game_vec, axis=0))))
    action, value = action.data.numpy()[0], value.data.numpy()[0]
    reward, state = env.step(action + np.random.normal(size=4) * 0.3, dumact(env.game_vec))

print(reward)
print(value)


