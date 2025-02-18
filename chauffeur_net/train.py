import torch
from model import *

print(f"PyTorch version: {torch.__version__}")
# print(f"PyTorch config:", torch.__config__.show())
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

a = torch.randn((64, 200, 200))
b = torch.randn((64, 200, 200))

tensor_list = [a, b]

c = torch.stack(tensor_list, dim=1)

print(c.shape)


# test AgentRNN initialization and forward
config = Config()
agent_rnn = AgentRNN(config)
agent_rnn.to(config.device)
# print(agent_rnn)

print("initialize x")
x = torch.randn((10, 256, 200, 200)).to(config.device)

print("forward")
output = agent_rnn(x)

print(output.shape)



