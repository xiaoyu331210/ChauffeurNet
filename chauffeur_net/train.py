import torch
from model import *
from chauffeur_net import *


chauffeur_net = ChauffeurNet(config)
chauffeur_net.to(config.device)

x = torch.randn((2, 10, 800, 800)).to(config.device)
print(x.shape)

print("forward")
output = chauffeur_net(x)
print(output["waypoint_map"].shape)



