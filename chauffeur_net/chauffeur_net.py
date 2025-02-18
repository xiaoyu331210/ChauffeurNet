from model import *

import torch.nn as nn

class ChauffeurNet(nn.Module):
    def __init__(self, config: Config):
        super(ChauffeurNet, self).__init__()
        self.config = config
        self.feature_net = FeatureNet(config)
        self.agent_rnn = AgentRNN(config)   
        
    def forward(self, x):
        x = self.feature_net(x)
        x = self.agent_rnn(x)
        output = {}
        output["waypoint_map"] = x  # shape: (batch_size, time_step, 3, 200, 200)
        return output

    def compute_waypoint_map_loss(self, waypoint_pred, waypoint_gt):
        # the waypoint_pred shape: (batch_size, time_step, 3, 200, 200) 
        # the waypoint_gt shape: (batch_size, time_step, 3, 200, 200)
        # the loss is the sum of the loss of each time step
        # there will be only 1 waypoint at each time_step. In waypoint_gt,
        # there should be a Gaussian distribution at the waypoint location.
    