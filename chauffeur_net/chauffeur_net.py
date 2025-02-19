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

    def compute_loss(self, pred, gt):
        total_loss = 0
        if "waypoint_map" in pred:
            total_loss += self.compute_waypoint_map_loss(pred["waypoint_map"], gt["waypoint_map"])
        return total_loss

    def compute_waypoint_map_loss(self, waypoint_pred, waypoint_gt):
        # the waypoint_pred shape: (batch_size, time_step, 1, 200, 200) 
        # the waypoint_gt shape: (batch_size, time_step, 1, 200, 200)
        #
        # the loss is the sum of the loss of each time step
        # there will be only 1 waypoint at each time_step. In waypoint_gt,
        # there should be a Gaussian distribution at the waypoint location.
        
        # the ground truth is dominated by the negative samples, so we need to use focal loss
        # the equation is taken from https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py#L156-L164
        gamma = 2
        #####
        # 1. Compute loss for positive samples
        #####
        positive_ind = waypoint_gt.eq(1)
        positive_pred = waypoint_pred[positive_ind]
        # positive loss equation: log(positive_pred) * (1 - positive_pred)^2
        positive_loss = torch.log(positive_pred) * torch.pow(1 - positive_pred, gamma)

        #####
        # 2. Compute loss for negative samples
        #####
        negative_ind_gt = waypoint_gt.lt(1)
        negative_pred = waypoint_pred[negative_ind_gt]
        # negative loss equation: log(1 - negative_pred) * negative_pred^2 * (1 - negative_gt)^alpha
        negative_loss = torch.log(1 - negative_pred) * torch.pow(negative_pred, gamma)

        alpha = 4.0
        negative_weight = torch.pow(1 - waypoint_gt[negative_ind_gt], alpha).float()
        negative_loss *= negative_weight

        #####
        # 3. Compute total loss
        #####
        # average loss by number of positive samples
        num_positive_samples = positive_ind.sum()
        # need to take "-" because the computed loss is negative
        waypoint_map_loss = -negative_loss.sum()
        if num_positive_samples != 0:
            waypoint_map_loss = (waypoint_map_loss - positive_loss.sum()) / num_positive_samples

        return waypoint_map_loss

