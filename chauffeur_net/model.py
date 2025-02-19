import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import model_zoo

from torchvision.models.resnet import ResNet,BasicBlock
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils import model_zoo

class Config():
    def __init__(self):
        self.batch_size = 128
        self.num_workers = 4
        self.learning_rate = 0.001
        self.num_epochs = 100
        self.input_image_dim = 10
        self.input_image_height = 800
        self.input_image_width = 800

        self.rnn_channel_num = 10
        self.rnn_time_step = 10
        self.feature_net_out_feature_num = 256
        self.waypoint_map_out_feature_num = 1

        # mps: Apple Silicon GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")


class FeatureNet(ResNet):
    def __init__(self, config: Config, imagenet_trained=False):
        super(FeatureNet, self).__init__(BasicBlock, [2, 2, 2, 2])
        self.config = config
        self.conv1 = nn.Conv2d(config.input_image_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if imagenet_trained:
            self.load_my_state_dict(model_zoo.load_url(ResNet18_Weights.IMAGENET1K_V1.url))

    def forward(self, x):
        # input tensor shape: (batch_size, 10, 800, 800)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)  # there is no maxpooling layer
        x = self.layer2(x)  # includes a maxpooling layer
        x = self.layer3(x) # includes a maxpooling layer
        # the output is with shape (batch_size, 256, 200, 200)
        return x 

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.Parameter):
                if name == "conv1.weight":
                    copy1 = param.data.clone()
                    copy2 = param.data.clone()
                    param = torch.cat([copy1, copy2], 1)
                else:
                    # backwards compatibility for serialized parameters
                    param = param.data
            own_state[name].copy_(param)

class WaypointHeatmapNet(nn.Module):
    def __init__(self, config: Config):
        super(WaypointHeatmapNet, self).__init__()
        self.config = config
        self.conv = nn.Sequential(
            nn.Conv2d(config.rnn_channel_num, config.waypoint_map_out_feature_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(config.waypoint_map_out_feature_num),
            nn.ReLU(inplace=True),
        )
        self.activation = nn.Softmax(dim=-1) # spatial softmax
    
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class AgentRNN(nn.Module):
    def __init__(self, config: Config):
        super(AgentRNN, self).__init__()
        self.config = config
        # input to input feature and hidden layer
        self.x_2_h = nn.Conv2d(config.feature_net_out_feature_num, config.rnn_channel_num, kernel_size=3, stride=1, padding=1, bias=False)
        self.x_2_x = nn.Conv2d(config.feature_net_out_feature_num, config.rnn_channel_num, kernel_size=3, stride=1, padding=1, bias=False)
        # hidden to hidden layer
        self.w_hh = nn.Conv2d(config.rnn_channel_num, config.rnn_channel_num, kernel_size=3, stride=1, padding=1, bias=False)
        # input feature to hidden layer 
        self.w_xh = nn.Conv2d(config.rnn_channel_num, config.rnn_channel_num, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.waypoint_map_net = WaypointHeatmapNet(config)
    
    def forward(self, x):
        h_t = self.x_2_h(x)
        h_t = self.relu(h_t)

        x_t = self.x_2_x(x)
        x_t = self.relu(x_t)

        output_list = []
        for _ in range(self.config.rnn_time_step):
            h_x = self.relu(self.w_xh(x_t))
            h_t = self.relu(self.w_hh(h_t))
            h_t = self.tanh(h_x + h_t)
            output = self.waypoint_map_net(h_t)
            output_list.append(output)    

        # output shape: (batch_size, time_step, feature_num, heatmap_height, heatmap_width)
        output_list = torch.stack(output_list, dim=1)

        return output_list
