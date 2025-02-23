import os
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
class ChauffeurNetDataset(Dataset):
    def __init__(self, data_folder):
        # "data_folder" is the folder that contains the images and the waypoints
        # Both images and waypoints are stored in npz format. The keys are "images" and "waypoints"
        # The images are stored as a list of numpy arrays, each array is a NxHxW image
        # The waypoints are stored as a list of 2d numpy array. Each array contains the following information:
        # - x: the x coordinate of the waypoint
        # - y: the y coordinate of the waypoint
        # - heading: the heading of the waypoint
        # - velocity: the velocity of the waypoint
        # - timestamp: the timestamp of the waypoint
        self.data_folder = data_folder
        # The image should be loaded as NxHxW numpy array
        self.images = []
        # The waypoints should be loaded as a 2d numpy array, and then
        # converted to a image with the same size as the image
        # Each timestamp should have 10 images, 1 for each future waypoint
        # Each image should have 1 channel:
        # - The first channel is probability distribution of the future waypoints
        # The heading and velocity are not used yet in the current implementation
        self.future_waypoints = []

        # load the images
        # there are 2 subfolders under the data_folder
        # the first subfolder is the scenario name folder
        # the second subfolder is the timestamp folder
        for scenario_name in os.listdir(self.data_folder):
            scenario_folder = os.path.join(self.data_folder, scenario_name)
            # Skip if not a directory or is hidden file
            if not os.path.isdir(scenario_folder) or scenario_name.startswith('.'):
                continue
            for timestamp in os.listdir(scenario_folder):
                timestamp_folder = os.path.join(scenario_folder, timestamp)
                # Skip if not a directory or is hidden file
                if not os.path.isdir(timestamp_folder) or timestamp.startswith('.'):
                    continue
                for file in os.listdir(timestamp_folder):
                    if not file.endswith(".npz"):
                        continue
                    data = np.load(os.path.join(timestamp_folder, file))
                    self.images.append(data["images"])
                    # convert the future waypoints to a image
                    future_waypoint_images = data["future_waypoints"]
                    self.future_waypoints.append(future_waypoint_images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.future_waypoints[idx]

# Test dataset
dataset = ChauffeurNetDataset("../images/")

# plot the first waypoint image
# overlap all waypoints images on the first image, each pixel take the maximum probability
for _ in range(5):
    random_idx = np.random.randint(0, len(dataset))
    data, labels = dataset[random_idx]
    image = data[:, :, 0]
    for i in range(len(labels)):
        image = np.maximum(image, labels[i][:, :, 0] * 255.)
    plt.imshow(image)
    plt.show()