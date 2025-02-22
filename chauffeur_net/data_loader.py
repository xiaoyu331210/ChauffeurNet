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
            for timestamp in os.listdir(scenario_folder):
                timestamp_folder = os.path.join(scenario_folder, timestamp)
                for file in os.listdir(timestamp_folder):
                    if not file.endswith(".npz"):
                        continue
                    data = np.load(os.path.join(timestamp_folder, file))
                    self.images.append(data["images"])
                    # convert the future waypoints to a image
                    image_size = data["images"].shape[1]
                    waypoints = data["future_waypoints"]
                    waypoint_images = []    
                    for i in range(waypoints.shape[1]):
                        waypoint = waypoints[0, i, 0, :]
                        # a 2D guassian distribution will be added around the waypoint
                        # the variance of the gaussian distribution is 0.5
                        # the mean of the gaussian distribution is the waypoint
                        image = np.zeros((image_size, image_size))
                        # add a 2D guassian distribution around the waypoint
                        x = waypoint[0]
                        y = waypoint[1]
                        waypoint_images.append(self.add_gaussian_distribution(image, x, y))
                    self.future_waypoints.append(waypoint_images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.future_waypoints[idx]

    def add_gaussian_distribution(self, image, x, y, radius=20):
        """Add a 2D Gaussian distribution to the image at the specified (x, y) location within a given radius."""
        size = image.shape[0]
        
        # Define the bounds of the region to update
        x_min = max(0, int(x) - radius)
        x_max = min(size, int(x) + radius + 1)
        y_min = max(0, int(y) - radius)
        y_max = min(size, int(y) + radius + 1)
        
        # Create a grid of (i, j) coordinates within the bounds
        x_indices, y_indices = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max), indexing='ij')
        
        # Calculate the Gaussian values for the specified region
        variance = 0.333 * radius
        gaussian_values = np.exp(-((x_indices - x) ** 2 + (y_indices - y) ** 2) / (2 * variance ** 2))
        
        # Add the Gaussian values to the image within the specified region
        image[x_min:x_max, y_min:y_max] += gaussian_values
        
        return image

# Test dataset
dataset = ChauffeurNetDataset("../images/")
print(len(dataset))
print(dataset[0][0].shape)
print(dataset[0][1][0].shape)

# plot the first waypoint image
plt.imshow(dataset[0][1][0])
plt.show()