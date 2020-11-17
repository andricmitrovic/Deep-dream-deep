import torch
import torch.nn as nn
import torchvision.models as models
from utils import device


class VGG19_modified(nn.Module):
    def __init__(self, layers : list):
        super(VGG19_modified, self).__init__()

        # mapping layer name to its respected id in model architecture
        mapping_dict = {"conv1_1": 0, "conv1_2": 2,
                        "conv2_1": 5, "conv2_2": 7,
                        "conv3_1": 10, "conv3_2": 12, "conv3_3": 14, "conv3_4": 16,
                        "conv4_1": 19, "conv4_2": 21, "conv4_3": 23, "conv4_4": 25,
                        "conv5_1": 28, "conv5_2": 30, "conv5_3": 32, "conv5_4": 34}

        # +1 so we get ReLU layer after Conv layer
        self.layers = [mapping_dict[layer] + 1 for layer in layers]

        # loading pretrained model and keeping it in eval state
        self.vgg19 = models.vgg19(pretrained=True, progress=True).features
        self.vgg19 = self.vgg19.to(device).eval()

    def forward(self, tensor):

        features = []

        # we feed the input tensor to one layer after another until we get to the wanted layer for dreaming
        # and then we return the output of that layer
        for id, layer in self.vgg19.named_children():
            tensor = layer(tensor)
            if int(id) in self.layers:
                features.append(tensor)
                return features
