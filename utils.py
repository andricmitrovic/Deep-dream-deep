import torchvision.transforms as T
import torch
import urllib
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using GPU.")
else:
    device = torch.device('cpu')
    print("GPU isn't available, CPU is being used.")


class Utils:
    def __init__(self, model_name):
        self.mean = []
        self.stdv = []

        self.model_name = model_name

        model_mapping = {"vgg19": self.vgg19_param_init}
        model_mapping[model_name]()

    def vgg19_param_init(self):
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.stdv = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def load_img(self, path):
        if self.model_name == "vgg19":
            return Image.open(path)

    def display_img(self, img_list, titles):

        titles = ["original"] + titles

        f = plt.figure(figsize=(12, 4))

        for i in range(len(img_list)):
            f.add_subplot(1, len(img_list), i + 1)
            plt.title(titles[i])
            plt.axis('off')
            plt.imshow(img_list[i])
        plt.show()
