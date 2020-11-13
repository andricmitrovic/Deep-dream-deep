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
        self.img_size = 0
        self.model_name = model_name

        # Map desired init functions for each model, one for now
        model_mapping = {"vgg19": self.vgg19_param_init}
        # Init mean, standard deviation and desired image size for resizing
        model_mapping[model_name]()

        self.norm = T.Normalize(mean=self.mean, std=self.stdv)
        self.resize = T.Resize(size=self.img_size, interpolation=2)

    def vgg19_param_init(self):
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.stdv = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.img_size = 512

    def load_img(self, path):
        img = Image.open(path)
        img = self.resize(img)

        return img

    def display_img(self, img_list, titles):

        titles = ["original"] + titles

        f = plt.figure(figsize=(12, 4))

        for i in range(len(img_list)):
            f.add_subplot(1, len(img_list), i + 1)
            plt.title(titles[i])
            plt.axis('off')
            plt.imshow(img_list[i])
        plt.show()
