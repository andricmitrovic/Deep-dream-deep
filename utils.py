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
        self.mean = np.array([])
        self.stdv = np.array([])
        self.img_size = 0
        self.model_name = model_name

        # Map desired init functions for each model, one for now
        model_mapping = {"vgg19": self.vgg19_param_init}
        # Init mean, standard deviation and desired image size for resizing
        model_mapping[model_name]()

        self.normalize = T.Normalize(mean=self.mean, std=self.stdv)
        self.resize = T.Resize(size=self.img_size, interpolation=2)

    def vgg19_param_init(self):
        """
        Initializes mean, standard deviaton and desired image size for resizing when using VGG19 model
        """
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.stdv = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.img_size = 512

    def load_img(self, path):
        """
        Loading image function.
        :param path: Path where the image is stored.
        :return: PIL resized image.
        """
        img = Image.open(path)
        img = self.resize(img)

        return img

    def display_img(self, img_list, titles):
        """
        Displays original and dream images.

        :param img_list: List of images to display, first one should be original one always.
        :param titles: List of layers used for dream images.
        """

        titles = ["original"] + titles

        f = plt.figure(figsize=(16, 8))

        for i in range(len(img_list)):
            f.add_subplot(1, len(img_list), i + 1)
            plt.title(titles[i])
            plt.axis('off')
            if i > 0:
                plt.imshow(self.denormalize(img_list[i]))
            else:
                plt.imshow(img_list[i])
        plt.show()

    def clip(self, tensor):
        """
        Clips the image in desired [min,max] pixel bounds.

        :param tensor: Image to be clipped.
        :return: Clipped tensor.
        """

        for channel in range(tensor.shape[1]):
            ch_m, ch_s = self.mean[channel], self.stdv[channel]
            tensor[0, channel] = torch.clamp(tensor[0, channel], -ch_m / ch_s, (1 - ch_m) / ch_s)

        return tensor

    def denormalize(self, tensor: torch.Tensor):
        """
        Denormalizes a tensor by multiplying it by stdv and adding mean, and then converts to a PIL image

        :param tensor: Tensor to be denormalized.
        :return: Denormalized tensor.
        """

        tensor = tensor.squeeze(0)  # Tensor format is [batch_sz, channels, w, h] we remove batch_sz which is 1 always
        stdv = self.stdv.reshape((3, 1, 1))
        mean = self.mean.reshape((3, 1, 1))

        tensor = (tensor * stdv) + mean # Inverse of normalization
        tensor = T.ToPILImage()(tensor)

        return tensor
