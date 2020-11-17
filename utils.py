import torchvision.transforms as T
import torch
import urllib
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from datetime import datetime
import os


if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using GPU.")
else:
    device = torch.device('cpu')
    print("GPU isn't available, CPU is being used.")


class Utils:
    def __init__(self, model_name: str):
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
        Initializes mean, standard deviaton and desired image size for resizing when using VGG19 model.
        """

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.stdv = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.img_size = 512

    def load_img(self, path: str):
        """
        Loading image function.

        :param path: Path where the image is stored.
        :return: PIL resized image.
        """

        img = Image.open(path)
        img = self.resize(img)

        return img

    def display_img(self, img_list: list, titles: list, save: bool = False):
        """
        Displays and saves dream images in Output folder if specified.

        :param img_list: List of images to display, first one should always be original image and the rest dream images.
        :param titles: Layers used for creating dream images.
        :param save: Optional argument specifying if dream images should be saved. Default behaviour is not saving them.
        :return:
        """

        print("\n Press ESC to exit.")

        cv.namedWindow("DeepDream", cv.WINDOW_NORMAL)

        titles = ["original"] + titles
        i = 0
        n = len(img_list)

        # Make a dir in Output folder with unique name corresponding to the current date and time
        if save:
            today = datetime.now()
            path = "./Output/" + today.strftime('%H_%M_%S_%d_%m_%Y')
            os.mkdir(path)

        while True:
            if i != 0:
                # We need to denormalize dream image before displaying it
                cv_img = cv.cvtColor(self.denormalize(img_list[i]), cv.COLOR_RGB2BGR)
                if save:
                    filename = path + "/" + titles[i] + ".jpg"
                    cv.imwrite(filename, cv_img * 255)  # mul by 255 because our img is in range [0,1]
            else:
                cv_img = cv.cvtColor(np.array(img_list[0]), cv.COLOR_RGB2BGR)

            cv.imshow("DeepDream", cv_img)

            k = cv.waitKey(100)

            if k == 100:
                i = (i+1)%n
            if k == 97:
                i = (i-1)%n
            if k == 27:
                break

        cv.destroyAllWindows()

    def clip(self, tensor: torch.Tensor):
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

        tensor = (tensor * stdv) + mean  # Inverse of normalization

        return tensor.numpy().transpose(1, 2, 0)
