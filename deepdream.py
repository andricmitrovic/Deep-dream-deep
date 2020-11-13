from Models.vgg19_modified import VGG19_modified
from utils import *


class DeepDreamClass():
    def __init__(self, image: torch.Tensor, layers: list):
        self.octave_scale = 1.30
        self.learning_rate = 5e-3
        self.image = image
        self.model = VGG19_modified(layers)

    def deepdream(self):
        pass


if __name__ == "__main__":
    # Define which model and layers we are using
    model_name = "vgg19"
    layers = ['conv4_4']

    ut = Utils(model_name)
    input_img = ut.load_img("./starry_night.jpg")

    dream_object = DeepDreamClass(input_img, layers)
    #output = dream_object.deepdream()

    ut.display_img([input_img, input_img], layers)
