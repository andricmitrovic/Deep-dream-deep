from Models.vgg19_modified import VGG19_modified
from utils import *





if __name__ == "__main__":

    # Define which model and layers we are using
    ut = Utils(model_name="vgg19")
    layers = ['conv4_4']

    model = VGG19_modified(layers=layers)

    input_img = ut.load_img(path="./starry_night.jpg")

    ut.display_img([input_img, input_img], layers)
