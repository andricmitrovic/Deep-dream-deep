from Models.vgg19_modified import VGG19_modified
from utils import *





if __name__ == "__main__":

    layers = ['conv4_4']
    model = VGG19_modified(layers=layers)

    input_img = load_img_local("./starry_night.jpg")




    display_img([input_img, input_img], layers)
