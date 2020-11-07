from Models.vgg19_modified import VGG19_modified
from utils import device





if __name__ == "__main__":
    model = VGG19_modified(layers=['conv4_4'])