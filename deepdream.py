from Models.vgg19_modified import VGG19_modified
from utils import *


class DeepDreamClass():
    def __init__(self, image: torch.Tensor, layers: list):
        # Metaparameters
        self.learning_rate = 5e-3   # Step_size to use while optimizing image
        self.scales = 5             # Number of different downsampled scales wrt to original image
        self.octave_scale = 1.30    # Parameter for downsampling images
        self.num_iters = 100        # Number of iterations to run at a single scale

        self.image = image
        self.model = VGG19_modified(layers)

    def deepdream(self):

        # Image pyramid, octaves contain the normalized original image at different scales
        octaves = []
        orig_shape = self.image.size[::-1]  # PIL.size inverts the width and height
        for scale in range(self.scales):
            new_shape = [int(shape * (self.octave_scale ** (-scale))) for shape in orig_shape]
            tfms = T.Compose([T.Resize(size=new_shape), T.ToTensor(), ut.norm])
            octaves += [tfms(self.image).unsqueeze(0)]
        octaves.reverse()   # Reverse octaves because we want to go from the the smallest scaled image to the original



if __name__ == "__main__":
    # Define which model and layers we are using
    model_name = "vgg19"
    layers = ['conv4_4']

    ut = Utils(model_name)
    input_img = ut.load_img("./starry_night.jpg")

    dream_object = DeepDreamClass(input_img, layers)
    output = dream_object.deepdream()

    #ut.display_img([input_img, input_img], layers)
