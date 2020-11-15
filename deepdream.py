from Models.vgg19_modified import VGG19_modified
from utils import *
import scipy.ndimage as nd

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

        # Try using Gaussian pyramids here maybe instead of manual octaves

        # Image pyramid, octaves contain the normalized original image at different scales
        octaves = []
        orig_shape = self.image.size[::-1]  # PIL.size inverts the width and height
        for scale in range(self.scales):
            new_shape = [int(shape * (self.octave_scale ** (-scale))) for shape in orig_shape]
            tfms = T.Compose([T.Resize(size=new_shape), T.ToTensor(), ut.norm])
            octaves += [tfms(self.image).unsqueeze(0)]
        octaves.reverse()   # Reverse octaves because we want to go from the the smallest scaled image to the original

        details = torch.zeros_like(octaves[0])  # Init details with 0 tensor with the shape same as the smallest scaled img
        dream_image = None

        for num, octave in enumerate(octaves):
            # Zoom the details tensor to the required size which is the size of the octave we are currently processing
            details = nd.zoom(details, np.array(octave.shape) / np.array(details.shape), order=1)
            print(f"\n{num+1}/{self.scales} : Current Shape of the details tensor: {details[0].shape}")

            # Combine details tensor which contains patters and original image
            enhanced_image = torch.tensor(details) + octave

            # Try to find the patterns at this scale
            dream_image = self.dream(enhanced_image, nb_iters=self.num_iters, lr=self.learning_rate)
            # Extract out the patterns that the model learned at this scale
            details = dream_image - enhanced_image

        return dream_image


if __name__ == "__main__":
    # Define which model and layers we are using
    model_name = "vgg19"
    layers = ['conv4_4']

    ut = Utils(model_name)
    input_img = ut.load_img("./starry_night.jpg")

    dream_object = DeepDreamClass(input_img, layers)
    output = dream_object.deepdream()

    #ut.display_img([input_img, input_img], layers)
