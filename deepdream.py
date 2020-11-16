from Models.vgg19_modified import VGG19_modified
from utils import *
import scipy.ndimage as nd
from scipy.ndimage.filters import gaussian_filter

class DeepDreamClass():
    def __init__(self, image: torch.Tensor, layers: list):
        # Metaparameters
        self.learning_rate = 5e-3   # Step_size to use while optimizing image
        self.scales = 5             # Number of different downsampled scales wrt to original image
        self.octave_scale = 1.30    # Parameter for downsampling images
        self.num_iters = 100        # Number of iterations to run at a single scale

        self.image = image
        self.model = VGG19_modified(layers)

    def enhance_patterns(self, image: torch.Tensor):

        # Clone and send image to gpu if available
        image = image.clone()
        image = image.to(device)

        '''
        We set requires_grad to True because we want to calculate the gradient with respect to the input tensor.
        Note that if we set requires_grad to True before cloning, since torch.clone() is a differentiable function 
        it would be included in our computational graph.
        '''
        image.requires_grad_(True)

        for i in range(1, self.num_iters + 1):
            # Output tensor is the features at the chosen layer
            output = self.model(image)
            # Loss will be the L2 norm of the activation maps
            loss = [out.norm() for out in output]
            # Total loss roughly represents how much our input image "lit up" the NN
            tot_loss = torch.mean(torch.stack(loss))
            tot_loss.backward()

            grad = image.grad.data.cpu()
            # Gaussian blur the grads using sigma which increases with the iterations
            sigma = (i * 4.0) / self.num_iters + .5
            grad_1 = gaussian_filter(grad, sigma=sigma * 0.5)
            grad_2 = gaussian_filter(grad, sigma=sigma * 1.0)
            grad_3 = gaussian_filter(grad, sigma=sigma * 2.0)
            grad = torch.tensor(grad_1 + grad_2 + grad_3, device=device)

            '''
            We optimize the image by doing gradient ascent, 
            actually we just add the gradient instead of subtracting like in gradient descent.
            '''
            image.data += self.learning_rate / torch.mean(torch.abs(grad)) * grad
            image.data = ut.clip(image.data)    # Clip image to be in the desired bounds
            image.grad.data.zero_()             # Clear the grad for the next iteration calculations

            if i % 50 == 0:
                print(f"After {i} iterations, Loss: {round(tot_loss.item(), 3)}")

        # In the main loop all tensor calculations are done on cpu so we return image on cpu with that in mind
        return image.data.cpu()

    def deepdream(self):

        # Try using Gaussian pyramids here maybe instead of manual octaves

        # Image pyramid, octaves contain the normalized original image at different scales
        octaves = []
        orig_shape = self.image.size[::-1]  # PIL.size inverts the width and height
        for scale in range(self.scales):
            new_shape = [int(shape * (self.octave_scale ** (-scale))) for shape in orig_shape]
            tfms = T.Compose([T.Resize(size=new_shape), T.ToTensor(), ut.normalize])
            octaves += [tfms(self.image).unsqueeze(0)]
        octaves.reverse()   # Reverse octaves because we want to go from the the smallest scaled image to the original

        details = torch.zeros_like(octaves[0])  # Init details with 0 tensor with the shape same as the smallest scaled img
        dream_image = None

        # For each octave we extract details and combine them with original image at that scale
        for num, octave in enumerate(octaves):
            # Zoom the details tensor to the required size which is the size of the octave we are currently processing
            details = torch.tensor(nd.zoom(details, np.array(octave.shape) / np.array(details.shape), order=1))
            print(f"\n{num+1}/{self.scales} : Current Shape of the details tensor: {details[0].shape}")

            # Combine details tensor which contains patters and original image
            enhanced_image = details + octave

            # Try to find the patterns at this scale
            dream_image = self.enhance_patterns(enhanced_image)
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
    output_img = dream_object.deepdream()

    ut.display_img([input_img, output_img], layers)
