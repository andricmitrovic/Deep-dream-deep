from Models.vgg19_modified import VGG19_modified
from utils import *
import scipy.ndimage as nd
from tqdm import tqdm
import utils


class DeepDreamClass():
    def __init__(self, image: torch.Tensor, layers: list):
        # Metaparameters

        '''
        self.learning_rate = 5e-3      # Step_size to use while optimizing image
        self.scales = 8                # Number of different downsampled scales wrt to original image
        self.octave_scale = 1.30       # Parameter for downsampling images
        self.num_iters = 100           # Number of iterations to run at a single scale
        '''

        # Much faster and same or even better results?
        self.learning_rate = 0.09      # Step_size to use while optimizing image
        self.scales = 4                # Number of different downsampled scales wrt to original image
        self.octave_scale = 1.8        # Parameter for downsampling images
        self.num_iters = 10            # Number of iterations to run at a single scale


        self.use_spatial_shift = True  # Set to False if you don't want to use spatial shifting
        self.spatial_shift_size = 32   # Number of pixels to randomly shift image before grad ascent

        self.smooth_function = "CascadeGaussianSmoothing"   # CascadeGaussianSmoothing or GaussianBlur
        self.smooth_coef = 0.5

        self.loss = "MSE"              # MSE or L2 norm

        self.image = image
        self.model = VGG19_modified(layers)

    def enhance_patterns(self, image: torch.Tensor):

        # Clone and send image to gpu if available
        image = image.clone()
        image = image.to(device)
        '''
        We set requires_grad to True because we want to calculate the gradient with respect to the input tensor,
        since our input image represents a learnable parameter in deep dream.
        Note that if we set requires_grad to True before cloning, since torch.clone() is a differentiable function 
        it would be included in our computational graph.
        '''
        image.requires_grad_(True)

        for i in range(1, self.num_iters + 1):

            if self.use_spatial_shift:
                h_shift, w_shift = np.random.randint(-self.spatial_shift_size, self.spatial_shift_size + 1, 2)
                image = ut.random_circular_spatial_shift(image, h_shift, w_shift)


            # Output tensor is the features at the chosen layer
            output = self.model(image)

            if self.loss == "L2":
                # Loss will be the L2 norm of the activation maps
                loss = [out.norm() for out in output]
            else:
                # MSE loss
                loss = [torch.nn.MSELoss(reduction='mean')(out, torch.zeros_like(out)) for out in output]

            # Total loss roughly represents how much our input image "lit up" the NN
            tot_loss = torch.mean(torch.stack(loss))
            tot_loss.backward()

            grad = image.grad #.data.cpu()

            if self.smooth_function == "CascadeGaussianSmoothing":
                sigma = ((i + 1) / self.num_iters) * 2.0 + self.smooth_coef
                grad = utils.CascadeGaussianSmoothing(3, sigma)(grad)  # applies 3 Gaussian kernels
            elif self.smooth_function == "GaussianBlur":
                sigma = (i * 4.0) / self.num_iters + .5
                grad = ut.gausian_blur(grad, sigma=sigma)

            '''
            We optimize the image by doing gradient ascent, 
            actually we just add the normalized gradient instead of subtracting like in gradient descent.
            '''
            # image.data += self.learning_rate / torch.mean(torch.abs(grad)) * grad
            image.data += self.learning_rate * (grad / torch.std(grad))

            image.data = ut.clip(image.data)    # Clip image to be in the desired bounds
            image.grad.data.zero_()             # Clear the grad for the next iteration calculations

            if self.use_spatial_shift:
                image = ut.random_circular_spatial_shift(image, h_shift, w_shift, should_undo=True)

            # if i % 50 == 0:
            #    print(f"After {i} iterations, Loss: {round(tot_loss.item(), 3)}")

        # In the main loop all tensor calculations are done on cpu so we return image on cpu with that in mind
        return image.data.cpu()

    def deepdream(self):

        # !!!!!!!!!!!!!!!!! Try cv pyramids on results instead of using them on the original image !!!!!!!!!!!

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
        for num, octave in enumerate(tqdm(octaves)):
            # Zoom the details tensor to the required size which is the size of the octave we are currently processing
            details = torch.tensor(nd.zoom(details, np.array(octave.shape) / np.array(details.shape), order=1))
            # print(f"\n{num+1}/{self.scales} : Current Shape of the details tensor: {details[0].shape}")

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

    '''
     layers = ['conv1_1', 'conv1_2',
              'conv2_1', 'conv2_2',
              'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
              'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
              'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4']
    '''

    layers = ['conv5_2']

    ut = Utils(model_name)
    input_img = ut.load_img("./starry_night.jpg")
    #input_img = ut.load_img()

    display_list = [input_img]
    for layer in layers:
        print(f"\n{layer}")
        dream_object = DeepDreamClass(input_img, [layer])
        output_img = dream_object.deepdream()
        display_list.append(output_img)

    ut.display_img(display_list, layers, save=True)
