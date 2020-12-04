from Models.vgg19_modified import VGG19_modified
from utils import *
import scipy.ndimage as nd
from tqdm import tqdm
import utils


class DeepDreamClass():
    def __init__(self, image: torch.Tensor, layers: list):
        # Metaparameters

        self.learning_rate = 0.09      # Step_size to use while optimizing image
        self.num_iters = 10            # Number of iterations to run at a single scale

        self.num_scales = 7           # Number of scaled down levels of the original image
        self.scale_coef = 1.6         # Parameter for downsampling image

        self.spatial_shift_size = 32   # Number of pixels to randomly shift image before grad ascent

        self.smooth_function = "CascadeGaussianSmoothing"   # CascadeGaussianSmoothing or GaussianBlur
        self.smooth_coef = 0.5

        self.loss = "MSE"              # MSE or L2
        self.kernel_size = 9

        self.octave_function = "original"
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

            h_shift, w_shift = np.random.randint(-self.spatial_shift_size, self.spatial_shift_size + 1, 2)
            image = ut.random_circular_spatial_shift(image, h_shift, w_shift)


            # Activations at the chosen layer
            activations = self.model(image)

            if self.loss == "L2":
                # Loss will be the L2 norm of the activation maps
                loss = [out.norm() for out in activations]
            else:
                # MSE loss
                loss = [torch.nn.MSELoss(reduction='mean')(out, torch.zeros_like(out)) for out in activations]

            # Total loss roughly represents how much our input image "lit up" the NN
            tot_loss = torch.mean(torch.stack(loss))
            tot_loss.backward()

            grad = image.grad.data #.data.cpu()

            if self.smooth_function == "CascadeGaussianSmoothing":
                sigma = ((i + 1) / self.num_iters) * 2.0 + self.smooth_coef
                grad = utils.CascadeGaussianSmoothing(self.kernel_size, sigma)(grad)  # applies 3 Gaussian kernels
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

            image = ut.random_circular_spatial_shift(image, h_shift, w_shift, should_undo=True)

            # if i % 50 == 0:
            #    print(f"After {i} iterations, Loss: {round(tot_loss.item(), 3)}")

        # In the main loop all tensor calculations are done on cpu so we return image on cpu with that in mind
        return image.data.cpu()

    def octave_made_from_original_img(self, scale):

        scale = self.num_scales - (scale + 1)        # Go from the smallest scaled image to the biggest one

        orig_shape = self.image.size[::-1]  # PIL.size inverts the width and height

        new_shape = [int(shape * (self.scale_coef ** (-scale))) for shape in orig_shape]
        tfms = T.Compose([T.Resize(size=new_shape), T.ToTensor(), ut.normalize])
        octave = tfms(self.image).unsqueeze(0)

        return octave

    def octave_made_from_input_img(self, scale, input_img):

        orig_shape = self.image.size[::-1]  # PIL.size inverts the width and height

        SHAPE_MARGIN = 10
        base_shape = orig_shape
        pyramid_level = scale
        pyramid_ratio = self.scale_coef
        pyramid_size = self.num_scales
        exponent = pyramid_level - pyramid_size + 1
        new_shape = np.round(np.float32(base_shape) * (pyramid_ratio ** exponent)).astype(np.int32)

        #print(new_shape)            # moras shape i onog sranja details da menjas ili ipak ne promeni se posle samo

        if new_shape[0] < SHAPE_MARGIN or new_shape[1] < SHAPE_MARGIN:
            print(
                f'Pyramid size {pyramid_size} with pyramid ratio {pyramid_ratio} gives too small pyramid levels with size={new_shape}')
            print(f'Please change parameters.')
            exit(0)

        if input_img is None:
            input_img = self.image
            tfms = T.Compose([T.Resize(size=new_shape), T.ToTensor(), ut.normalize])
            octave = tfms(input_img).unsqueeze(0)
        else:
            input_img = ut.denormalize(input_img)
            input_img = cv.resize(input_img, (new_shape[1], new_shape[0]))
            octave = T.ToTensor()(input_img).unsqueeze(0)

        return octave

    def init_details(self):

        scale = self.num_scales - 1            # Go from the smallest scaled image to the biggest one

        orig_shape = self.image.size[::-1]     # PIL.size inverts the width and height

        new_shape = [int(shape * (self.scale_coef ** (-scale))) for shape in orig_shape]

        new_shape = [1, 3] + new_shape
        return torch.zeros(new_shape)

    def deepdream(self):

        # !!!!!!!!!!!!!!!!! Try cv pyramids on results instead of using them on the original image !!!!!!!!!!!

        details = self.init_details()  # Init details with 0 tensor with the shape same as the smallest scaled img
        dream_image = None
        input_img = None

        for scale in tqdm(range(self.num_scales)):
            if self.octave_function == "original":
                input_img = self.octave_made_from_original_img(scale)
            elif self.octave_function == "input":
                input_img = self.octave_made_from_input_img(scale, input_img)
            #print(details.shape)
            #print(input_img.shape)
            details = torch.tensor(nd.zoom(details, np.array(input_img.shape) / np.array(details.shape), order=1))

            enhanced_image = details + input_img

            # Try to find the patterns at this scale
            dream_image = self.enhance_patterns(enhanced_image)
            # Extract out the patterns that the model learned at this scale
            details = dream_image - enhanced_image

        return dream_image


if __name__ == "__main__":
    # Define which model and layers we are using
    model_name = "vgg19"


    layers = ['conv1_1', 'conv1_2',
              'conv2_1', 'conv2_2',
              'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
              'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
              'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4']


    #layers = ['conv5_2']                        # Specify the layers for the nn to use to generate the dream
    dream_iteratively = False                   # False if dream is generate with one pass of the program,
                                                # True if you want the output to be fed back to the program.
    num_of_runs = 10                            # If dream_iteratively is True set this to the desired number of runs

    path = "Input/leaves.jpg"            # Path of the input image or None for static image
    #path = None


    ut = Utils(model_name)
    input_img = ut.load_img(path)

    display_list = [input_img]

    if dream_iteratively:

        layers = layers[0] * num_of_runs

        for run in range(num_of_runs):
            print(f"\n{layers[run]} {run}")

            print(input_img.size)

            dream_object = DeepDreamClass(input_img, [layers[run]])
            if path is None:
                dream_object.octave_function = "input"
            output_img = dream_object.deepdream()
            display_list.append(output_img)

            output_img = ut.denormalize(output_img)
            output_img = (output_img*255).astype(np.uint8)

            #output_img = nd.zoom(output_img, 1.1)

            input_img = Image.fromarray(output_img, 'RGB')

            layers[run] = layers[run] + "_" + str(run)
    else:
        for layer in layers:
            print(f"\n{layer}")
            dream_object = DeepDreamClass(input_img, [layer])
            if path is None:
                dream_object.octave_function = "input"
            output_img = dream_object.deepdream()
            display_list.append(output_img)

    ut.display_img(display_list, layers, save=True)
