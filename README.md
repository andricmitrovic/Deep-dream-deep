# DeepDream

<h3>Introduction</h3>

Implementation of DeepDream in PyTorch based on the Google Research <a href="https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html">blog post</a> (you can read here more about the inspiration for DeepDream and how is it supposed to work). \

You can also find the original implementation github repo <a href="https://github.com/google/deepdream">here</a>.



<h3>About</h3>

This project represents an implementation of DeepDream algorithm in PyTorch.\
You can specify which layers the algorithm should use to produce the dream (note that I implemented the use of only one layer for dreaming the image so when specifying more than one layer it will produce a dream image for each of those layers) by using dream_using_different_layers_display_and_maybe_save function or create an endless number of different dream images by calling dream_iteratively_through_layers_save and specifying num_of_runs_per_layer to the desired number of generated images per layer. \
The dream_iteratively_through_layers_save function will take the input image and for each layer you specified will produce a dream image, after which it will take that dream image as a new input and apply zooming to give us the illusion of going deeper into the dream.




<h3>Implementation</h3>

For generating images, first we need a model which we will tinker with to produce them. The model I used is vgg19 pre-trained on the ImageNet dataset. \
Since we need to get the activations from the specific layer during the forward pass of our input image, we can't use this model out of the box. The model is modified so that it does forward pass with input tensor going through each layer of it until it gets to the desired layer, where we save the activations and return them for later use. \
To get the value representing how much our image lit up the neural network at the chosen layer we use the activations that our model provided for us and apply a loss function on them. The goal is to modify the input image so that it will maximize that value, which is done by calculating the gradient of the input image, so we call the backpropagation algorithm with the total loss we calculated. \
After getting the gradient we simply optimize the input image with it but the trick is that we don't subtract it from the input image, we add it to it to maximize the total loss. \
The important thing before we optimize is to smooth the gradients to get better-looking results and after optimizing to clip the image in the desired range because after some number of iterations some pixel values would blow out of proportions. We repeat these optimizing steps num_iters times. \
On top of this to get better-looking results the algorithm is applied on a differently scaled original image. So basically it downscales the original image desired number of times and we get so-called octaves or image pyramid levels, and we apply the algorithm on each level and combine the result with the original input image to get better-looking dreams since at different scales the DeepDream algorithm will amplify different kinds of features. 


<h3>Results</h3>

In the filename, you can see which layer was used to produce the dream.

<br>

<p align="center">
<img src="https://github.com/andricmitrovic/DeepDream-PyTorch/blob/main/Input/starry_night.jpg" width="400">
<img src="https://github.com/andricmitrovic/DeepDream-PyTorch/blob/main/Beautiful_dreams/starry_night_vgg19_conv3_2.jpg" width="400">
<img src="https://github.com/andricmitrovic/DeepDream-PyTorch/blob/main/Beautiful_dreams/starry_night_vgg19_conv5_1.jpg" width="400">
</p>
<br>

<p align="center">
<img src="https://github.com/andricmitrovic/DeepDream-PyTorch/blob/main/Input/leaves.jpg" width="400">
<img src="https://github.com/andricmitrovic/DeepDream-PyTorch/blob/main/Beautiful_dreams/leaves_vgg19_conv4_2.jpg" width="400">
<img src="https://github.com/andricmitrovic/DeepDream-PyTorch/blob/main/Beautiful_dreams/leaves_vgg19_conv5_4.jpg" width="400">
</p>
<br>

<p align="center">
<img src="https://github.com/andricmitrovic/DeepDream-PyTorch/blob/main/Input/nikola.jpeg" width="400">
<img src="https://github.com/andricmitrovic/DeepDream-PyTorch/blob/main/Beautiful_dreams/nikola_vgg19_conv3_4.jpg" width="400">
</p>
<br>

Video I generated with dream_iteratively_through_layers_save function by reusing the output from the dream as new input with zooming: <br>
https://www.youtube.com/watch?v=1ZqhB3AkeV4&ab_channel=NikolaAndricMitrovic


<h3>Tips for running the code</h3>

VGG19 convolution layers: \
conv1_1, conv1_2, \
conv2_1, conv2_2, \
conv3_1, conv3_2, conv3_3, conv3_4, \
conv4_1, conv4_2, conv4_3, conv4_4, \
conv5_1, conv5_2, conv5_3, conv5_4  \

In main, set which layers you want to use. \
If the path is None the input image will be random noise. \
Pick dream_iteratively_through_layers_save or dream_using_different_layers_display_and_maybe_save for generating dreams. As described above the first function will use each layer in layers list num_of_runs_per_layer times and zoom in the dream, while the second function will generate one dream image per layer in layers. \
In the DeepDreamClass several parameters can be changed to get different looking dreams.


<h3>Materials</h3>

The starting point for my DeepDream project was the <a href="https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html">original blog</a> and the <a href ="https://github.com/google/deepdream">original implementation</a>. \
After that I need more theoretical knowledge and <a href="https://www.youtube.com/watch?v=6rVrh5gnpwk&t=427s&ab_channel=TheAIEpiphany">this</a> youtube tutorial helped me a lot with it. \
From there I tried to follow <a href="https://github.com/gordicaleksa/pytorch-deepdream">the implementation</a> that was complementing the video tutorial but at the time I struggled to understand most of it so I followed the materials that were given in this project and found <a href="https://github.com/Adi-iitd/AI-Art/blob/master/src/StyleTransfer/DeepDream.py">this</a> implementation from where I got the inspiration for the core of my project. It is minimalistic and easy to understand. \
After a good analysis of that project, I started from scratch but I was heavily influenced by it and the core remained somewhat the same, from there I built upon it. \
In the end, I came back to the first repo I mentioned since my results weren't looking as nice as I hoped due to the naive gradient smoothening function. There I realized that more complex gradient smoothening would be needed to get the results I wanted, so incorporated CascadeGaussianSmoothing from that project into mine, while still leaving the old method for comparison.
