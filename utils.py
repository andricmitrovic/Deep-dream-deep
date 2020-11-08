import torchvision
import torch
import urllib
from PIL import Image
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using GPU.")
else:
    device = torch.device('cpu')
    print("GPU isn't available, CPU is being used.")


def load_img_local(path):
    return Image.open(path)


def display_img(img_list, titles):

    titles = ["original"] + titles

    f = plt.figure(figsize=(12, 4))

    for i in range(len(img_list)):
        f.add_subplot(1, len(img_list), i + 1)
        plt.title(titles[i])
        plt.axis('off')
        plt.imshow(img_list[i])
    plt.show()


def load_img_url():
    pass
    url, filename = ("https://raw.githubusercontent.com/andricmitrovic/Dream-of-Samurai/main/dog.jpg", "dog.jpg")
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)

    input_image = Image.open(filename)

