import torchvision
import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using GPU.")
else:
    device = torch.device('cpu')
    print("GPU isn't available, CPU is being used.")
