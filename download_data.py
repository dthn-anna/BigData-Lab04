from torchvision.datasets import CIFAR10
from torchvision import transforms
import os

# Tự động tải nếu chưa có
dataset = CIFAR10(root='./data', train=True, download=True)