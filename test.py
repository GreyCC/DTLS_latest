import torch
import torchvision
import matplotlib.pyplot as plt

sigmoid = torch.nn.Softmax2d()

img = torchvision.io.read_image("test_img.jpg")/255
img = img * 2 - 1

sigmoid_img = (sigmoid(img) + 1) /2

img = (img + 1) / 2
plt.imshow(img.permute(1, 2, 0))
plt.show()

plt.imshow(sigmoid_img.permute(1, 2, 0))
plt.show()

