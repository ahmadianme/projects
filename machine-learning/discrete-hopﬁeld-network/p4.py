import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import random

# Set the threshold value
threshold = 180

path = os.getcwd()

path_train = path + '/pacman_Train.jpg'
path_test  = path + '/ghost.jpg'

def read_binarize_img(path_img):

    # Read the image
    img_train = Image.open(path_img).convert(mode="L")
    img_train = img_train.resize(size=(100,100))

    # Binarize the image
    img_train_array = np.asarray(img_train,dtype=np.uint8)
    x = np.zeros(img_train_array.shape,dtype=np.float)
    x[img_train_array > threshold] = 1
    x[x==0] = -1

    return x


# Read images
x = read_binarize_img(path_train)
y = read_binarize_img(path_test)

# Plot images
# plt.imshow(np.repeat(x[:,:,np.newaxis], repeats=3, axis=2))
# plt.show()
# plt.imshow(np.repeat(y[:,:,np.newaxis], repeats=3, axis=2))
# plt.show()

# Create the weights matrix

# Enter your code here ....

# print(x.shape)
# print(y.shape)













# Update the weights matrix until convergence

# Enter your code here ....



dim = len(x)

identity = np.identity(dim*dim)

w = np.zeros((dim*dim, dim*dim))

w = np.outer(x, np.transpose(x))

w -= identity








theta = 0
iterations = 100000




print("Updating y vector...")


yVector = y.flatten()

yVectorLength = len(yVector)
for s in range(iterations):
    i = random.randint(0, yVectorLength - 1)
    u = np.dot(w[i][:], yVector) - theta

    if u > 0:
        yVector[i] = 1
    elif u < 0:
        yVector[i] = -1



yVector = yVector.reshape(y.shape)


output = np.zeros(yVector.shape,dtype=np.uint8)
output[yVector == -1] = 255
output[yVector == 1] = 0
img = Image.fromarray(output, mode="L")
img.show()

print('Done.')
