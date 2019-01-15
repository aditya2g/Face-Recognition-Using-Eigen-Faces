import sys
import glob,os
import PIL
from PIL import Image as im
import numpy as np
import matplotlib
from matplotlib import pyplot as plt


## Converting image to grayscale resized image and then return data matrix
def convert_image_to_mat(f):
	imsiz = 64
	data = []
	for i in f:
		I = im.open(i).convert('L')
		I = I.resize((imsiz,imsiz), im.ANTIALIAS)
		I1 = np.array(I)
		I1 = I1.flatten()
		data.append(I1)
	data = np.asarray(data)
	return data

## PCA Algorithm Returns the Coefficient matirx and the Eigen Vector Matrix for top 32 eigen vectors
def pca(data,m):
	k = 32     													### Taking top 32 eigen vectors
	data = data - m
	U,S,V = np.linalg.svd(data,full_matrices=True)			
	V = np.transpose(V)
	coeff = np.matmul(data,V)	
	return coeff[:,:k],V[:,:k]

## Calculating score for each image
def calculate_score(s1,ind,train_image):
	max_val = np.max(s1[:,train_image])
	num = np.exp(s1[ind,train_image]-max_val)
	den = 0
	for i in (s1[:,train_image]):
		den = den + np.exp(i-max_val)
	return (num/den)


## Training thr weight matrix
def train_data(coeff,labels,labels_unique):
	W_prev = (np.random.rand(len(labels_unique),coeff.shape[1])*200) - 100
	# print(W.shape)
	N = 1000				## Maximum number of interations
	eta = 0.05				## Learning Rate
	for i in range(N):
		data_update = np.zeros(W_prev.shape)
		s1 = np.matmul(W_prev,np.transpose(coeff))
		for j in range(len(labels)):
			ind = labels_unique.index(labels[j])
			score = calculate_score(s1,ind,j)
			# for k in range(data_update.shape[0]):
				# data_update[k,:] = data_update[k,:] - (score)*coeff[j,:]	
			data_update[ind,:] = data_update[ind,:] + (1-score)*coeff[j,:]
			# print(W_prev.shape)
			# print(data_update.shape)
		W_next = W_prev + eta*data_update
		W_prev = W_next
	return W_next


## Testing the images for trained data
def test_data(test_images,weight,m,V,labels_unique):
	data = convert_image_to_mat(test_images)
	data = data - m
	coeff = np.matmul(data,V)
	ans = []
	for i in range(coeff.shape[0]):
		score = np.matmul(weight,np.transpose(coeff[i,:]))
		score = list(score)
		ind = score.index(max(score))
		ans.append(labels_unique[ind])
	return ans
	


### Main Code starts here
training_file = sys.argv[1]
test_file = sys.argv[2]
images = []
labels = []
test_images = []

## Reading data from Training and Test Files
with open(training_file) as file:
	for line in file:
		line = line.strip()
		arr = line.split()
		images.append(arr[0])
		labels.append(arr[1])

with open(test_file) as file1:
	for line in file1:
		line = line.strip()
		test_images.append(line)

data = convert_image_to_mat(images)
## Getting the mean of input test data
m = np.mean(data, axis=0)

### PCA of training data
### Each column of V represents one eigen vector
coeff, V = pca(data,m)
labels_unique = list(set(labels))
weight = train_data(coeff,labels,labels_unique)
ans = test_data(test_images,weight,m,V,labels_unique)
for i in ans:
	print(i)