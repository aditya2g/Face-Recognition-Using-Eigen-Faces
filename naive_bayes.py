import sys
import glob,os
import PIL
from PIL import Image as im
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import math

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

### Finds the mean and variance of each class of labels
def train_data(coeff,labels,labels_unique):
	trained_data = []
	dataset = {}
	for i in labels_unique:
		dataset[i] = []
	for j in range(len(labels)):
		dataset[labels[j]].append(coeff[j,:])  	
	for i in labels_unique:
		arr = np.array(dataset[i])
		m1 = np.mean(arr,axis = 0)
		var1 = np.var(arr,axis = 0)
		trained_data.append(m1)
		trained_data.append(var1)
	return trained_data


## Finding Normal Probability for given mean and variance
def norm_prob(x,m,v):
	var = (float(v))
	pi = 3.1415926
	denom = (2*pi*var)**.5
	num = math.exp(-(float(x)-float(m))**2/(2*var))
	return num/denom


## Returns the list of labels for test data
def test_data(test_images,m,V,final_data,labels_unique):
	data = convert_image_to_mat(test_images)
	data = data - m
	coeff = np.matmul(data,V)
	ans = []
	p = [None] * coeff.shape[1]
	prob = [None] * len(labels_unique)
	for i in range(coeff.shape[0]):
		for j in range(len(labels_unique)):
			for k in range(coeff.shape[1]):
				p[k] = norm_prob(coeff[i][k],final_data[2*j][k],final_data[2*j+1][k])
			prob[j] = np.prod(p)
		max_val = np.max(prob)
		max_ind = prob.index(max_val)
		ans.append(labels_unique[max_ind])
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

## Training the data
labels_unique = list(set(labels))
final_data = train_data(coeff,labels,labels_unique)
final_data = np.asarray(final_data)

## Testing the test data
ans = test_data(test_images,m,V,final_data,labels_unique)
for i in ans:
	print(i)