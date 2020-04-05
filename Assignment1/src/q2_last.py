import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import color
from sklearn.cluster import KMeans
import matplotlib as mpl

def colorbar(im, k):

	clusters = KMeans(n_clusters=k)

	clusters.fit(im)
	n_bins = np.arange(0, k+1)
	label = clusters.labels_
	most_dominant_shades = clusters.cluster_centers_
	most_dominant_shades = most_dominant_shades.astype(int)
	# print(most_dominant_shades)

	colorbar_d = np.zeros((70, 700, 3), np.uint8)
	first_position = 0

	(freq, _) = np.histogram(label, bins = n_bins)
	freq = freq.astype("float")
	freq = freq/(freq.sum())
	# shade_freq = zip(most_dominant_shades,hist)
 #    desc_list = sorted(shade_freq,key = lambda x:x[1],reverse = True)
	# desc_list = list(desc_list)
	# shade, freq = zip(*desc_list)
	# print(desc_list)
	
	most_dominant_shades = most_dominant_shades[(-freq).argsort()]
	freq = freq[(-freq).argsort()] 

	for i in range(k):
		r = most_dominant_shades[i][0]
		g = most_dominant_shades[i][1]
		b = most_dominant_shades[i][2]
		second_position = freq[i] * 700 + first_position
		cv2.rectangle(colorbar_d,(int(first_position),0),(int(second_position),70),(r,g,b),-1)
		first_position = second_position	
        
	return colorbar_d

def linContrastStretching(im,a,b):
	w = im.shape[0]
	h = im.shape[1]

	a_low = np.min(im)
	a_high = np.max(im)
	new_im = np.zeros((w,h),dtype = 'uint8')
	# img = im.load()

	for i in range(w):
		for j in range(h):
			new_im[i][j] = a + (im[i][j] - a_low)*(b - a)/(a_high - a_low) 
	return new_im	

img = cv2.imread('../input_data/washed.png',0)

a = 0
b = 255

contrast_img = linContrastStretching(img,a,b)

contrast = np.hstack((img,contrast_img))  

im = cv2.imread('../input_data/washed.png')
col = cv2.imread('../input_data/d.png',0)
colbr = cv2.imread('../input_data/e.png',0)
cv2.imshow('low and high',contrast)
cv2.imshow('colorbar',col)
cv2.imshow('colorbar2',colbr)
cv2.waitKey(0)
k = 3
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im = np.array(im)
im = im.reshape((im.shape[0] * im.shape[1], 3))
im = np.array(im)

contrast = contrast.reshape((contrast.shape[0] * contrast.shape[1], 3))

cbr1 = colorbar(im,k)
plt.figure()
plt.subplot(2,2,1)
plt.axis("off")
plt.imshow(img)

cbr2 = colorbar(contrast,k)
plt.subplot(2,2,2)
plt.axis("off")
plt.imshow(contrast_img)

plt.subplot(2,2,3)
plt.axis("off")
plt.imshow(cb1)
plt.subplot(2,2,4)
plt.axis("off")
plt.imshow(cbr2)

plt.show()

