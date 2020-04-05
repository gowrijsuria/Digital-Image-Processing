import cv2 
import matplotlib.pyplot as plt
import numpy as np 

def generate_8kbit(lena,k):
	im = lena>>(8-k)
	img = im<<(8-k)
	return img

def negative(im,max_intensity):
	im_neg = max_intensity - im
	return im_neg

def gamma_transform(im,gamma):
	g_t_im = im ** gamma
	plt.figure()
	plt.subplot(1,2,1)
	plt.imshow(im,'gray')
	plt.subplot(1,2,2)
	plt.imshow(g_t_im,'gray')
	plt.show()


def piece_transform(im,k1,k2,a,b):
	rows = k1.shape[0]
	# print(rows)
	w = im.shape[0]
	h = im.shape[1]	
	# out = np.zeros((w,h),dtype = int)
	out = im 
	# plt.figure()
	# plt.imshow(im,'gray')
	# plt.show()
	for i in range(rows):
		for j in range(w):
			for k in range(h):
				if(a[i] <= im[j][k] and im[j][k] <= b[i]):
					out[j][k] = k1[i]*im[j][k] + k2[i]
				else:
					out[j][k] = im[j][k]
	return out				

im = cv2.imread('../input_data/lena.jpg',0)	


for k in range(1,9):
	lena_k = generate_8kbit(im,k)
	max_intensity = (2**k) - 1 
	neg = negative(lena_k,max_intensity)
	plt.figure()
	plt.subplot(1,2,1)
	plt.imshow(lena_k,'gray')
	plt.subplot(1,2,2)
	plt.imshow(neg,'gray')	
	plt.show()

gamma_im = cv2.imread('../input_data/gamma-corr.png',0)
gamma = 0.01
g_1 = gamma_transform(gamma_im,gamma)




k1_1 = np.array([0,4/3,-2,0])
k2_1 = np.array([0,0,2,0])
a_1 = np.array([0,0.3,0.6,0.8])
b_1 = np.array([0.3,0.6,0.8,1])

k1_2 = np.array([0,0,0,0,0])
k2_2 = np.array([0,0.2,0.4,0.6,0.8])
a_2 = np.array([0,0.2,0.4,0.6,0.8])
b_2 = np.array([0.2,0.4,0.6,0.8,1])

# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(im,'gray')
# plt.show()

img = im 
normalised_im = (img - np.min(img))/(np.max(img) - np.min(img))
normalised_im_1 = (im - np.min(im))/(np.max(im) - np.min(im))
out_1 = piece_transform(normalised_im,k1_1,k2_1,a_1,b_1)

out_2 = piece_transform(normalised_im_1,k1_2,k2_2,a_2,b_2)

plt.figure()
plt.subplot(1,3,1)
plt.imshow(img,'gray')
plt.subplot(1,3,2)
plt.imshow(out_1,'gray')
plt.subplot(1,3,3)
plt.imshow(out_2,'gray')
plt.show()
