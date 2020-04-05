import cv2
import numpy as np
import matplotlib.pyplot as plt

def BitQuantizeImage(im,k):
	img = ((im)>>(8-k))
	img2 = ((img)<<(8-k))
	return img2	

def bitget(value, bit_number): 
    return int((value & (1 << bit_number)) != 0) 

def bitplaneslicing(img):
	bit_img = np.unpackbits(img)
	w = img.shape[0]
	h = img.shape[1]
	bitplane1 = np.zeros((w,h),dtype=int)
	bitplane2 = np.zeros((w,h),dtype=int)
	bitplane3 = np.zeros((w,h),dtype=int)
	bitplane4 = np.zeros((w,h),dtype=int)
	bitplane5 = np.zeros((w,h),dtype=int)
	bitplane6 = np.zeros((w,h),dtype=int)
	bitplane7 = np.zeros((w,h),dtype=int)
	bitplane8 = np.zeros((w,h),dtype=int)
	for i in range(w):
		for j in range(h):
			bitplane1[i][j] = bitget(img[i][j],0)
			bitplane2[i][j] = bitget(img[i][j],1)
			bitplane3[i][j] = bitget(img[i][j],2)
			bitplane4[i][j] = bitget(img[i][j],3)
			bitplane5[i][j] = bitget(img[i][j],4)
			bitplane6[i][j] = bitget(img[i][j],5)
			bitplane7[i][j] = bitget(img[i][j],6)
			bitplane8[i][j] = bitget(img[i][j],7)
	plt.figure()
	plt.subplot(2,4,1)
	plt.title('plane1')
	plt.imshow(bitplane1,'gray')		
	plt.subplot(2,4,2)
	plt.title('plane2')
	plt.imshow(bitplane2,'gray')
	plt.subplot(2,4,3)
	plt.title('plane3')
	plt.imshow(bitplane3,'gray')
	plt.subplot(2,4,4)
	plt.title('plane4')
	plt.imshow(bitplane4,'gray')
	plt.subplot(2,4,5)
	plt.title('plane5')
	plt.imshow(bitplane5,'gray')
	plt.subplot(2,4,6)
	plt.title('plane6')
	plt.imshow(bitplane6,'gray')
	plt.subplot(2,4,7)
	plt.title('plane7')
	plt.imshow(bitplane7,'gray')
	plt.subplot(2,4,8)
	plt.title('plane8')
	plt.imshow(bitplane8,'gray')						
	plt.show()		

def lena_op(lena,im1,im2,im3):
	w = lena.shape[0]
	h = lena.shape[1]
	bitplane_lena = np.zeros((w,h),dtype=int)
	for i in range(w):
		for j in range(h):
			bitplane_lena[i][j] = bitget(lena[i][j],4)
		
	lena2 = BitQuantizeImage(lena,2)
	lena3 = BitQuantizeImage(lena,1)

	plt.figure()
	plt.subplot(3,2,1)
	plt.imshow(im1,'gray')
	plt.subplot(3,2,2)
	plt.imshow(bitplane_lena,'gray')
	plt.subplot(3,2,3)
	plt.imshow(im2,'gray')
	plt.subplot(3,2,4)
	plt.imshow(lena2,'gray')
	plt.subplot(3,2,5)
	plt.imshow(im3,'gray')
	plt.subplot(3,2,6)
	plt.imshow(lena3,'gray')	
	plt.show()

im = cv2.imread('../input_data/quantize.jpg')
k = 2
quantized_im = BitQuantizeImage(im,k)
plt.figure()
plt.imshow(quantized_im)
plt.show()

lena = cv2.imread('../input_data/lena.jpg',0)
lena2 = cv2.imread('../input_data/lena2.jpg',0)
lena3 = cv2.imread('../input_data/lena3.jpg',0)
lena1 = cv2.imread('../input_data/lena1.jpg',0)
lena_op(lena,lena1,lena2,lena3)

cameraman = cv2.imread('../input_data/cameraman.png',0)	
bitplaneslicing(cameraman)