import cv2 
import matplotlib.pyplot as plt
import numpy as np 

def histEqualization(lena):

	flat_im = lena.flatten()
	w = lena.shape[0]
	h = lena.shape[1]

	histogram,bins = np.histogram(flat_im,256,[0,256]) 

	cdf = histogram.cumsum()
	cdf_normalized = (histogram.max()/ cdf.max()) * cdf
	masked_value = np.ma.masked_equal(cdf,0)
	#print(cdf)

	masked_value = 255/(masked_value.max()-masked_value.min())*(masked_value - masked_value.min())
	
	cdf = np.ma.filled(masked_value,0)
	cdf = cdf.astype('uint8')
	lena2 = cdf[lena]

	plt.figure()
	plt.subplot(2,2,1)
	plt.imshow(lena,'gray')
	plt.subplot(2,2,3)
	plt.hist(lena)
	plt.subplot(2,2,2)
	plt.imshow(lena2,'gray')
	plt.subplot(2,2,4)
	plt.hist(lena2)
	plt.show()
	return lena2
    	
def histMatching(input_im,ref):	
	w = input_im.shape[0]
	h = input_im.shape[1]

	flat_im = input_im.flatten()
	histogram,bins = np.histogram(flat_im,256,[0,256])
	cdf_in = histogram.cumsum()
	flat_ref = ref.flatten()
	histogram_ref,bins_ref = np.histogram(flat_ref,256,[0,256])
	cdf_ref = histogram_ref.cumsum() 

	range_im = np.arange(256)
	inverse = np.interp(cdf_in, cdf_ref, range_im) 
	final_im = np.reshape(inverse[flat_im], input_im.shape)
	final_im = final_im.astype(np.uint8)
	return final_im

def retrieve_im(im1,im2,im3,im4,im):
	w = im.shape[0]
	h = im.shape[1]
	# print(w,h)
	# print(im1.shape,im2.shape)
	# print(im3.shape,im4.shape)
	part1 = im[0:int(w/2)-1,0:int(h/2)]
	part2 = im[0:int(w/2)-1,int(h/2)+1:h-1]
	part3 = im[int(w/2):w-1,0:int(h/2)]
	part4 = im[int(w/2):w-1,int(h/2)+1:h-1]
	# print(im1.shape,part1.shape)
	eq1 = histMatching(im1,part1)	
	eq2 = histMatching(im2,part2)
	eq3 = histMatching(im3,part3)
	eq4 = histMatching(im4,part4)
	# print(eq1.shape,eq2.shape)
	# print(eq3.shape,eq4.shape)
	final = im
	final[0:int(w/2)-1,0:int(h/2)] = eq1[0:173,0:387]
	final[0:int(w/2)-1,int(h/2)+1:h-1] = eq2[0:173,0:386]
	final[int(w/2):w-1,0:int(h/2)] = eq3[0:173,0:387]
	final[int(w/2):w-1,int(h/2)+1:h-1] = eq4[0:173,0:386]
	return final

im = cv2.imread('../input_data/cameraman.png',0)
input_im = cv2.imread('../input_data/eye.png',0)
ref = cv2.imread('../input_data/eyeref.png',0)	
# ref = cv2.cvtColor(eyeref, cv2.COLOR_BGR2GRAY)
im1 = cv2.imread('../input_data/part1.png',0)
im2 = cv2.imread('../input_data/part2.png',0)
im3 = cv2.imread('../input_data/part3.png',0)
im4 = cv2.imread('../input_data/part4.png',0)
im_canyon = cv2.imread('../input_data/canyon.png',0)

# plt.figure()
# plt.subplot(2,2,1)
# plt.imshow(im1,'gray')
# plt.subplot(2,2,2)
# plt.imshow(im2,'gray')
# plt.subplot(2,2,3)
# plt.imshow(im3,'gray')
# plt.subplot(2,2,4)
# plt.imshow(im4,'gray')
# plt.show()

eq = histEqualization(im)
style_im = histMatching(input_im,ref)
plt.figure()
plt.subplot(2,3,1)
plt.imshow(input_im,'gray')
plt.subplot(2,3,2)
plt.imshow(ref,'gray')
plt.subplot(2,3,3)
plt.imshow(style_im,'gray')
plt.subplot(2,3,4)
plt.hist(input_im)
plt.subplot(2,3,5)
plt.hist(ref)
plt.subplot(2,3,6)
plt.hist(style_im)
plt.show()

plt.figure()
plt.subplot(1,2,1)
plt.imshow(im_canyon,'gray')
retrieved_im = retrieve_im(im1,im2,im3,im4,im_canyon)
plt.subplot(1,2,2)
plt.imshow(retrieved_im,'gray')
plt.show()
