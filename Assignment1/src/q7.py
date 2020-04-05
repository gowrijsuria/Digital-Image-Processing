import numpy as np 
import cv2 
import matplotlib.pyplot as plt 

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

	return lena2

def histEqualization1(lena):

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

im = cv2.imread('../input_data/lena.jpg',0)
eq = histEqualization(im)
eq1 = eq
dbl_eq = histEqualization1(eq1)

# plt.figure()
# plt.subplot(1,3,1)
# plt.imshow(im,'gray')
# plt.subplot(1,3,2)
# plt.imshow(equ,'gray')
# plt.subplot(1,3,3)
# plt.imshow(equl,'gray')
# plt.show()

plt.figure()
plt.subplot(3,2,1)
plt.imshow(im,'gray')
plt.subplot(3,2,2)
plt.hist(im)
plt.subplot(3,2,3)
plt.imshow(eq,'gray')
plt.subplot(3,2,4)
plt.hist(eq)
plt.subplot(3,2,5)
plt.imshow(dbl_eq,'gray')
plt.subplot(3,2,6)
plt.hist(dbl_eq)
plt.show()

dark = cv2.imread('../input_data/dark2.jpg',0)
light = cv2.imread('../input_data/light.jpg',0)

hist1 = cv2.imread('../input_data/hist1.jpg',0)
hist2 = cv2.imread('../input_data/hist2.jpg',0)
# dark = cv2.imread('church.png')
# light = cv2.imread('sky.jpg')

hist = histMatching(hist1,hist2)	
light1 = cv2.resize(light,(1023,677))

dark_to_light = histMatching(dark,light1)	
light_to_dark = histMatching(light1,dark)	

plt.figure()
plt.subplot(1,3,1)
plt.imshow(hist1,'gray')
plt.subplot(1,3,2)
plt.imshow(hist2,'gray')
plt.subplot(1,3,3)
plt.imshow(hist,'gray')
plt.show()

plt.figure()
plt.subplot(1,3,1)
plt.imshow(dark,'gray')
plt.subplot(1,3,2)
plt.imshow(light1,'gray')
plt.subplot(1,3,3)
plt.imshow(dark_to_light,'gray')
plt.show()

plt.figure()
plt.subplot(1,3,1)
plt.imshow(light1,'gray')
plt.subplot(1,3,2)
plt.imshow(dark,'gray')
plt.subplot(1,3,3)
plt.imshow(light_to_dark,'gray')
plt.show()
