import cv2 
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def mostfreq_colr(im):
    most_dom_cluster = KMeans(n_clusters = 1)
    colors = most_dom_cluster.fit_predict(im)
    freq_color = Counter(colors)
    #print(freq_color)
    most_freq = most_dom_cluster.cluster_centers_[freq_color.most_common(1)[0][0]]
    return most_freq

def mergeImage(fg,bg,p):
    b0 = p[0]
    g0 = p[1]
    r0 = p[2]
    th1_green = np.array([b0-11,g0-160,r0-24])
    th2_green = np.array([b0+104,g0+18,r0+104])
    full_im = fg
    plt.figure()
    # plt.imshow(fg)
    
    rgb_im = cv2.cvtColor(full_im, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_im)
    plt.show()
    newimg = cv2.inRange(full_im,th1_green,th2_green)

    thru = fg
    thru[newimg != 0] = [0,0,0]
 
    plt.figure()
    plt.imshow(bg)
    plt.show()

    # plt.imshow(thru)
    # plt.show()
    cropbkgnd = bg

    cropbkgnd[newimg == 0] = [0,0,0] 
#     plt.imshow(final)
#     cv2.imshow('image',black_white_bg)
#     cv2.waitKey(0)
#     cv2.de
    final = thru + cropbkgnd
    final_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)

    return final_rgb

# return 

def anotherimg(im,fg,bg):
    most_p = mostfreq_colr(im)
    print(most_p)
    fg_r = cv2.resize(fg,(147,214))
    bg_r = cv2.resize(bg,(147,214))
    imag = mergeImage(fg_r,bg_r,most_p)
    return imag

im = cv2.imread('../input_data/fg.jpg')    
im = im.reshape((im.shape[0] * im.shape[1], 3))
fg = cv2.imread('../input_data/fg.jpg')    
bg = cv2.imread('../input_data/bg.jpg')

p = mostfreq_colr(im)
print(p)
 
chroma_img = mergeImage(fg,bg,p)
plt.figure()
plt.imshow(chroma_img)
plt.show()

im2 = cv2.imread('../input_data/fg2.jpg')    
im2 = im2.reshape((im2.shape[0] * im2.shape[1], 3))
fg2 = cv2.imread('../input_data/fg2.jpg')    
bg2 = cv2.imread('../input_data/bg2.jpg')

anot_img = anotherimg(im2,fg2,bg2)
plt.figure()
plt.imshow(anot_img)
plt.show()


