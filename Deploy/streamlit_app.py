import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import base64
import imgaug.augmenters as iaa
aug = iaa.Sharpen(alpha=(1.0), lightness=(1.5))
from st_clickable_images import clickable_images
st.title('Brain MR Image segmentation ')
data_load_state = st.text('Loading data...')
gdown --id 1UQIRoLzDCM2vAp0fiQwwuhDXVocPrGXd
unet=load_model('unet.h5',compile=False)
data_load_state.text('Loading data...done!')
st.subheader('Select a image in which you wish to detect tumor')

#=============================================================================
def plot_final(Data,return_image=False):
    image1=cv2.imread(Data)
    image = aug.augment_image(image1)
    image=image[:,:,1]
    image[image <0.2]=0.5
    image = image / 255
    predicted  = unet.predict(image[np.newaxis,:,:])
    predicted[predicted <0.25]=0
    img = predicted[0,:,:,0]
    mean,std=cv2.meanStdDev(img)
    
    pixels = cv2.countNonZero(img)
    image_area = img.shape[0] * img.shape[1]
    area_ratio = (pixels / image_area) * 100
    img = img*255
    img[img<1]=1
    img[img>100]=255
    M= cv2.moments(img)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    if return_image:
      return img,area_ratio,std,(cX,cY)
    else:
      return area_ratio,std,(cX,cY)

#===========================================================================


images = []
for file in ["1.jpeg", "2.jpeg","3.jpeg", "4.jpeg", "5.jpeg"]:
    with open(file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
        images.append(f"data:image/jpeg;base64,{encoded}")

clicked = clickable_images(
    images,
    titles=[f"Image #{str(i)}" for i in range(2)],
    div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
    img_style={"margin": "5px", "height": "200px"},
)
#===========================================================================

if clicked>-1:
  mask,area,std,coordinates = plot_final(str(clicked)+".tif",return_image=True)
  fig = plt.figure()
  plt.imshow(cv2.imread(str(clicked)+".tif"))
  plt.imshow(mask,alpha=0.4,cmap='gray')
  if area==0.0:
      plt.title("No tumor detected", fontsize=20)
  else:
      plt.title("Area={} \n STD={} \n Centroid={}".format(area,std,(coordinates[0],coordinates[1])))
  plt.xticks([])
  plt.yticks([])

  st.pyplot(fig)

else:
  st.markdown("No Image selected")
