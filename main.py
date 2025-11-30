#libraries
import numpy as np
import cv2
#NumPy used for numeric operations
#OpenCV used for image processing and DNN inference


#Models link:https://github.com/richzhang/colorization/tree/caffe/colorization/models
#Points: https://github.com/richzhang/colorization/blob/caffe/colorization/resources/pts_in_hull.npy
#İnspired by:https://github.com/opencv/opencv/blob/master/samples/dnn/colorization.py and NeuralNine Youtube
#link https://drive.google.com/drive/folders/1FaDajjtAsntF_Sw5gqF0WyakviA5l8-a

#specifying the path
prototxt_path = 'models/colorization_deploy_v2.prototxt'
model_path = 'models/colorization_release_v2.caffemodel'
kernel_path = 'models/pts_in_hull.npy'
image_path = 'images/d.jpg'
#Prototxt: network architecture
#Caffemodel: pretrained weights
#pts_in_hull.npy: 313 ab cluster points
#Image path: grayscale image to colorize

#specifying net
#readNetFromCaffe we use this function to read pre trained models
net = cv2.dnn.readNetFromCaffe(prototxt_path,model_path)
points = np.load(kernel_path)

#reshaping,defining ,loading
points = points.transpose().reshape(2,313,1,1)
#it reshapes data into the format expected by the model.
net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
#Injects the ab cluster points into the class8_ab layer.
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1,313],2.606,dtype="float32")]
#Adds a bias term used by the colorization model.

#LAB-> L=Lightness a* b*

#normalizing the image part
bw_image = cv2.imread(image_path)
#it reads the grayscale/BGR image.
normalized = bw_image.astype("float32")/255.0
#Scales pixel values to the range [0,1].
lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
#Converts the image to LAB color space (L: lightness, a/b: color).


#resizing image
resized = cv2.resize(lab,(224,224))
#The model requires a fixed 224×224 input.Dont change the number
L = cv2.split(resized)[0]
#it extracts the L (lightness) channel.
L -=50
#centering the L channeel,you can change it for example -30 L is gonna be more bright ,-70 L is gonna be more darker

#running the model paert
net.setInput(cv2.dnn.blobFromImage(L))
#Creates a blob and feeds the L channel into the network.
ab = net.forward()[0, :, :, :].transpose((1,2,0))
#Gets the predicted a/b channels from the network.

#resizing the output image
ab=cv2.resize(ab, (bw_image.shape[1],bw_image.shape[0]))
#Upscales the predicted ab channels to match the original image size.
L =cv2.split(lab)[0]
#Extracts the original high-resolution L channel.

#Merge LAB channels and convert back to BGR
colorized = np.concatenate((L[:,:,np.newaxis], ab), axis=2)
#Combines L + a + b to form a full LAB image.
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
#Converts LAB back to BGR color space.
colorized = (255.0 * colorized).astype("uint8")
#Rescales to 0–255 and converts to 8-bit.

#displaying result part
cv2.imshow("BW Image",bw_image)
cv2.imshow("Colorized", colorized)
#Shows the original and the colorized image.
cv2.waitKey(0)
cv2.destroyAllWindows()
#Keeps windows open until a key is pressed.