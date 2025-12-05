#libraries
import streamlit as st
import numpy as np
import cv2
#NumPy used for numeric operations
#OpenCV used for image processing and DNN inference
#streamlit used for ui

#Models link:https://github.com/richzhang/colorization/tree/caffe/colorization/models
#Points: https://github.com/richzhang/colorization/blob/caffe/colorization/resources/pts_in_hull.npy
#İnspired by:https://github.com/opencv/opencv/blob/master/samples/dnn/colorization.py and NeuralNine Youtube
#link https://drive.google.com/drive/folders/1FaDajjtAsntF_Sw5gqF0WyakviA5l8-a

# loading colorization model
prototxt_path = "models/colorization_deploy_v2.prototxt"
model_path = "models/colorization_release_v2.caffemodel"
kernel_path = "models/pts_in_hull.npy"
#Prototxt: network architecture
#Caffemodel: pretrained weights
#pts_in_hull.npy: 313 ab cluster points (renk kümesi)

# load network
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
pts = np.load(kernel_path)
#readNetFromCaffe we use this function to read pre trained models

# set cluster centers
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId("class8_ab")).blobs = [pts.astype(np.float32)]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [
    np.full([1, 313], 2.606, dtype="float32")
]


# Colorize Function

def colorize_image(img): #take bgr picture turns to colorized one
    H, W = img.shape[:2] #taking shape

    normalized = img.astype("float32") / 255.0 #taking pixels 0-1 instead of 0-255
    lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB) #opencv models works only with LAB format

    resized = cv2.resize(lab, (224, 224)) #making the size 224x224
    L = cv2.split(resized)[0]
    L -= 50 #lightness 

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (W, H))
    #models in 224x224 returns ab output
    #we resize it in original shape

    L_original = cv2.split(lab)[0]

    colorized = np.concatenate((L_original[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = (colorized * 255).astype("uint8")
    #LAB->BGR
    #making 0-255 format again

    return colorized #colorized image


# Streamlit UI

st.set_page_config(page_title="AI Colorizer", page_icon="🎨", layout="centered")

st.title("🎨 AI Image Colorizer")
st.write("Upload a black & white image and let the model bring it to life.")
#up to now settings for streamlit ui

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
#uploading picture

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    bw_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

#colorizing the picture part
    with st.spinner("Colorizing... ⏳"):
        output = colorize_image(bw_img)

    
    # display images side by side
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(bw_img, channels="BGR", use_container_width=True)

    with col2:
        st.subheader("Colorized Result")
        st.image(output, channels="BGR", use_container_width=True)

    
    # Download button

    _, result_png = cv2.imencode(".png", output)

#making the picture in png so we can download it
    st.download_button(
        label="Download Colorized Image",
        data=result_png.tobytes(),
        file_name="colorized.png",
        mime="image/png"
    )

    #to make it run app.py write terminal: python -m streamlit run app.py
    #to run main.py version : python main.py
