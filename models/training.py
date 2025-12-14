# sadece nvidia A100 GPU da çalışması için yazdım (40GB VRAM var çünkü)

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import requests
import uuid
import concurrent.futures
from io import BytesIO
import torch
import warnings
import tensorflow as tf
import shutil 

import pathlib
temp = pathlib.PosixPath
if os.name == 'nt':
    pathlib.PosixPath = pathlib.WindowsPath

st.set_page_config(
    page_title="AI GrandMaster Colorizer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

warnings.filterwarnings("ignore")

SAMPLE_DIR = "sample_images"
MODEL_DIR = "models" 

if not os.path.exists(SAMPLE_DIR): os.makedirs(SAMPLE_DIR)
if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)

MY_GAN_FILE = "VGG-Based-U-Net-GAN.h5"
MY_GAN_PATH = os.path.join(MODEL_DIR, MY_GAN_FILE)
GITHUB_LFS_URL = "https://github.com/mirhanayd/Colorizer-vEUL/raw/main/models/VGG-Based-U-Net-GAN.h5"

DEOLDIFY_WEIGHTS_NAME = "ColorizeArtistic_gen.pth"
DEOLDIFY_WEIGHTS_PATH = os.path.join(MODEL_DIR, DEOLDIFY_WEIGHTS_NAME)
DEOLDIFY_URL = "https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth"

TEMP_IMG_PATH = "temp_process_img.jpg"

if 'gallery_id' not in st.session_state: st.session_state.gallery_id = str(uuid.uuid4())
if 'selected_image_path' not in st.session_state: st.session_state.selected_image_path = None
if 'is_processed' not in st.session_state: st.session_state.is_processed = False

def download_file(url, dest_path, desc="Downloading"):
    if not os.path.exists(dest_path):
        with st.spinner(f"{desc}: {os.path.basename(dest_path)}..."):
            try:
                r = requests.get(url, stream=True)
                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk: f.write(chunk)
            except Exception as e:
                st.error(f"Download Error: {e}")
                return False
    return True

def download_single_image(args):
    i, gallery_id = args
    url = f"https://picsum.photos/seed/{gallery_id}_{i}/600/400?grayscale"
    path = os.path.join(SAMPLE_DIR, f"sample_{i}.jpg")
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img.save(path)
            return path
    except: return None

@st.cache_data(show_spinner=False)
def download_picsum_images_parallel(gallery_id):
    args = [(i, gallery_id) for i in range(6)]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(download_single_image, args))
    return [r for r in results if r is not None]

@st.cache_resource
def load_deoldify_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not download_file(DEOLDIFY_URL, DEOLDIFY_WEIGHTS_PATH, desc="Downloading DeOldify Weights"):
        return None

    try:
        colorizer = torch.hub.load('jantic/DeOldify:master', 'get_image_colorizer', artistic=True)
    except Exception:
        st.warning(" DeOldify önbelleği bozuk. Otomatik temizleniyor, lütfen bekleyin...")
        torch_cache = os.path.join(os.path.expanduser('~'), '.cache', 'torch', 'hub', 'jantic_DeOldify_master')
        if os.path.exists(torch_cache): shutil.rmtree(torch_cache, ignore_errors=True)
        colorizer = torch.hub.load('jantic/DeOldify:master', 'get_image_colorizer', artistic=True)
    
    try:
        state_dict = torch.load(DEOLDIFY_WEIGHTS_PATH, map_location=device)
        if 'model' in state_dict: state_dict = state_dict['model']
        colorizer.learn.model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Local weight loading warning: {e}. Using default weights.")

    colorizer.device = device
    return colorizer

@st.cache_resource
def load_caffe_model():
    prototxt = os.path.join(MODEL_DIR, "colorization_deploy_v2.prototxt")
    caffemodel = os.path.join(MODEL_DIR, "colorization_release_v2.caffemodel")
    pts_path = os.path.join(MODEL_DIR, "pts_in_hull.npy")
    
    urls = {
        prototxt: "https://raw.githubusercontent.com/richzhang/colorization/master/models/colorization_deploy_v2.prototxt",
        pts_path: "https://github.com/richzhang/colorization/raw/master/resources/pts_in_hull.npy",
        caffemodel: "https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1"
    }
    for path, url in urls.items():
        download_file(url, path, desc="Downloading Caffe Component")

    net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
    pts = np.load(pts_path)
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype(np.float32)]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    return net

def colorize_caffe(img_pil, net):
    img = np.array(img_pil.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    img_float = img.astype("float32") / 255.0
    lab = cv2.cvtColor(img_float, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    l_resized = cv2.resize(l_channel, (224, 224))
    l_resized -= 50.0
    net.setInput(cv2.dnn.blobFromImage(l_resized))
    ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab_resized = cv2.resize(ab_channel, (w, h))
    result_lab = np.concatenate((l_channel[:, :, np.newaxis], ab_resized), axis=2)
    result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    result_bgr = np.clip(result_bgr, 0, 1)
    return cv2.cvtColor((result_bgr * 255).astype("uint8"), cv2.COLOR_BGR2RGB)

@st.cache_resource
def load_my_gan_model():
    if not os.path.exists(MY_GAN_PATH) or os.path.getsize(MY_GAN_PATH) < 10000000:
        download_file(GITHUB_LFS_URL, MY_GAN_PATH, desc="Downloading Custom GAN")

    try:
        model = tf.keras.models.load_model(MY_GAN_PATH, compile=False)
        return model, None
    except Exception as e: return None, str(e)

def colorize_my_gan(img_pil, model):
    img = np.array(img_pil.convert("RGB"))
    h, w = img.shape[:2]
    img_resized = cv2.resize(img, (256, 256))
    img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
    img_float = img_bgr.astype("float32") / 255.0
    lab = cv2.cvtColor(img_float, cv2.COLOR_BGR2LAB)
    
    l_channel = lab[:,:,0] 
    l_input = (l_channel / 50.0) - 1.0
    l_input = l_input.reshape(1, 256, 256, 1)
    
    ab_output = model.predict(l_input)[0] 
    ab_resized = cv2.resize(ab_output, (w, h))
    ab_final = ab_resized * 128.0
    
    original_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    original_float = original_bgr.astype("float32") / 255.0
    original_lab = cv2.cvtColor(original_float, cv2.COLOR_BGR2LAB)
    original_l = original_lab[:,:,0]
    
    result_lab = np.concatenate((original_l[:,:,np.newaxis], ab_final), axis=2)
    result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    result_bgr = np.clip(result_bgr, 0, 1)
    
    return cv2.cvtColor((result_bgr * 255).astype("uint8"), cv2.COLOR_BGR2RGB)

with st.sidebar:
    st.header(" Ayarlar")
    
    model_choice = st.radio(
        "Model Seçiniz:",
        (" DeOldify (Pro Kalite)", " My Custom GAN (A100 Trained)", " Caffe (Hızlı)")
    )
    
    st.markdown("---")
    
    render_factor = 35
    if "DeOldify" in model_choice:
        st.info(f"Model: {DEOLDIFY_WEIGHTS_NAME}\nPath: `models/`")
        render_factor = st.slider("Render Factor", 7, 45, 35)
    elif "My Custom GAN" in model_choice:
        st.info(f"Model: `{MY_GAN_FILE}`\nPath: `models/`")
    else:
        st.info("Uses OpenCV DNN.")

st.title(" AI GrandMaster Colorizer")

def on_upload_change():
    st.session_state.selected_image_path = "uploaded"
    st.session_state.is_processed = False

tab1, tab2 = st.tabs([" Resim Yükle", " Galeri"])

current_img = None

with tab1:
    uploaded = st.file_uploader("Siyah Beyaz Resim Yükle:", type=["jpg", "png", "jpeg"], on_change=on_upload_change)
    if uploaded:
        with open(TEMP_IMG_PATH, "wb") as f: f.write(uploaded.getbuffer())
        current_img = Image.open(TEMP_IMG_PATH).convert("RGB")
        st.session_state.selected_image_path = "uploaded"

with tab2:
    col_a, col_b = st.columns([6, 1])
    with col_b:
        if st.button(" Yenile"):
            st.session_state.gallery_id = str(uuid.uuid4())
            st.rerun()
    
    with st.spinner("Galeri yükleniyor..."):
        sample_images = download_picsum_images_parallel(st.session_state.gallery_id)
    
    cols = st.columns(6)
    for i, img_path in enumerate(sample_images):
        with cols[i]:
            st.image(img_path, use_container_width=True)
            if st.button(f"Seç", key=f"btn_{i}", use_container_width=True):
                st.session_state.selected_image_path = img_path
                st.session_state.is_processed = False
                st.rerun()

if st.session_state.selected_image_path == "uploaded":
    if os.path.exists(TEMP_IMG_PATH):
        current_img = Image.open(TEMP_IMG_PATH).convert("RGB")
elif st.session_state.selected_image_path:
    if os.path.exists(st.session_state.selected_image_path):
        current_img = Image.open(st.session_state.selected_image_path).convert("RGB")
        current_img.save(TEMP_IMG_PATH)

if current_img is not None:
    st.divider()
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Orijinal")
        st.image(current_img, use_container_width=True)

    if st.button(" Renklendir", type="primary", use_container_width=True):
        with c2:
            st.subheader("Sonuç")
            
            if "DeOldify" in model_choice:
                with st.spinner("DeOldify (Local Model) çalışıyor..."):
                    try:
                        deoldify_model = load_deoldify_model()
                        if deoldify_model:
                            result_img = deoldify_model.get_transformed_image(path=TEMP_IMG_PATH, render_factor=render_factor)
                            st.image(result_img, use_container_width=True)
                            
                            buf = BytesIO()
                            result_img.save(buf, format="PNG")
                            st.download_button(" İndir", buf.getvalue(), "deoldify_result.png", "image/png", use_container_width=True)
                    except Exception as e: st.error(f"Error: {e}")

            elif "My Custom GAN" in model_choice:
                with st.spinner("Senin modelin (A100 Trained) çalışıyor..."):
                    model, err = load_my_gan_model()
                    if model:
                        try:
                            result_arr = colorize_my_gan(current_img, model)
                            st.image(result_arr, use_container_width=True)
                            
                            res_pil = Image.fromarray(result_arr)
                            buf = BytesIO()
                            res_pil.save(buf, format="PNG")
                            st.download_button(" İndir", buf.getvalue(), "my_gan_result.png", "image/png", use_container_width=True)
                        except Exception as e: st.error(f"İşlem Hatası: {e}")
                    else:
                        st.error(f"Model Yüklenemedi: {err}")

            else:
                with st.spinner("Caffe modeli (Hızlı) çalışıyor..."):
                    try:
                        caffe_net = load_caffe_model()
                        result_arr = colorize_caffe(current_img, caffe_net)
                        st.image(result_arr, use_container_width=True)
                        
                        res_pil = Image.fromarray(result_arr)
                        buf = BytesIO()
                        res_pil.save(buf, format="PNG")
                        st.download_button(" İndir", buf.getvalue(), "caffe_result.png", "image/png", use_container_width=True)
                    except Exception as e: st.error(f"Hata: {e}")