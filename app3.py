import os
import requests
from io import BytesIO
from PIL import Image
import uuid
import concurrent.futures 

# ---------------------------
# 🚀 SYSTEM SETTINGS
# ---------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

import streamlit as st
import numpy as np
import cv2

# Page Config
st.set_page_config(
    page_title="AI Colorizer Pro",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# 0. FAST GALLERY
# ---------------------------
SAMPLE_DIR = "sample_images"
if not os.path.exists(SAMPLE_DIR):
    os.makedirs(SAMPLE_DIR)

if 'gallery_id' not in st.session_state:
    st.session_state.gallery_id = str(uuid.uuid4())
if 'selected_image_path' not in st.session_state:
    st.session_state.selected_image_path = None
if 'is_processed' not in st.session_state:
    st.session_state.is_processed = False

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

# ---------------------------
# 1. MODEL LOADING
# ---------------------------
@st.cache_resource
def load_caffe_model():
    prototxt = "models/colorization_deploy_v2.prototxt"
    model_path = "models/colorization_release_v2.caffemodel"
    kernel_path = "models/pts_in_hull.npy"
    if not os.path.exists(model_path): return None, "Caffe missing"
    try:
        net = cv2.dnn.readNetFromCaffe(prototxt, model_path)
        pts = np.load(kernel_path)
        pts = pts.transpose().reshape(2, 313, 1, 1)
        net.getLayer(net.getLayerId("class8_ab")).blobs = [pts.astype(np.float32)]
        net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]
        return net, None
    except Exception as e: return None, str(e)

@st.cache_resource
def load_gan_model():
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    custom_path = "models\VGG-Based-U-Net-GAN.h5" 
    if not os.path.exists(custom_path): return None, "GAN missing"
    try:
        model = load_model(custom_path, compile=False)
        return model, None
    except Exception as e: return None, str(e)

# ---------------------------
# 2. COLORIZATION ENGINE
# ---------------------------
def colorize_engine(img, net_caffe, model_gan, mode, alpha=0.5, saturation=1.0, 
                   green_red_shift=0, blue_yellow_shift=0):
    h, w = img.shape[:2]
    
    # CAFFE
    normalized = img.astype("float32") / 255.0
    lab_c = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
    resized_c = cv2.resize(lab_c, (224, 224))
    L_c = cv2.split(resized_c)[0] - 50
    net_caffe.setInput(cv2.dnn.blobFromImage(L_c))
    ab_c = net_caffe.forward()[0, :, :, :].transpose((1, 2, 0))
    ab_c = cv2.resize(ab_c, (w, h))

    if "Caffe" in mode:
        final_ab = ab_c
    else:
        # GAN
        try:
            gan_input_size = model_gan.input_shape[1]
            if gan_input_size is None: gan_input_size = 256
        except: gan_input_size = 256

        img_resized = cv2.resize(img, (gan_input_size, gan_input_size))
        img_float = img_resized.astype("float32") / 255.0
        lab_gan = cv2.cvtColor(img_float, cv2.COLOR_BGR2LAB)
        l_gan = (lab_gan[:,:,0] / 50.0) - 1.0 
        l_gan = l_gan.reshape(1, gan_input_size, gan_input_size, 1)
        ab_gan = model_gan.predict(l_gan)[0] * 128.0
        
        # Calibration
        ab_gan[:,:,0] += green_red_shift 
        ab_gan[:,:,1] += blue_yellow_shift
        ab_gan = cv2.resize(ab_gan, (w, h))
        
        if "GAN" in mode and "Hybrid" not in mode:
            final_ab = ab_gan
        else:
            final_ab = cv2.addWeighted(ab_gan, alpha, ab_c, 1 - alpha, 0)

    # MERGE
    img_float_full = img.astype("float32") / 255.0
    lab_full = cv2.cvtColor(img_float_full, cv2.COLOR_BGR2LAB)
    l_full = lab_full[:,:,0]
    result_lab = np.concatenate((l_full[:,:,np.newaxis], final_ab), axis=2)
    result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    result = np.clip(result_bgr * 255, 0, 255).astype("uint8")

    if saturation != 1.0:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype("float32")
        hsv[:,:,1] = hsv[:,:,1] * saturation
        hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
        result = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    return result

# ---------------------------
# 3. UI LAYOUT
# ---------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    model_mode = st.radio("Method:", ( "🤖 Professional (Caffe)", "🧪 My Model (GAN)", "🏆 Hybrid Mode"))
    st.markdown("---")
    
    gr_shift, by_shift, blend_val, sat_val = 0, 0, 0.5, 1.0
    if "Hybrid" in model_mode:
        st.subheader("🎨 Adjustments")
        blend_val = st.slider("Model Balance", 0.0, 1.0, 0.6)
        sat_val = st.slider("Vibrance", 0.8, 1.5, 1.5)
        with st.expander("🎛️ Advanced Calibration", expanded=True):
            gr_shift = st.slider("Green 🟢 <-> 🔴 Red", -30, 30, 0)
            by_shift = st.slider("Blue 🔵 <-> 🟡 Yellow", -30, 30, 0)
    elif "Caffe" in model_mode:
        st.subheader("🎨 Adjustments")
        sat_val = st.slider("Vibrance", 0.8, 1.5, 1.5)
        with st.expander("🎛️ Advanced Calibration", expanded=True):
            gr_shift = st.slider("Green 🟢 <-> 🔴 Red", -30, 30, 0)
            by_shift = st.slider("Blue 🔵 <-> 🟡 Yellow", -30, 30, 0)
    elif "GAN" in model_mode:
        st.subheader("🎨 Adjustments")
        sat_val = st.slider("Vibrance", 0.8, 1.5, 1.5)
        with st.expander("🎛️ Advanced Calibration", expanded=True):
            gr_shift = st.slider("Green 🟢 <-> 🔴 Red", -30, 30, 0)
            by_shift = st.slider("Blue 🔵 <-> 🟡 Yellow", -30, 30, 0)
    else:
        st.info("Switch to Hybrid/GAN for more controls.")


st.title("🎨 AI Colorizer Pro")

tab1, tab2 = st.tabs(["📤 Upload Image", "🖼️ Gallery"])

bw_img = None

# --- DÜZELTME BURADA: Callback Fonksiyonu ---
def on_upload_change():
    """Dosya yüklendiğinde çalışır, modu 'uploaded' yapar."""
    st.session_state.selected_image_path = "uploaded"
    st.session_state.is_processed = False

# TAB 1: UPLOAD
with tab1:
    # on_change parametresi ile sadece yükleme anında tetiklenir
    uploaded = st.file_uploader("Upload B&W Image:", type=["jpg", "png", "jpeg"], on_change=on_upload_change)
    
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        # Sadece mod 'uploaded' ise bunu göster, yoksa galeri seçimini koru
        if st.session_state.selected_image_path == "uploaded":
            bw_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# TAB 2: GALLERY
with tab2:
    col_a, col_b = st.columns([6, 1])
    with col_b:
        if st.button("🔄 Refresh"):
            st.session_state.gallery_id = str(uuid.uuid4())
            st.rerun()
    
    with st.spinner("Downloading samples..."):
        sample_images = download_picsum_images_parallel(st.session_state.gallery_id)
    
    cols = st.columns(6)
    for i, img_path in enumerate(sample_images):
        with cols[i]:
            st.image(img_path, use_container_width=True)
            # Seçince modu dosya yoluna çevir ve işlemi başlat
            if st.button(f"Select", key=f"btn_{i}", use_container_width=True):
                st.session_state.selected_image_path = img_path
                st.session_state.is_processed = True
                st.rerun()

# Eğer seçim galeriden ise resmi yükle
if st.session_state.selected_image_path and st.session_state.selected_image_path != "uploaded":
    if os.path.exists(st.session_state.selected_image_path):
        bw_img = cv2.imread(st.session_state.selected_image_path)

# ---------------------------
# 4. EXECUTION
# ---------------------------
if bw_img is not None:
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Original")
        st.image(bw_img, channels="BGR", use_container_width=True)

    # Buton kontrolü (Sadece manuel yüklemede buton gerekli)
    if st.session_state.selected_image_path == "uploaded" and not st.session_state.is_processed:
        if st.button("🚀 Colorize Now", type="primary", use_container_width=True):
            st.session_state.is_processed = True
            st.rerun()
    
    # İşleme
    if st.session_state.is_processed:
        with c2:
            st.subheader("Result")
            net, e1 = load_caffe_model()
            gan, e2 = load_gan_model()
            
            if net and gan:
                try:
                    res = colorize_engine(bw_img, net, gan, model_mode, blend_val, sat_val, gr_shift, by_shift)
                    st.image(res, channels="BGR", use_container_width=True)
                    _, buf = cv2.imencode(".png", res)
                    st.download_button("📥 Download HD", buf.tobytes(), "colorized.png", "image/png", use_container_width=True)
                except Exception as e: st.error(str(e))
            else: st.error(f"Models missing: {e1 or e2}")