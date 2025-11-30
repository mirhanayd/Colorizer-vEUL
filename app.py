import streamlit as st
import numpy as np
import cv2
import os

# ---------------------------
# Load Colorization Model
# ---------------------------
prototxt_path = "models/colorization_deploy_v2.prototxt"
model_path = "models/colorization_release_v2.caffemodel"
kernel_path = "models/pts_in_hull.npy"

# Check if model files exist
if not os.path.exists(prototxt_path):
    st.error(f"❌ Model file not found: {prototxt_path}")
    st.stop()
if not os.path.exists(model_path):
    st.error(f"❌ Model file not found: {model_path}")
    st.stop()
if not os.path.exists(kernel_path):
    st.error(f"❌ Kernel file not found: {kernel_path}")
    st.stop()

try:
    # Load network
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    pts = np.load(kernel_path)

    # Set cluster centers
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId("class8_ab")).blobs = [pts.astype(np.float32)]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# ---------------------------
# Colorize Function
# ---------------------------
def colorize_image(img):
    H, W = img.shape[:2]

    normalized = img.astype("float32") / 255.0
    lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (W, H))

    L_original = cv2.split(lab)[0]

    colorized = np.concatenate((L_original[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = (colorized * 255).astype("uint8")

    return colorized

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="AI Colorizer", page_icon="🎨", layout="centered")

st.title("🎨 AI Image Colorizer")
st.write("Upload a black & white image and let the model bring it to life.")

# Get list of images from images folder
images_folder = "images"
available_images = []
if os.path.exists(images_folder):
    available_images = [f for f in os.listdir(images_folder) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

# Create tabs for different input methods (Upload first)
tab1, tab2 = st.tabs(["� Upload Your Own", "�📁 Select from Images Folder"])

bw_img = None
source_name = None

with tab1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded_file is not None:
        # Read as numpy array
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        bw_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        source_name = uploaded_file.name

with tab2:
    if available_images:
        # Professional dropdown with image count
        st.markdown(f"### 📂 Image Library")
        st.caption(f"💡 {len(available_images)} images available in your /images folder")
        
        # Styled selectbox
        selected_image = st.selectbox(
            "Choose an image to colorize:",
            options=[""] + sorted(available_images),
            format_func=lambda x: "-- Select an image --" if x == "" else f"🖼️ {x}",
            help="Select from your local image collection"
        )
        
        # Show preview when image is selected
        if selected_image:
            image_path = os.path.join(images_folder, selected_image)
            preview_img = cv2.imread(image_path)
            
            if preview_img is not None:
                # Get image dimensions
                h, w = preview_img.shape[:2]
                file_size = os.path.getsize(image_path) / 1024  # KB
                
                st.divider()
                
                # Preview section with info
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("#### 🔍 Preview")
                    st.image(preview_img, channels="BGR", width=400)
                
                with col2:
                    st.markdown("#### 📊 Image Info")
                    st.metric("Filename", selected_image)
                    st.metric("Dimensions", f"{w} × {h} px")
                    st.metric("File Size", f"{file_size:.1f} KB")
                    
                    # Colorize button
                    if st.button("🎨 Colorize This Image", type="primary", use_container_width=True):
                        bw_img = preview_img
                        source_name = selected_image
                
                # Set image for processing (also triggered by button)
                if bw_img is None:
                    bw_img = preview_img
                    source_name = selected_image
    else:
        st.warning(f"⚠️ No images found in '{images_folder}' folder. Please add some images or use the Upload tab.")

if bw_img is not None:

    st.divider()
    
    with st.spinner("Colorizing... ⏳"):
        output = colorize_image(bw_img)

    st.success("✅ Colorization complete!")
    
    # ---------------------------
    # Yan yana gösterim
    # ---------------------------
    col1, col2 = st.columns(2)
    col1.subheader("Original Image")
    col1.image(bw_img, channels="BGR", use_container_width=True)

    col2.subheader("Colorized Result")
    col2.image(output, channels="BGR", use_container_width=True)

    # ---------------------------
    # Download button
    # ---------------------------
    result_bgr = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    _, result_png = cv2.imencode(".png", result_bgr)
    
    download_filename = f"colorized_{source_name}" if source_name else "colorized.png"
    st.download_button(
        label="📥 Download Colorized Image",
        data=result_png.tobytes(),
        file_name=download_filename,
        mime="image/png",
        use_container_width=True
    )