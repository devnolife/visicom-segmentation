"""
Streamlit App untuk Flood Segmentation - All in One
"""
import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import io
import os
from streamlit_drawable_canvas import st_canvas
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from models.unet import get_model
from utils.data_loader import get_inference_transform
from utils.visualization import overlay_mask
from utils.water_detection import detect_water_hsv, combine_detection_methods


# Konfigurasi halaman
st.set_page_config(
    page_title="Flood Segmentation System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model(model_path, device='cpu'):
    """Load model dengan caching"""
    try:
        model = get_model('unet', n_channels=3, n_classes=2)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model, checkpoint.get('val_iou', 'N/A')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


def predict_flood(image, model, device, image_size=512):
    """Prediksi segmentasi banjir"""
    # Preprocess
    transform = get_inference_transform(image_size)
    image_rgb = np.array(image)
    original_size = image_rgb.shape[:2]
    
    transformed = transform(image=image_rgb)
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_mask = probs[:, 1, :, :].cpu().numpy()[0]
    
    # Resize to original
    pred_mask_resized = cv2.resize(pred_mask, (original_size[1], original_size[0]))
    
    return pred_mask_resized


def annotation_tool_page():
    """Halaman Annotation Tool dengan HSV Auto-detection"""
    st.title("✏️ Flood Mask Annotation Tool")
    st.markdown("**Tool untuk membuat anotasi mask area banjir dengan deteksi HSV otomatis**")
    
    # Settings
    col_settings1, col_settings2, col_settings3 = st.columns([2, 1, 1])
    
    with col_settings1:
        image_dir = st.text_input("Image Directory", "dataset/images")
        mask_dir = st.text_input("Mask Output Directory", "dataset/masks")
    
    with col_settings2:
        stroke_width = st.slider("Brush Size", 1, 50, 20)
        
    with col_settings3:
        hsv_sensitivity = st.selectbox("HSV Sensitivity", ["low", "medium", "high"], index=1)
    
    stroke_color = "#FFFFFF"
    bg_color = "#000000"
    
    # Create mask directory
    os.makedirs(mask_dir, exist_ok=True)
    
    # Load images
    if os.path.exists(image_dir):
        image_files = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        
        if image_files:
            st.info(f"📁 **{len(image_files)} gambar ditemukan**")
            
            # Select image
            selected_image = st.selectbox("Pilih Gambar untuk Anotasi", image_files)
            
            if selected_image:
                # Load image
                image_path = os.path.join(image_dir, selected_image)
                image = Image.open(image_path)
                image_np = np.array(image)
                
                st.info(f"Ukuran: {image.size[0]} x {image.size[1]}")
                
                # Auto-detect button
                col_auto1, col_auto2, col_auto3 = st.columns([2, 1, 1])
                
                with col_auto1:
                    st.markdown("### 🤖 Auto-Detection")
                    
                with col_auto2:
                    if st.button("🔍 Deteksi HSV", help="Deteksi otomatis area air/banjir"):
                        with st.spinner("Mendeteksi area banjir..."):
                            # Detect water using HSV
                            auto_mask = detect_water_hsv(image_np, sensitivity=hsv_sensitivity)
                            st.session_state['auto_mask'] = auto_mask
                            st.success("✅ Deteksi selesai! Refine di canvas jika perlu.")
                
                with col_auto3:
                    if st.button("🔄 Reset Mask"):
                        if 'auto_mask' in st.session_state:
                            del st.session_state['auto_mask']
                        st.rerun()
                
                # Prepare background image for canvas
                if 'auto_mask' in st.session_state:
                    # Create overlay for visualization
                    auto_mask = st.session_state['auto_mask']
                    overlay_img = image_np.copy()
                    overlay_img[auto_mask > 0] = [255, 100, 100]  # Red tint for detected areas
                    background_image = Image.fromarray(overlay_img)
                else:
                    background_image = image
                
                # Main content
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🖼️ Gambar Asli")
                    st.image(image, use_column_width=True)
                
                with col2:
                    st.subheader("✏️ Edit Mask")
                    st.markdown("**🤖 Auto-detect HSV atau gambar manual dengan brush**")
                    
                    # Canvas
                    canvas_result = st_canvas(
                        fill_color="rgba(255, 255, 255, 0.3)",
                        stroke_width=stroke_width,
                        stroke_color=stroke_color,
                        background_color=bg_color,
                        background_image=background_image,
                        update_streamlit=True,
                        height=image.size[1],
                        width=image.size[0],
                        drawing_mode="freedraw",
                        key="canvas",
                    )
                
                # Save mask
                st.markdown("---")
                
                col_save1, col_save2, col_save3 = st.columns([2, 1, 1])
                
                with col_save1:
                    mask_filename = st.text_input(
                        "Nama file mask",
                        value=os.path.splitext(selected_image)[0] + "_mask.png"
                    )
                
                with col_save2:
                    if st.button("💾 Simpan Mask", type="primary"):
                        final_mask = None
                        
                        # Prioritas: canvas drawing > auto detection
                        if canvas_result.image_data is not None:
                            # Convert canvas to mask
                            mask = canvas_result.image_data[:, :, :3].astype(np.uint8)
                            mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
                            _, canvas_mask = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
                            
                            # Combine with auto mask if exists
                            if 'auto_mask' in st.session_state:
                                auto_mask = st.session_state['auto_mask']
                                # Combine: auto + manual refinement
                                final_mask = cv2.bitwise_or(auto_mask, canvas_mask)
                            else:
                                final_mask = canvas_mask
                        
                        elif 'auto_mask' in st.session_state:
                            # Only auto detection
                            final_mask = st.session_state['auto_mask']
                        
                        if final_mask is not None:
                            # Save
                            mask_path = os.path.join(mask_dir, mask_filename)
                            cv2.imwrite(mask_path, final_mask)
                            st.success(f"✅ Mask disimpan: {mask_path}")
                            
                            # Clear auto mask after saving
                            if 'auto_mask' in st.session_state:
                                del st.session_state['auto_mask']
                        else:
                            st.warning("⚠️ Belum ada mask! Gunakan deteksi HSV atau gambar manual.")
                
                with col_save3:
                    if st.button("🔄 Reset Canvas"):
                        st.rerun()
                
                # Preview saved masks
                st.markdown("---")
                st.subheader("📂 Mask yang Sudah Disimpan")
                
                saved_masks = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
                
                if saved_masks:
                    st.info(f"Total mask tersimpan: {len(saved_masks)}")
                    
                    # Display saved masks
                    mask_cols = st.columns(4)
                    for idx, mask_file in enumerate(saved_masks[:8]):
                        with mask_cols[idx % 4]:
                            mask_path = os.path.join(mask_dir, mask_file)
                            mask_img = Image.open(mask_path)
                            st.image(mask_img, caption=mask_file, use_column_width=True)
                else:
                    st.info("Belum ada mask yang disimpan")
        else:
            st.warning(f"⚠️ Tidak ada gambar di direktori: {image_dir}")
    else:
        st.error(f"❌ Direktori tidak ditemukan: {image_dir}")


@st.cache_resource
def load_huggingface_model():
    """Load SegFormer model from HuggingFace"""
    try:
        processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        return processor, model
    except Exception as e:
        st.error(f"Error loading HuggingFace model: {e}")
        return None, None


def predict_with_huggingface(image, processor, model):
    """Prediksi menggunakan SegFormer dari HuggingFace"""
    # Preprocess
    inputs = processor(images=image, return_tensors="pt")
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Resize to original size
    logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False
    )
    
    # Get segmentation map
    seg_map = logits.argmax(dim=1)[0].cpu().numpy()
    
    # SegFormer ADE20K: class 21 = water, class 60 = river
    # Create binary mask for water-related classes
    water_classes = [21, 60, 26]  # water, river, sea
    water_mask = np.isin(seg_map, water_classes).astype(np.uint8)
    
    return water_mask


def prediction_page():
    """Halaman Prediksi/Inference"""
    st.title("🔍 Flood Detection & Segmentation")
    st.markdown("**Upload gambar untuk deteksi dan segmentasi area banjir**")
    
    # Model selection
    st.sidebar.header("🤖 Pilih Model")
    model_type = st.sidebar.radio(
        "Model Type:",
        ["Custom U-Net (Trained)", "HuggingFace SegFormer (Pre-trained)"],
        help="Pilih model untuk inference"
    )
    
    # Sidebar settings
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Pengaturan")
    
    if model_type == "Custom U-Net (Trained)":
        # Custom model settings
        model_path = st.sidebar.text_input(
            "Path Model",
            value="checkpoints/best_model.pth"
        )
        
        # Device selection
        device_option = st.sidebar.selectbox(
            "Device",
            ["CPU", "CUDA (GPU)"]
        )
        device = 'cuda' if device_option == "CUDA (GPU)" and torch.cuda.is_available() else 'cpu'
        
        # Image size
        image_size = st.sidebar.slider("Ukuran Input Model", 256, 1024, 512, 32)
        
        # Threshold
        threshold = st.sidebar.slider("Threshold Deteksi", 0.0, 1.0, 0.5, 0.05)
        
        st.sidebar.markdown("---")
        st.sidebar.info(f"Device aktif: **{device.upper()}**")
        
        # Load model
        if os.path.exists(model_path):
            with st.spinner("Loading custom model..."):
                model, val_iou = load_model(model_path, device)
            
            if model is not None:
                st.sidebar.success("✅ Custom Model loaded!")
                if val_iou != 'N/A':
                    st.sidebar.metric("Validation IoU", f"{val_iou:.4f}")
            else:
                st.error("Gagal memuat model!")
                return
        else:
            st.warning(f"⚠️ Model tidak ditemukan di: `{model_path}`")
            st.info("Silakan latih model terlebih dahulu menggunakan `train.py`")
            return
    
    else:  # HuggingFace SegFormer
        st.sidebar.info("📦 **Model**: nvidia/segformer-b0-finetuned-ade-512-512")
        st.sidebar.markdown("Pre-trained on ADE20K dataset")
        
        # Load HuggingFace model
        with st.spinner("Loading HuggingFace model..."):
            processor, hf_model = load_huggingface_model()
        
        if processor is None or hf_model is None:
            st.error("Gagal memuat HuggingFace model!")
            return
        
        st.sidebar.success("✅ HuggingFace Model loaded!")
        device = 'cpu'
        threshold = st.sidebar.slider("Threshold Deteksi", 0.0, 1.0, 0.5, 0.05)
    
    # Overlay alpha (common for both)
    alpha = st.sidebar.slider("Transparansi Overlay", 0.0, 1.0, 0.4, 0.05)
    
    # Upload and analyze
    # File uploader
    uploaded_file = st.file_uploader(
        "Pilih gambar banjir untuk dianalisis",
        type=['jpg', 'jpeg', 'png', 'webp'],
        help="Upload gambar dalam format JPG, PNG, atau WEBP"
    )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file).convert('RGB')
        
        # Create columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🖼️ Gambar Asli")
            st.image(image, use_column_width=True)
        
        # Predict button
        if st.button("🔍 Analisis Banjir", type="primary"):
            with st.spinner("Menganalisis gambar..."):
                # Predict based on model type
                if model_type == "Custom U-Net (Trained)":
                    pred_mask = predict_flood(image, model, device, image_size)
                    binary_mask = (pred_mask > threshold).astype(np.uint8)
                else:  # HuggingFace
                    binary_mask = predict_with_huggingface(image, processor, hf_model)
                
                # Calculate metrics
                total_pixels = binary_mask.size
                flood_pixels = np.sum(binary_mask)
                flood_percentage = (flood_pixels / total_pixels) * 100
                
                # Create overlay
                image_np = np.array(image)
                overlay = overlay_mask(image_np, binary_mask, alpha=alpha, color=[255, 0, 0])
            
            with col2:
                st.subheader("🎯 Hasil Segmentasi")
                st.image(overlay, use_column_width=True)
            
            # Metrics
            st.markdown("---")
            st.subheader("📈 Statistik Analisis")
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric(
                    "Area Banjir",
                    f"{flood_percentage:.2f}%",
                    help="Persentase area yang terdeteksi banjir"
                )
            
            with metric_col2:
                st.metric(
                    "Pixel Banjir",
                    f"{flood_pixels:,}",
                    help="Jumlah pixel yang teridentifikasi sebagai banjir"
                )
            
            with metric_col3:
                severity = "Rendah" if flood_percentage < 20 else "Sedang" if flood_percentage < 50 else "Tinggi"
                st.metric(
                    "Tingkat Keparahan",
                    severity,
                    help="Estimasi tingkat keparahan banjir"
                )
            
            # Download results
            st.markdown("---")
            st.subheader("💾 Download Hasil")
            
            download_col1, download_col2 = st.columns(2)
            
            with download_col1:
                # Save mask
                mask_pil = Image.fromarray((binary_mask * 255).astype(np.uint8))
                mask_buffer = io.BytesIO()
                mask_pil.save(mask_buffer, format='PNG')
                mask_buffer.seek(0)
                
                st.download_button(
                    label="📥 Download Mask",
                    data=mask_buffer,
                    file_name="flood_mask.png",
                    mime="image/png"
                )
            
            with download_col2:
                # Save overlay
                overlay_pil = Image.fromarray(overlay)
                overlay_buffer = io.BytesIO()
                overlay_pil.save(overlay_buffer, format='PNG')
                overlay_buffer.seek(0)
                
                st.download_button(
                    label="📥 Download Overlay",
                    data=overlay_buffer,
                    file_name="flood_overlay.png",
                    mime="image/png"
                )


def about_page():
    """Halaman About"""
    st.title("📊 Tentang Sistem")
    
    st.markdown("""
        ### 🎯 Tujuan
        Sistem ini dikembangkan untuk mendeteksi dan mensegmentasi area banjir secara otomatis 
        menggunakan teknologi Deep Learning, khususnya arsitektur U-Net.
        
        ### 🔬 Teknologi
        - **Model**: U-Net dengan encoder-decoder architecture
        - **Framework**: PyTorch
        - **Augmentasi**: Albumentations
        - **Interface**: Streamlit
        
        ### 📊 Metrik Evaluasi
        - **IoU (Intersection over Union)**: Mengukur overlap antara prediksi dan ground truth
        - **Dice Coefficient**: Metrik similarity untuk segmentasi
        - **Precision & Recall**: Akurasi deteksi area banjir
        
        ### 🎨 Interpretasi Warna
        - 🔴 **Merah**: Area yang terdeteksi sebagai banjir
        - ⚪ **Putih**: Area tanpa banjir
        
    ### ⚡ Performa
    Sistem ini dapat memproses gambar dalam hitungan detik dan memberikan visualisasi 
    yang jelas tentang area yang terdampak banjir.
    """)


def guide_page():
    """Halaman Guide"""
    st.title("📖 Panduan Penggunaan")
    
    st.markdown("""
        ### 1️⃣ Persiapan Model
        ```bash
        # Install dependencies
        pip install -r requirements.txt
        
        # Train model (jika belum ada)
        python train.py --image_dir dataset/images --mask_dir dataset/masks
        ```
        
        ### 2️⃣ Menjalankan Aplikasi
        ```bash
        streamlit run app.py
        ```
        
        ### 3️⃣ Menggunakan Sistem
        1. Upload gambar banjir melalui tab "Upload Gambar"
        2. Klik tombol "Analisis Banjir"
        3. Lihat hasil segmentasi dan statistik
        4. Download hasil jika diperlukan
        
        ### 4️⃣ Pengaturan Advanced
        - **Threshold**: Sesuaikan sensitivitas deteksi (0.3-0.7 recommended)
        - **Transparansi**: Atur transparansi overlay untuk visualisasi yang lebih baik
        - **Image Size**: Ukuran input model (lebih besar = lebih akurat tapi lebih lambat)
        
    ### 5️⃣ Tips
    - Gunakan gambar dengan resolusi yang baik untuk hasil optimal
    - Threshold 0.5 biasanya memberikan hasil yang seimbang
    - Gunakan GPU jika tersedia untuk inferensi yang lebih cepat
    """)
    
    st.info("💡 **Tips**: Untuk hasil terbaik, gunakan gambar dengan pencahayaan yang baik dan area banjir yang jelas terlihat.")


def main():
    # Header
    st.title("🌊 Flood Segmentation System")
    st.markdown("**Sistem deteksi dan segmentasi area banjir menggunakan Deep Learning**")
    
    # Sidebar Menu
    st.sidebar.title("📋 Menu Navigasi")
    menu = st.sidebar.radio(
        "Pilih Halaman:",
        ["🏠 Home", "✏️ Annotation Tool", "🔍 Detection & Prediction", "📊 About", "📖 Guide"],
        index=0
    )
    
    st.sidebar.markdown("---")
    
    # Routing
    if menu == "🏠 Home":
        st.markdown("""
        ## Selamat Datang di Flood Segmentation System! 👋
        
        ### 🎯 Fitur Utama:
        
        #### ✏️ **Annotation Tool**
        Buat mask/label untuk dataset Anda dengan mudah:
        - Upload gambar dari folder
        - Gambar area banjir menggunakan brush
        - Simpan mask untuk training
        
        #### 🔍 **Detection & Prediction**
        Deteksi area banjir secara otomatis:
        - Upload gambar untuk dianalisis
        - Visualisasi hasil segmentasi
        - Download mask dan overlay
        - Statistik persentase area banjir
        
        #### 📊 **About**
        Informasi tentang teknologi dan metrik yang digunakan
        
        #### 📖 **Guide**
        Panduan lengkap penggunaan sistem
        
        ---
        
        ### 🚀 Quick Start:
        
        1. **Buat Anotasi** → Gunakan menu "Annotation Tool" untuk membuat mask
        2. **Training Model** → Jalankan `python train.py` di terminal
        3. **Prediksi** → Gunakan menu "Detection & Prediction" untuk analisis
        
        ### 📝 Status Dataset:
        """)
        
        # Check dataset status
        image_dir = "dataset/images"
        mask_dir = "dataset/masks"
        
        if os.path.exists(image_dir):
            images = [f for f in os.listdir(image_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
            st.success(f"✅ {len(images)} gambar ditemukan di `{image_dir}`")
        else:
            st.error(f"❌ Folder `{image_dir}` tidak ditemukan")
        
        if os.path.exists(mask_dir):
            masks = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
            if masks:
                st.success(f"✅ {len(masks)} mask ditemukan di `{mask_dir}`")
            else:
                st.warning(f"⚠️ Belum ada mask di `{mask_dir}`. Gunakan Annotation Tool untuk membuat mask.")
        else:
            st.warning(f"⚠️ Folder `{mask_dir}` belum ada")
        
        # Check model
        model_path = "checkpoints/best_model.pth"
        if os.path.exists(model_path):
            st.success(f"✅ Model ditemukan: `{model_path}`")
        else:
            st.warning(f"⚠️ Model belum ada. Jalankan training terlebih dahulu: `python train.py`")
        
    elif menu == "✏️ Annotation Tool":
        annotation_tool_page()
    
    elif menu == "🔍 Detection & Prediction":
        prediction_page()
    
    elif menu == "📊 About":
        about_page()
    
    elif menu == "📖 Guide":
        guide_page()


if __name__ == '__main__':
    main()
