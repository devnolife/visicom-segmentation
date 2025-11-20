"""
Streamlit App untuk Flood Segmentation
"""
import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import io
import os

from models.unet import get_model
from utils.data_loader import get_inference_transform
from utils.visualization import overlay_mask


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


def main():
    # Header
    st.title("🌊 Flood Segmentation System")
    st.markdown("**Sistem deteksi dan segmentasi area banjir menggunakan Deep Learning**")
    
    # Sidebar
    st.sidebar.header("⚙️ Pengaturan")
    
    # Model selection
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
    
    # Overlay alpha
    alpha = st.sidebar.slider("Transparansi Overlay", 0.0, 1.0, 0.4, 0.05)
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"Device aktif: **{device.upper()}**")
    
    # Load model
    if os.path.exists(model_path):
        with st.spinner("Loading model..."):
            model, val_iou = load_model(model_path, device)
        
        if model is not None:
            st.sidebar.success("✅ Model loaded!")
            if val_iou != 'N/A':
                st.sidebar.metric("Validation IoU", f"{val_iou:.4f}")
        else:
            st.error("Gagal memuat model!")
            return
    else:
        st.warning(f"⚠️ Model tidak ditemukan di: `{model_path}`")
        st.info("Silakan latih model terlebih dahulu menggunakan `train.py`")
        return
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📸 Upload Gambar", "📊 Tentang", "📖 Panduan"])
    
    with tab1:
        st.header("Upload dan Analisis Gambar")
        
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
                    # Predict
                    pred_mask = predict_flood(image, model, device, image_size)
                    
                    # Binarize
                    binary_mask = (pred_mask > threshold).astype(np.uint8)
                    
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
    
    with tab2:
        st.header("Tentang Sistem")
        
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
    
    with tab3:
        st.header("📖 Panduan Penggunaan")
        
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


if __name__ == '__main__':
    main()
