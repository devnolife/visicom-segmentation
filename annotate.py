"""
Tool untuk membuat anotasi mask untuk dataset segmentasi banjir
"""
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import cv2
import os


st.set_page_config(
    page_title="Flood Mask Annotation Tool",
    page_icon="✏️",
    layout="wide"
)


def main():
    st.title("✏️ Flood Mask Annotation Tool")
    st.markdown("**Tool untuk membuat anotasi mask area banjir pada gambar**")
    
    # Sidebar
    st.sidebar.header("⚙️ Pengaturan")
    
    # Directories
    image_dir = st.sidebar.text_input("Image Directory", "dataset")
    mask_dir = st.sidebar.text_input("Mask Output Directory", "dataset/masks")
    
    # Create mask directory if not exists
    os.makedirs(mask_dir, exist_ok=True)
    
    # Drawing settings
    st.sidebar.markdown("### 🎨 Pengaturan Drawing")
    stroke_width = st.sidebar.slider("Brush Size", 1, 50, 20)
    stroke_color = st.sidebar.color_picker("Brush Color", "#FFFFFF")
    bg_color = "#000000"
    
    # Load images
    if os.path.exists(image_dir):
        image_files = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        
        if image_files:
            st.sidebar.markdown("---")
            st.sidebar.markdown(f"📁 **{len(image_files)} gambar ditemukan**")
            
            # Select image
            selected_image = st.sidebar.selectbox("Pilih Gambar", image_files)
            
            if selected_image:
                # Load image
                image_path = os.path.join(image_dir, selected_image)
                image = Image.open(image_path)
                
                # Display image info
                st.sidebar.info(f"Ukuran: {image.size[0]} x {image.size[1]}")
                
                # Main content
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🖼️ Gambar Asli")
                    st.image(image, use_column_width=True)
                
                with col2:
                    st.subheader("✏️ Gambar Mask (Gambar area banjir)")
                    st.markdown("**Gunakan brush untuk menandai area banjir (warna putih)**")
                    
                    # Canvas
                    canvas_result = st_canvas(
                        fill_color="rgba(255, 255, 255, 0.3)",
                        stroke_width=stroke_width,
                        stroke_color=stroke_color,
                        background_color=bg_color,
                        background_image=image,
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
                        if canvas_result.image_data is not None:
                            # Convert to grayscale mask
                            mask = canvas_result.image_data[:, :, :3].astype(np.uint8)
                            mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
                            
                            # Binarize
                            _, mask_binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
                            
                            # Save
                            mask_path = os.path.join(mask_dir, mask_filename)
                            cv2.imwrite(mask_path, mask_binary)
                            
                            st.success(f"✅ Mask disimpan: {mask_path}")
                        else:
                            st.warning("⚠️ Belum ada mask yang digambar!")
                
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
                    for idx, mask_file in enumerate(saved_masks[:8]):  # Show max 8
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
    
    # Instructions
    with st.expander("📖 Panduan Penggunaan"):
        st.markdown("""
        ### Cara Menggunakan Annotation Tool
        
        1. **Pilih Gambar**: Pilih gambar dari dropdown di sidebar
        2. **Gambar Area Banjir**: Gunakan brush untuk menandai area yang terkena banjir (warna putih)
        3. **Sesuaikan Brush**: Atur ukuran dan warna brush sesuai kebutuhan
        4. **Simpan Mask**: Klik tombol "Simpan Mask" setelah selesai menggambar
        5. **Ulangi**: Pilih gambar berikutnya dan ulangi proses
        
        ### Tips
        - Gunakan brush size yang besar untuk area luas
        - Gunakan brush size kecil untuk detail
        - Pastikan seluruh area banjir tercover dengan warna putih
        - Mask akan disimpan sebagai gambar grayscale binary (0 atau 255)
        - Area putih = banjir, Area hitam = tidak ada banjir
        """)


if __name__ == '__main__':
    main()
