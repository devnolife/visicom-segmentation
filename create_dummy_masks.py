"""
Script untuk membuat dummy masks untuk testing
CATATAN: Ini hanya untuk testing! Gunakan annotate.py untuk membuat mask yang benar
"""
import cv2
import os
import numpy as np

# Paths
image_dir = "dataset/images"
mask_dir = "dataset/masks"

# Create mask directory
os.makedirs(mask_dir, exist_ok=True)

# Get all images
images = [f for f in os.listdir(image_dir) 
          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

print(f"Creating dummy masks for {len(images)} images...")

for img_file in images:
    # Load image
    img_path = os.path.join(image_dir, img_file)
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Error loading: {img_file}")
        continue
    
    height, width = img.shape[:2]
    
    # Create dummy mask (random blob in the image)
    # CATATAN: Ini hanya contoh! Gunakan annotate.py untuk mask yang benar
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Add some random "flood" areas (for testing purposes only)
    center_x, center_y = width // 2, height // 2
    cv2.ellipse(mask, (center_x, center_y), 
                (width // 4, height // 4), 0, 0, 360, 255, -1)
    
    # Save mask
    mask_name = os.path.splitext(img_file)[0] + '_mask.png'
    mask_path = os.path.join(mask_dir, mask_name)
    cv2.imwrite(mask_path, mask)
    
    print(f"✓ Created: {mask_name}")

print("\n" + "="*50)
print("⚠️  PERINGATAN: Ini adalah dummy masks untuk TESTING!")
print("📝 Untuk hasil yang baik, gunakan: streamlit run annotate.py")
print("    untuk membuat mask yang benar sesuai area banjir!")
print("="*50)
