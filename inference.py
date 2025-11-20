"""
Inference Script untuk Prediksi Segmentasi Banjir
"""
import os
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from models.unet import get_model
from utils.data_loader import get_inference_transform
from utils.visualization import overlay_mask, visualize_prediction


class FloodSegmentationPredictor:
    """Predictor untuk segmentasi banjir"""
    
    def __init__(self, model_path, device='cuda', image_size=512):
        """
        Args:
            model_path: Path ke model checkpoint
            device: Device untuk inference
            image_size: Ukuran input gambar
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        self.transform = get_inference_transform(image_size)
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = get_model('unet', n_channels=3, n_classes=2)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
    
    def predict_image(self, image_path):
        """
        Prediksi segmentasi untuk satu gambar
        
        Args:
            image_path: Path ke gambar
        
        Returns:
            original_image, prediction_mask, probability_map
        """
        # Load dan preprocess gambar
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image_rgb.shape[:2]
        
        # Transform
        transformed = self.transform(image=image_rgb)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred_mask = probs[:, 1, :, :].cpu().numpy()[0]  # Probabilitas kelas banjir
        
        # Resize ke ukuran asli
        pred_mask_resized = cv2.resize(pred_mask, (original_size[1], original_size[0]))
        
        # Binarisasi
        binary_mask = (pred_mask_resized > 0.5).astype(np.uint8)
        
        return image_rgb, binary_mask, pred_mask_resized
    
    def predict_batch(self, image_dir, output_dir, save_overlay=True):
        """
        Prediksi batch untuk semua gambar di direktori
        
        Args:
            image_dir: Direktori berisi gambar
            output_dir: Direktori untuk menyimpan hasil
            save_overlay: Simpan overlay mask pada gambar
        """
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)
        if save_overlay:
            os.makedirs(os.path.join(output_dir, 'overlays'), exist_ok=True)
        
        # Daftar file gambar
        image_files = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        
        print(f"Processing {len(image_files)} images...")
        
        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)
            print(f"Processing: {img_file}")
            
            # Predict
            original, binary_mask, prob_map = self.predict_image(img_path)
            
            # Save mask
            mask_name = os.path.splitext(img_file)[0] + '_mask.png'
            mask_path = os.path.join(output_dir, 'masks', mask_name)
            cv2.imwrite(mask_path, binary_mask * 255)
            
            # Save overlay
            if save_overlay:
                overlay = overlay_mask(original, binary_mask, alpha=0.4, color=[255, 0, 0])
                overlay_name = os.path.splitext(img_file)[0] + '_overlay.png'
                overlay_path = os.path.join(output_dir, 'overlays', overlay_name)
                cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            
            # Visualisasi
            fig = visualize_prediction(
                original, None, binary_mask,
                save_path=os.path.join(output_dir, os.path.splitext(img_file)[0] + '_result.png')
            )
            plt.close(fig)
        
        print(f"\nPrediksi selesai! Hasil disimpan di: {output_dir}")
    
    def calculate_flood_percentage(self, image_path):
        """
        Hitung persentase area banjir pada gambar
        
        Args:
            image_path: Path ke gambar
        
        Returns:
            flood_percentage
        """
        _, binary_mask, _ = self.predict_image(image_path)
        
        total_pixels = binary_mask.size
        flood_pixels = np.sum(binary_mask)
        flood_percentage = (flood_pixels / total_pixels) * 100
        
        return flood_percentage


def main():
    parser = argparse.ArgumentParser(description='Flood Segmentation Inference')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--image_path', type=str, help='Path to single image for inference')
    parser.add_argument('--image_dir', type=str, help='Directory containing images for batch inference')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--image_size', type=int, default=512, help='Image size for model input')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        print("Please train the model first using train.py")
        return
    
    # Create predictor
    predictor = FloodSegmentationPredictor(
        model_path=args.model_path,
        device=args.device,
        image_size=args.image_size
    )
    
    # Single image inference
    if args.image_path:
        print(f"\nProcessing single image: {args.image_path}")
        original, binary_mask, prob_map = predictor.predict_image(args.image_path)
        
        # Calculate flood percentage
        flood_pct = predictor.calculate_flood_percentage(args.image_path)
        print(f"Flood coverage: {flood_pct:.2f}%")
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save mask
        mask_path = os.path.join(args.output_dir, 'predicted_mask.png')
        cv2.imwrite(mask_path, binary_mask * 255)
        
        # Save overlay
        overlay = overlay_mask(original, binary_mask, alpha=0.4, color=[255, 0, 0])
        overlay_path = os.path.join(args.output_dir, 'overlay.png')
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        # Visualize
        visualize_prediction(
            original, None, binary_mask,
            save_path=os.path.join(args.output_dir, 'result.png')
        )
        
        print(f"Results saved to {args.output_dir}")
    
    # Batch inference
    elif args.image_dir:
        predictor.predict_batch(
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            save_overlay=True
        )
    
    else:
        print("Error: Please provide either --image_path or --image_dir")


if __name__ == '__main__':
    main()
