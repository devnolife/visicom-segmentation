"""
Utilities untuk visualisasi hasil segmentasi
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image


def overlay_mask(image, mask, alpha=0.5, color=[0, 0, 255]):
    """
    Overlay mask pada gambar
    
    Args:
        image: Gambar asli (H, W, 3) atau (3, H, W)
        mask: Mask segmentasi (H, W) atau (1, H, W)
        alpha: Transparansi overlay
        color: Warna overlay [R, G, B]
    
    Returns:
        Gambar dengan overlay mask
    """
    # Konversi ke numpy array jika perlu
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    
    # Pastikan format (H, W, 3) untuk image
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    
    # Pastikan format (H, W) untuk mask
    if len(mask.shape) == 3:
        mask = mask.squeeze()
    
    # Normalize image ke [0, 255]
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    # Buat overlay
    overlay = image.copy()
    mask_bool = mask > 0.5
    
    # Apply color pada area mask
    for c in range(3):
        overlay[:, :, c] = np.where(mask_bool, color[c], overlay[:, :, c])
    
    # Blend dengan gambar asli
    result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    return result


def visualize_prediction(image, mask, prediction, save_path=None):
    """
    Visualisasi gambar, ground truth mask, dan prediksi
    
    Args:
        image: Gambar asli
        mask: Ground truth mask
        prediction: Prediksi mask
        save_path: Path untuk menyimpan gambar
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Konversi format jika perlu
    if isinstance(image, np.ndarray):
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
    
    # Gambar asli
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth mask
    if mask is not None:
        if isinstance(mask, np.ndarray) and len(mask.shape) == 3:
            mask = mask.squeeze()
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')
    
    # Prediksi
    if isinstance(prediction, np.ndarray) and len(prediction.shape) == 3:
        prediction = prediction.squeeze()
    axes[2].imshow(prediction, cmap='gray')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')
    
    # Overlay
    overlay = overlay_mask(image, prediction, alpha=0.4, color=[255, 0, 0])
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    return fig


def calculate_metrics(pred, target, threshold=0.5):
    """
    Hitung metrik evaluasi: IoU, Dice, Precision, Recall
    
    Args:
        pred: Prediksi (H, W) atau (N, H, W)
        target: Ground truth (H, W) atau (N, H, W)
        threshold: Threshold untuk binarisasi prediksi
    
    Returns:
        Dictionary berisi metrik
    """
    # Binarisasi prediksi
    pred_binary = (pred > threshold).astype(float)
    target_binary = (target > threshold).astype(float)
    
    # True Positive, False Positive, False Negative
    tp = np.sum((pred_binary == 1) & (target_binary == 1))
    fp = np.sum((pred_binary == 1) & (target_binary == 0))
    fn = np.sum((pred_binary == 0) & (target_binary == 1))
    tn = np.sum((pred_binary == 0) & (target_binary == 0))
    
    # IoU (Intersection over Union)
    intersection = tp
    union = tp + fp + fn
    iou = intersection / (union + 1e-8)
    
    # Dice Coefficient
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    
    # Precision & Recall
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    
    return {
        'iou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }


def create_mask_annotation(image_path, save_dir):
    """
    Helper untuk membuat mask annotation secara manual
    Gunakan tool drawing untuk menandai area banjir
    
    Args:
        image_path: Path ke gambar
        save_dir: Direktori untuk menyimpan mask
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print(f"Membuat mask untuk: {image_path}")
    print("Instruksi: Gunakan Streamlit app dengan drawable canvas untuk membuat mask")
    print(f"Simpan mask ke: {save_dir}")
    
    return image
