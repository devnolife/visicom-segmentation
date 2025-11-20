"""
Script utility untuk membantu setup dan maintenance project
"""
import os
import argparse
import shutil


def create_folders():
    """Membuat struktur folder yang diperlukan"""
    folders = [
        'dataset/images',
        'dataset/masks',
        'checkpoints',
        'outputs',
        'outputs/masks',
        'outputs/overlays',
        'models',
        'utils'
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"✓ Created: {folder}")
    
    print("\n✅ All folders created successfully!")


def check_dataset(image_dir='dataset/images', mask_dir='dataset/masks'):
    """Check dataset integrity"""
    print("Checking dataset...")
    
    if not os.path.exists(image_dir):
        print(f"❌ Image directory not found: {image_dir}")
        return
    
    if not os.path.exists(mask_dir):
        print(f"⚠️  Mask directory not found: {mask_dir}")
        print("   You may need to create masks using annotate.py")
        return
    
    # List images
    images = [f for f in os.listdir(image_dir) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    
    # List masks
    masks = [f for f in os.listdir(mask_dir) 
             if f.lower().endswith('.png')]
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Images: {len(images)}")
    print(f"   Masks: {len(masks)}")
    
    # Check matching
    matched = 0
    unmatched_images = []
    
    for img in images:
        base_name = os.path.splitext(img)[0]
        possible_masks = [
            f"{base_name}_mask.png",
            f"{base_name}.png",
        ]
        
        found = False
        for mask_name in possible_masks:
            if mask_name in masks:
                matched += 1
                found = True
                break
        
        if not found:
            unmatched_images.append(img)
    
    print(f"   Matched pairs: {matched}")
    
    if unmatched_images:
        print(f"\n⚠️  {len(unmatched_images)} images without masks:")
        for img in unmatched_images[:5]:  # Show first 5
            print(f"      - {img}")
        if len(unmatched_images) > 5:
            print(f"      ... and {len(unmatched_images) - 5} more")
    else:
        print("\n✅ All images have corresponding masks!")
    
    # Recommendations
    print("\n💡 Recommendations:")
    if len(images) < 50:
        print("   - Consider adding more images for better model performance")
    if matched < len(images):
        print("   - Use annotate.py to create masks for unmatched images")
    if matched >= 50:
        print("   - You have enough data to start training!")


def clean_outputs(outputs_dir='outputs', checkpoints_dir='checkpoints'):
    """Clean output directories"""
    print("Cleaning outputs...")
    
    dirs_to_clean = [outputs_dir]
    
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            for item in os.listdir(dir_path):
                item_path = os.path.join(dir_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            print(f"✓ Cleaned: {dir_path}")
    
    print("\n✅ Outputs cleaned!")


def test_model(model_path='checkpoints/best_model.pth'):
    """Test if model can be loaded"""
    import torch
    from models.unet import get_model
    
    print(f"Testing model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("   Train a model first using: python train.py")
        return
    
    try:
        model = get_model('unet', n_channels=3, n_classes=2)
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print("✅ Model loaded successfully!")
        print(f"   Validation IoU: {checkpoint.get('val_iou', 'N/A')}")
        print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")


def main():
    parser = argparse.ArgumentParser(description='Utility script for flood segmentation project')
    parser.add_argument('--action', type=str, required=True,
                       choices=['setup', 'check', 'clean', 'test'],
                       help='Action to perform')
    parser.add_argument('--image_dir', type=str, default='dataset/images')
    parser.add_argument('--mask_dir', type=str, default='dataset/masks')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth')
    
    args = parser.parse_args()
    
    if args.action == 'setup':
        create_folders()
    elif args.action == 'check':
        check_dataset(args.image_dir, args.mask_dir)
    elif args.action == 'clean':
        clean_outputs()
    elif args.action == 'test':
        test_model(args.model_path)


if __name__ == '__main__':
    main()
