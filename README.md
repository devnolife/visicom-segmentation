# Segmentation System

Sistem deteksi dan segmentasi area banjir menggunakan Deep Learning dengan arsitektur U-Net.

## 🎯 Fitur

- **Segmentasi Banjir**: Deteksi otomatis area banjir pada gambar
- **U-Net Architecture**: Model deep learning yang powerful untuk segmentasi
- **Data Augmentation**: Augmentasi data untuk meningkatkan performa model
- **Streamlit App**: Interface web yang user-friendly
- **Annotation Tool**: Tool untuk membuat label/mask secara manual
- **Batch Inference**: Prediksi untuk banyak gambar sekaligus

## 📁 Struktur Project

```
banjir/
├── dataset/               # Dataset gambar
│   ├── data-1.jpg
│   ├── data-2.jpeg
│   ├── data-3.jpg
│   └── data-4.webp
├── models/               # Arsitektur model
│   └── unet.py
├── utils/                # Utility functions
│   ├── data_loader.py
│   └── visualization.py
├── checkpoints/          # Model checkpoints
├── outputs/             # Output hasil prediksi
├── train.py             # Script training
├── inference.py         # Script inference
├── app.py              # Streamlit app utama
├── annotate.py         # Annotation tool
├── requirements.txt    # Dependencies
└── README.md
```

## 🚀 Instalasi

### 1. Clone repository atau setup environment

```powershell
git clone https://github.com/devnolife/visicom-segmentation.git
cd visicom-segmentation
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

## 📝 Cara Penggunaan

### Step 1: Buat Anotasi Mask (Opsional, jika belum punya mask)

Jika Anda belum memiliki mask/label untuk dataset, gunakan annotation tool:

```powershell
streamlit run annotate.py
```

Fitur annotation tool:
- Upload gambar dari folder dataset
- Gambar area banjir menggunakan brush
- Simpan mask sebagai file PNG
- Review mask yang sudah dibuat

### Step 2: Training Model

Setelah memiliki dataset gambar dan mask, latih model:

```powershell
# Training dengan default settings
python train.py --image_dir dataset/images --mask_dir dataset/masks

# Training dengan custom settings
python train.py --image_dir dataset/images --mask_dir dataset/masks --epochs 100 --batch_size 8 --lr 0.0001
```

Parameter training:
- `--image_dir`: Direktori berisi gambar training
- `--mask_dir`: Direktori berisi mask/label
- `--epochs`: Jumlah epoch (default: 50)
- `--batch_size`: Ukuran batch (default: 4)
- `--lr`: Learning rate (default: 1e-4)
- `--image_size`: Ukuran input gambar (default: 512)

### Step 3: Inference (Prediksi)

#### Prediksi Single Image

```powershell
python inference.py --model_path checkpoints/best_model.pth --image_path dataset/data-1.jpg --output_dir outputs
```

#### Prediksi Batch (Multiple Images)

```powershell
python inference.py --model_path checkpoints/best_model.pth --image_dir dataset --output_dir outputs
```

### Step 4: Streamlit Web App

Jalankan aplikasi web interaktif:

```powershell
streamlit run app.py
```

Fitur web app:
- Upload gambar banjir
- Deteksi dan segmentasi otomatis
- Visualisasi hasil dengan overlay
- Statistik persentase area banjir
- Download hasil (mask dan overlay)
- Pengaturan threshold dan transparansi

## 📊 Model Architecture

Model menggunakan **U-Net**, arsitektur encoder-decoder yang populer untuk segmentasi:

```
Encoder (Downsampling):
- Conv Block 1: 3 → 64 channels
- Conv Block 2: 64 → 128 channels
- Conv Block 3: 128 → 256 channels
- Conv Block 4: 256 → 512 channels
- Bottleneck: 512 → 1024 channels

Decoder (Upsampling):
- Up Block 1: 1024 → 512 (+ skip connection)
- Up Block 2: 512 → 256 (+ skip connection)
- Up Block 3: 256 → 128 (+ skip connection)
- Up Block 4: 128 → 64 (+ skip connection)
- Output: 64 → 2 classes (flood/non-flood)
```

## 🎨 Data Augmentation

Augmentasi yang digunakan saat training:
- Horizontal Flip
- Vertical Flip
- Random Rotation (90°)
- Shift, Scale, Rotate
- Brightness & Contrast adjustment
- Gaussian Noise
- Blur

## 📈 Metrik Evaluasi

- **IoU (Intersection over Union)**: Mengukur overlap antara prediksi dan ground truth
- **Dice Coefficient**: Metrik similarity untuk segmentasi
- **Precision**: Akurasi deteksi positif
- **Recall**: Coverage deteksi
- **F1 Score**: Harmonic mean dari precision dan recall

## 💡 Tips

### Untuk Dataset Kecil:
- Gunakan data augmentation yang agresif
- Gunakan pretrained encoder (DeepLabV3)
- Lakukan transfer learning

### Untuk Meningkatkan Akurasi:
- Tambahkan lebih banyak data training
- Pastikan kualitas anotasi mask bagus
- Tune hyperparameter (learning rate, batch size)
- Gunakan ensemble model

### Untuk Inference Cepat:
- Gunakan GPU
- Reduce image size (256 atau 384)
- Gunakan model yang lebih ringan

## 🔧 Troubleshooting

### Error: CUDA out of memory
- Kurangi batch_size
- Kurangi image_size
- Gunakan CPU jika GPU tidak cukup

### Error: Model tidak ditemukan
- Pastikan sudah training model dulu
- Check path ke checkpoint

### Hasil segmentasi tidak akurat
- Tambah lebih banyak data training
- Improve kualitas anotasi
- Tune threshold di app

## 📚 Dependencies

- Python 3.8+
- PyTorch
- torchvision
- Streamlit
- OpenCV
- Albumentations
- NumPy
- Matplotlib
- Pillow

## 🤝 Kontribusi

Sistem ini dapat dikembangkan lebih lanjut dengan:
- Menambah model architecture (DeepLabV3+, PSPNet, etc.)
- Multi-class segmentation (tingkat keparahan banjir)
- Real-time video processing
- Deployment ke cloud
- Mobile app version

## 📄 License

Sistem ini dibuat untuk tujuan edukasi dan penelitian.

## 📞 Kontak

Untuk pertanyaan atau masalah, silakan buat issue di repository.

---

**Happy Coding! by devnolife🚀**
