# Flood Segmentation Dataset

Dataset untuk training model segmentasi banjir.

## Struktur Folder

```
dataset/
├── images/          # Gambar asli
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── masks/           # Mask/label segmentasi
│   ├── img001_mask.png
│   ├── img002_mask.png
│   └── ...
└── README.md
```

## Format Data

### Images
- Format: JPG, JPEG, PNG, WEBP
- Rekomendasi ukuran: 512x512 atau lebih besar
- RGB images

### Masks
- Format: PNG (grayscale)
- Binary mask:
  - Pixel value 255 (putih) = area banjir
  - Pixel value 0 (hitam) = tidak ada banjir
- Ukuran harus sama dengan gambar asli

## Cara Membuat Dataset

### Opsi 1: Gunakan Annotation Tool
```bash
streamlit run annotate.py
```

### Opsi 2: Manual dengan Image Editor
1. Buka gambar di software image editing (Photoshop, GIMP, Paint.NET)
2. Buat layer baru untuk mask
3. Gambar area banjir dengan warna putih (255, 255, 255)
4. Background hitam (0, 0, 0)
5. Save sebagai PNG grayscale

## Dataset Guidelines

### Good Practices:
- ✅ Gambar dengan pencahayaan yang baik
- ✅ Berbagai kondisi banjir (ringan, sedang, berat)
- ✅ Berbagai sudut pengambilan gambar
- ✅ Variasi lokasi dan lingkungan
- ✅ Mask yang akurat dan detail

### Avoid:
- ❌ Gambar blur atau low quality
- ❌ Mask yang tidak akurat
- ❌ Dataset yang sangat imbalanced

## Naming Convention

Gunakan naming yang konsisten:
- Image: `flood_001.jpg`, `flood_002.jpg`
- Mask: `flood_001_mask.png`, `flood_002_mask.png`

## Jumlah Data Minimal

- **Minimum**: 50-100 gambar untuk proof of concept
- **Recommended**: 500+ gambar untuk production model
- **Optimal**: 1000+ gambar dengan berbagai variasi

## Split Data

Rekomendasi split:
- Training: 70%
- Validation: 20%
- Test: 10%

Split dilakukan otomatis saat training.

## Data Augmentation

Model akan menggunakan augmentasi otomatis:
- Flip (horizontal, vertical)
- Rotation
- Brightness/Contrast adjustment
- Scaling
- Noise injection

## Contoh Dataset

Folder ini berisi 4 sample images untuk testing:
- `data-1.jpg`
- `data-2.jpeg`
- `data-3.jpg`
- `data-4.webp`

**Note**: Untuk training model yang baik, Anda perlu menambahkan lebih banyak data dan membuat mask untuk setiap gambar.

## Source Data

Anda dapat mengumpulkan data dari:
- Public flood image datasets
- Google Images (pastikan license)
- Drone footage
- CCTV footage
- Social media (dengan permission)
- Government disaster databases

## Quality Check

Sebelum training, pastikan:
1. Setiap image memiliki mask yang sesuai
2. Ukuran image dan mask sama
3. Mask dalam format binary (0 dan 255)
4. Tidak ada file corrupt
5. Naming konsisten

---

**Good luck dengan dataset Anda! 📊**
