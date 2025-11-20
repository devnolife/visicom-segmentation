"""
Model U-Net untuk Segmentasi Banjir
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling dengan maxpool kemudian double conv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling kemudian double conv"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Padding jika ukuran tidak sama
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution"""
    
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net Architecture untuk Segmentasi
    
    Args:
        n_channels: Jumlah channel input (3 untuk RGB)
        n_classes: Jumlah kelas output (2 untuk binary: banjir/tidak banjir)
        bilinear: Gunakan bilinear upsampling atau transposed conv
    """
    
    def __init__(self, n_channels=3, n_classes=2, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
    
    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder path dengan skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        return logits


class DiceLoss(nn.Module):
    """Dice Loss untuk segmentasi"""
    
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        pred = pred[:, 1, :, :]  # Ambil channel untuk kelas positif
        target = target.float()
        
        intersection = (pred * target).sum(dim=(1, 2))
        union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """Kombinasi Cross Entropy Loss dan Dice Loss"""
    
    def __init__(self, weight_ce=0.5, weight_dice=0.5):
        super(CombinedLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.weight_ce * ce + self.weight_dice * dice


def get_model(model_name='unet', n_channels=3, n_classes=2, pretrained=False):
    """
    Mendapatkan model segmentasi
    
    Args:
        model_name: Nama model ('unet', 'deeplabv3')
        n_channels: Jumlah channel input
        n_classes: Jumlah kelas
        pretrained: Load pretrained weights
    
    Returns:
        Model
    """
    if model_name.lower() == 'unet':
        model = UNet(n_channels=n_channels, n_classes=n_classes)
    elif model_name.lower() == 'deeplabv3':
        # Alternatif: menggunakan DeepLabV3 dari segmentation_models_pytorch
        try:
            import segmentation_models_pytorch as smp
            model = smp.DeepLabV3Plus(
                encoder_name="resnet34",
                encoder_weights="imagenet" if pretrained else None,
                in_channels=n_channels,
                classes=n_classes,
            )
        except ImportError:
            print("segmentation_models_pytorch tidak tersedia, gunakan UNet")
            model = UNet(n_channels=n_channels, n_classes=n_classes)
    else:
        raise ValueError(f"Model {model_name} tidak dikenali")
    
    return model


if __name__ == "__main__":
    # Test model
    model = UNet(n_channels=3, n_classes=2)
    x = torch.randn(1, 3, 512, 512)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Hitung parameter
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
