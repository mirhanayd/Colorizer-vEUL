# 🎨 AI Image Colorizer / Yapay Zeka Görsel Renklendirici

[English](#english) | [Türkçe](#turkish)

---

<a name="english"></a>
## 🇬🇧 English

### Overview
AI-powered image colorization tool that transforms black & white images into colorful photos using deep learning. Built with Streamlit and OpenCV's DNN module.

### Features
- ✨ Automatic colorization of grayscale images
- 📤 Upload your own images or select from local folder
- 🖼️ Side-by-side comparison view
- 💾 Download colorized results
- 🚀 Easy-to-use web interface
- 🔬 Based on Zhang et al.'s colorization research

### Requirements
- **Python Version**: Python 3.7 - 3.11 (Recommended: Python 3.9 or 3.10)
- Git (for cloning the repository)
- Git LFS (for downloading large model files)

### Installation Steps

#### Step 1: Install Git LFS
```bash
# Windows (using Git for Windows - already included)
# Or download from: https://git-lfs.github.com/

# Verify installation
git lfs version
```

#### Step 2: Clone the Repository
```bash
# Clone the repository
git clone https://github.com/mirhanayd/Colorizer-vEUL.git

# Navigate to project directory
cd Colorizer-vEUL
```

#### Step 3: Download Large Files with Git LFS
```bash
# Pull LFS files (model files)
git lfs pull
```

#### Step 4: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Windows CMD:
venv\Scripts\activate.bat

# Linux/Mac:
source venv/bin/activate
```

#### Step 5: Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt
```

### Running the Application

```bash
# Run Streamlit app
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

### Usage

1. **Upload Method**:
   - Click on "Upload Your Own" tab
   - Choose an image file (JPG, PNG, BMP)
   - View the colorized result

2. **Select from Folder**:
   - Click on "Select from Images Folder" tab
   - Choose an image from the dropdown
   - Preview and colorize

3. **Download**:
   - Click "Download Colorized Image" button to save the result

### Project Structure
```
Colorizer-vEUL/
│
├── app.py                          # Main Streamlit application
├── main.py                         # Alternative script (if exists)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── models/                         # Pre-trained models (Git LFS)
│   ├── colorization_deploy_v2.prototxt
│   ├── colorization_release_v2.caffemodel  (129 MB)
│   └── pts_in_hull.npy
│
└── images/                         # Sample images folder (optional)
```

### Troubleshooting

**Issue: Model files not found**
```bash
# Make sure Git LFS is installed and pull the files
git lfs install
git lfs pull
```

**Issue: Python version incompatibility**
- Use Python 3.7 to 3.11
- Recommended: Python 3.9 or 3.10

**Issue: Module not found**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Technology Stack
- **Frontend**: Streamlit
- **Computer Vision**: OpenCV (DNN module)
- **Deep Learning Model**: Caffe
- **Language**: Python

### Credits
This project uses the colorization model from:
- Zhang, Richard, et al. "Colorful image colorization." ECCV 2016.

### License
This project is open source and available for educational purposes.

---

<a name="turkish"></a>
## 🇹🇷 Türkçe

### Genel Bakış
Derin öğrenme kullanarak siyah-beyaz görüntüleri renkli fotoğraflara dönüştüren yapay zeka destekli görüntü renklendirme aracı. Streamlit ve OpenCV'nin DNN modülü ile geliştirilmiştir.

### Özellikler
- ✨ Gri tonlamalı görüntülerin otomatik renklendirilmesi
- 📤 Kendi görsellerinizi yükleyin veya yerel klasörden seçin
- 🖼️ Yan yana karşılaştırma görünümü
- 💾 Renklendirilmiş sonuçları indirin
- 🚀 Kullanımı kolay web arayüzü
- 🔬 Zhang ve arkadaşlarının renklendirme araştırmasına dayalı

### Gereksinimler
- **Python Sürümü**: Python 3.7 - 3.11 (Önerilen: Python 3.9 veya 3.10)
- Git (depoyu klonlamak için)
- Git LFS (büyük model dosyalarını indirmek için)

### Kurulum Adımları

#### Adım 1: Git LFS'i Kurun
```bash
# Windows (Git for Windows ile birlikte gelir)
# Veya şuradan indirin: https://git-lfs.github.com/

# Kurulumu doğrulayın
git lfs version
```

#### Adım 2: Depoyu Klonlayın
```bash
# Depoyu klonlayın
git clone https://github.com/mirhanayd/Colorizer-vEUL.git

# Proje dizinine gidin
cd Colorizer-vEUL
```

#### Adım 3: Git LFS ile Büyük Dosyaları İndirin
```bash
# LFS dosyalarını çekin (model dosyaları)
git lfs pull
```

#### Adım 4: Sanal Ortam Oluşturun (Önerilen)
```bash
# Sanal ortam oluşturun
python -m venv venv

# Sanal ortamı etkinleştirin
# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Windows CMD:
venv\Scripts\activate.bat

# Linux/Mac:
source venv/bin/activate
```

#### Adım 5: Bağımlılıkları Yükleyin
```bash
# Gerekli paketleri yükleyin
pip install -r requirements.txt
```

### Uygulamayı Çalıştırma

```bash
# Streamlit uygulamasını çalıştırın
streamlit run app.py
```

Uygulama otomatik olarak varsayılan web tarayıcınızda `http://localhost:8501` adresinde açılacaktır.

### Kullanım

1. **Yükleme Yöntemi**:
   - "Upload Your Own" sekmesine tıklayın
   - Bir görüntü dosyası seçin (JPG, PNG, BMP)
   - Renklendirilmiş sonucu görüntüleyin

2. **Klasörden Seçim**:
   - "Select from Images Folder" sekmesine tıklayın
   - Açılır menüden bir görüntü seçin
   - Önizleme yapın ve renklendirin

3. **İndirme**:
   - Sonucu kaydetmek için "Download Colorized Image" düğmesine tıklayın

### Proje Yapısı
```
Colorizer-vEUL/
│
├── app.py                          # Ana Streamlit uygulaması
├── main.py                         # Alternatif betik (varsa)
├── requirements.txt                # Python bağımlılıkları
├── README.md                       # Bu dosya
│
├── models/                         # Önceden eğitilmiş modeller (Git LFS)
│   ├── colorization_deploy_v2.prototxt
│   ├── colorization_release_v2.caffemodel  (129 MB)
│   └── pts_in_hull.npy
│
└── images/                         # Örnek görüntü klasörü (isteğe bağlı)
```

### Sorun Giderme

**Sorun: Model dosyaları bulunamadı**
```bash
# Git LFS'in kurulu olduğundan emin olun ve dosyaları çekin
git lfs install
git lfs pull
```

**Sorun: Python sürüm uyumsuzluğu**
- Python 3.7 - 3.11 arası kullanın
- Önerilen: Python 3.9 veya 3.10

**Sorun: Modül bulunamadı**
```bash
# Bağımlılıkları yeniden yükleyin
pip install --upgrade -r requirements.txt
```

### Teknoloji Yığını
- **Ön Yüz**: Streamlit
- **Bilgisayarlı Görü**: OpenCV (DNN modülü)
- **Derin Öğrenme Modeli**: Caffe
- **Dil**: Python

### Teşekkürler
Bu proje aşağıdaki renklendirme modelini kullanmaktadır:
- Zhang, Richard, et al. "Colorful image colorization." ECCV 2016.

### Lisans
Bu proje açık kaynaklıdır ve eğitim amaçlı kullanıma açıktır.

---

## 📧 Contact / İletişim

For questions or suggestions / Sorular veya öneriler için:
- GitHub: [@mirhanayd](https://github.com/mirhanayd)
- Repository: [Colorizer-vEUL](https://github.com/mirhanayd/Colorizer-vEUL)

---

**Made with ❤️ using Streamlit and OpenCV**
