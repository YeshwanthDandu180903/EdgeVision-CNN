# EdgeVision CNN: Mobile Plant Disease Detection 🌱🔬

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-Supported-yellow.svg)](https://colab.research.google.com/)
[![Status](https://img.shields.io/badge/Status-In%20Development-red.svg)]()

> **⚠️ Project Status: Currently in Development**  
> This project is actively being developed as a collaborative learning initiative for mobile plant disease detection.

## 🎯 Project Overview

EdgeVision CNN is a **lightweight convolutional neural network** designed specifically for **real-time plant disease detection** on mobile devices. The project focuses on creating an efficient architecture that balances accuracy with mobile deployment constraints such as model size, inference speed, and battery consumption.

### 🌟 Key Features

- 🏗️ **Custom CNN Architecture** with progressive filter reduction.
- 📱 **Mobile-Optimized** targeting <5MB model size for mobile deployment
- ⚡ **Real-time Inference** optimized for response time and battery efficiency
- 🔬 **Research-Grade Evaluation** comparing against state-of-the-art models
- 🌐 **Google Colab Integration** for cloud-based GPU training



**Total Parameters:** ~2.8M (4-8x smaller than traditional models)

## 📊 Datasets

The project evaluates performance across multiple plant disease datasets:

| Dataset | Domain | Classes | Images | Status |
|---------|---------|---------|---------|---------|
| **Fruit & Vegetable Diseases** | Mixed crops | ~25 | ~12,000 | 📤 Pending |
| **Crop Pest and Disease** | Agricultural pests | ~18 | ~8,000 | 📤 Pending |
| **CropDisease** | General crops | ~15 | ~6,000 | 📤 Pending |
| **GuavaDiseaseDataset** | Guava-specific | ~8 | ~3,000 | 📤 Pending |

## 🔬 Baseline Comparisons

EdgeVision CNN is benchmarked against established architectures:

- **VGG16** (Transfer Learning)
- **ResNet50** (Transfer Learning)
- **MobileNetV2** (Mobile-optimized)
- **EfficientNetB0** (Efficiency-focused)

## 🛠️ Installation & Setup

### Prerequisites

```bash
Python 3.8+
TensorFlow 2.x
NumPy
Pandas
Scikit-learn
Matplotlib
```

### Quick Start with Google Colab

1. **Open in Colab:**
   ```
   https://colab.research.google.com/github/your-username/EdgeVision-CNN
   ```

2. **Mount Google Drive:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Run the framework:**
   ```python
   # Clone and run
   !git clone https://github.com/your-username/EdgeVision-CNN.git
   %cd EdgeVision-CNN
   !python our_plan.py
   ```

### Local Installation

```bash
# Clone the repository
git clone https://github.com/YeshwanthDandu180903/EdgeVision-CNN.git
cd EdgeVision-CNN

# Install dependencies
pip install -r requirements.txt

# Run the training framework
python our_plan.py
```

## 📁 Project Structure

```
EdgeVision-CNN/
├── our_plan.py                 # Main training framework
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── datasets/                   # Dataset storage (to be added)
│   ├── Fruit_And_Vegetable_Diseases_Dataset/
│   ├── Crop_pest_and_disease/
│   ├── CropDisease/
│   └── GuavaDiseaseDataset/
├── models/                     # Saved models (generated)
├── logs/                       # Training logs (generated)
└── results/                    # Research results (generated)
```

## 🎯 Research Goals

### Primary Objectives
- [ ] Develop lightweight CNN architecture for mobile deployment
- [ ] Achieve competitive accuracy with significantly reduced parameters
- [ ] Optimize for real-time inference on mobile devices
- [ ] Comprehensive evaluation against state-of-the-art models

### Performance Targets
- **Model Size:** <5MB
- **Inference Time:** <100ms on mobile CPU
- **Accuracy:** Competitive with larger models
- **Battery Impact:** Minimal power consumption

## 📈 Current Progress

### ✅ Completed
- [x] Core CNN architecture design
- [x] Google Colab integration
- [x] Data preprocessing pipeline
- [x] Baseline model implementations
- [x] Training framework with adaptive early stopping

### 🚧 In Progress
- [ ] Dataset collection and preprocessing
- [ ] Comprehensive model evaluation
- [ ] Mobile optimization (quantization, pruning)
- [ ] Performance benchmarking
- [ ] Research paper preparation

### 📅 Upcoming
- [ ] Mobile app integration (PlantPal)
- [ ] Real-world testing
- [ ] Model deployment optimization
- [ ] Documentation completion

## 🤝 Team & Collaboration

This project is being developed as a **collaborative learning initiative** by a team of developers exploring deep learning applications in agriculture.

**Team Members:** [Add team member names]

**Contributions Welcome:** We're actively seeking collaborators interested in:
- Plant disease detection
- Mobile AI applications
- Agricultural technology
- Deep learning research

## 📊 Preliminary Results

> **Note:** Results will be updated as training completes

| Model | Parameters | Size (MB) | Accuracy | Inference (ms) |
|-------|------------|-----------|----------|----------------|
| EdgeVision CNN | ~2.8M | ~11MB | TBD | TBD |
| VGG16 | ~15M | ~60MB | TBD | TBD |
| ResNet50 | ~25M | ~100MB | TBD | TBD |
| MobileNetV2 | ~3.5M | ~14MB | TBD | TBD |
| EfficientNetB0 | ~5M | ~20MB | TBD | TBD |

## 🔧 Usage

### Training Custom Model

```python
from our_plan import ColabResearchFramework

# Initialize framework
framework = ColabResearchFramework()

# Run complete study
results = framework.run_complete_colab_study()

# View results
print("Training completed! Check results in /content/complete_research_results.csv")
```

### Model Architecture

```python
# Create EdgeVision CNN
model = framework.create_edgevision_cnn(num_classes=10)

# View architecture
model.summary()
```

## 📝 Research Applications

This project aims to contribute to:
- **Agricultural AI:** Efficient disease detection for farmers
- **Mobile Computing:** Lightweight deep learning models
- **Edge AI:** On-device inference capabilities
- **Academic Research:** Comparative analysis of CNN architectures

## 🐛 Known Issues

- [ ] Dataset upload pending
- [ ] Training time optimization needed
- [ ] Mobile quantization not implemented
- [ ] Documentation incomplete

## 📚 Documentation

- [ ] API Documentation (Coming soon)
- [ ] Training Guide (Coming soon)
- [ ] Mobile Deployment Guide (Coming soon)
- [ ] Research Paper (In preparation)

## 🔗 Related Projects

- **PlantPal Mobile App** (Integration target)
- **Agricultural AI Research** (Academic context)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google Colab for providing free GPU resources
- TensorFlow team for the deep learning framework
- Open-source plant disease datasets
- Agricultural AI research community

## 📞 Contact

For questions, suggestions, or collaboration opportunities:

- **Email:** [yeshwanthdandu2003@gmail.com]
- **GitHub Issues:** [Create an issue](https://github.com/your-username/EdgeVision-CNN/issues)
- **Discussions:** [Join the discussion](https://github.com/your-username/EdgeVision-CNN/discussions)

---

**⭐ Star this repository if you find it interesting!**

**🔄 Watch for updates as we continue development!**

---

*EdgeVision CNN - Bringing AI-powered plant disease detection to mobile devices* 🌱📱🤖
