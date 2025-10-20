# Recycling Sentry

**An AI-Powered Image-Classification Recycling Device**

*Reducing recycling contamination and human error by using AI to categorize waste items at the point of disposal.*

---

## 🏆 Awards & Recognition

- 🥇 **California Life Sciences - Amgen State Hackathon** - 4th Place (Best Presentation)
- 🎯 **Conrad - Equinor Innovation Challenge** - Semi-Finalist
- 🌊 **Blue Ocean International Pitch Competition** - Top 2%

---

## 📊 The Problem

Recycling contamination occurs when non-recyclable materials enter the recycling stream. This is primarily caused by citizens being unaware of proper disposal methods, which happens nearly **50% of the time**.

### Impact:
- **75%** of waste in landfills is actually recyclable
- Nearly **1,000 recycling centers** in California alone have closed in the last 2 years
- A nationwide waste management company recently closed **25% of their facilities**
- Many facilities send even slightly contaminated batches directly to landfills

---

## 💡 Our Solution

The **Recycling Sentry** is an AI-powered device that uses computer vision to identify waste items and determine their proper disposal method in real-time. By replacing the flawed human decision-making process with AI, we can virtually eliminate recycling contamination.

### Key Features:
- ✅ **94% accuracy rate** (vs. 50% average citizen accuracy)
- ✅ Real-time object detection and classification
- ✅ Identifies 6 common waste categories
- ✅ Affordable and accessible design
- ✅ Under 10 second operation time
- ✅ Raspberry Pi-based compact design

---

## 🎯 Model Performance

Our detection-based AI model achieves **93.7% overall accuracy** across 6 waste categories:

| Item | Precision Rate |
|------|----------------|
| Plastic | 98.4% |
| Paper | 94.5% |
| Cardboard | 94.0% |
| Glass Bottle | 92.4% |
| Trash | 92.0% |
| Metal Can | 91.1% |

**Comparison with existing solutions:**
- Recycling Sentry: 94%
- TrashBot: 94%
- AMP Cortex: 99%

*Our advantage: Same quality at a fraction of the cost, making it accessible for households and small businesses.*

---

## 🛠️ Technology Stack


### Software & Tools
- **Deep Learning Framework:** PyTorch 2.0+
- **Model Architecture:** EfficientNet-B0 (Transfer Learning)
- **Training Platform:** Google Colab
- **Dataset Management:** Roboflow
- **Monitoring:** Weights & Biases (WandB)
- **Deployment:** FastAPI + React (coming soon)

### Model Details
- **Type:** Object Detection (YOLOv8-based)
- **Training Dataset:** 12,000+ annotated data points
- **Training Time:** 14 revisions, 5-6 hours each
- **Augmentations:** Rotation, flip, brightness/contrast, crop, shear

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.11+
PyTorch 2.0+
torchvision
Roboflow account (for dataset management)
```

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/recycling-sentry.git
cd recycling-sentry
```

2. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Training the Model

1. **Download dataset from Roboflow:**
   - Use the 4000+ image dataset (no pre-augmentations)
   - Export format: YOLOv8 or Folder Structure
   - Split: 70% train / 20% validation / 10% test

2. **Organize your data:**
```bash
python3 reorganize_data.py
```

3. **Train the model:**
```bash
python3 train_model.py
```

Training outputs:
- `waste_classifier.pth` - Trained model weights
- `class_names.json` - Class label mappings
- `training_history.png` - Loss/accuracy curves
- `confusion_matrix.png` - Model evaluation


---

## 🔮 Future Work

- [ ] Deploy full-stack web application
- [ ] Expand to 10+ item categories
- [ ] Mobile app for on-the-go scanning
- [ ] Multi-language support for instructions

---

## 🌟 Impact

By implementing the Recycling Sentry at businesses, households, and recycling centers, we can:

- **Nearly double** recycling accuracy (50% → 94%)
- **Reduce contamination** by 45%
- **Save thousands** of recycling centers from closure
- **Divert 75%** of current landfill waste back to recycling
- **Minimize environmental impact** through proper waste sorting

---

## 📧 Contact

For questions or collaboration opportunities, please reach out to me at sanjith.shan24@gmail.com!

---

**Made with 💚 for a cleaner planet**
