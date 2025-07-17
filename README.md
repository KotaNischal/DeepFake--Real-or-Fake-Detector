# DeepFake Image Detection using ResNet18

This project implements a deep learning model based on **ResNet18** to detect deepfake images. It fine-tunes a pretrained ResNet18 architecture on a dataset of real and GAN-generated faces for binary classification.

---

## 📌 Overview

- **Model**: ResNet18 (pretrained on ImageNet)
- **Dataset**: Real and Fake Face Detection Dataset ([Kaggle](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection))
- **Framework**: PyTorch
- **Accuracy Achieved**: ~67%

---

## 📁 Project Structure

deepfake-resnet18/
├── model/
│ └── resnet_model.py # ResNet18 model definition
├── scripts/
│ ├── train_resnet.py # Training script
│ └── infer_image.py # Inference script (can be shared with MobileNet)
├── data/
│ └── (Real and Fake images)
├── resnet18_deepfake.pth # Trained model weights
├── requirements.txt
└── README.md

---

## 🧠 Model Details

- Uses `torchvision.models.resnet18(pretrained=True)`
- Last fully connected layer is replaced for binary classification (Real vs Fake)
- Uses transfer learning to improve performance on a relatively small dataset

---

## 🔧 Training Configuration

- Optimizer: `Adam` with learning rate = 0.0001
- Loss Function: `CrossEntropyLoss`
- Batch Size: 32
- Epochs: 10
- Image Size: 224x224
- Mixed precision training (optional with AMP)

---

## 🖼️ Inference

To detect whether an input image is real or fake:

```bash
python scripts/infer_image.py --image path_to_image.jpg
