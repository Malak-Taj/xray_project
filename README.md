# Project Overview
* CNN X_ray Lung Classification: Normal, Pneumonia and Tuberculosis(TB)
* Build 5 custom CNN architectures using TensorFlow / Keras
* Treating Imbalance (Class_weight, Focal Loss)
* Feature Extraction :
  - Load U-Net Segmentation Model
  - (https://www.kaggle.com/datasets/farhanhaikhan/unet-lung-segmentation-weights-for-chest-x-rays/data)
  - Masked Generation Using U-Net Pretrained Model
  - Masked Data Generation Using U-Net Pretrained Model
  - Cleaning Masked Data 
* Explainable AI(XAI):
  - Grad_CAM: Gradient-weighted Class Activation Mapping
  - Grad_CAM is used for pixel Analysis
* Convolutional Block Attention Module:
  - "CBAM: Convolutional Block Attention Module" (Woo et al., ECCV 2018)
  - Formula: Feature → Channel Attention → Spatial Attention → Refined Feature
  - CBAM block: Conv → BN → ReLU → CBAM → next layer
  - (https://joonyoung-cv.github.io/assets/paper/18_eccv_cbam.pdf)
* Callbacks and Techniques:
  - EarlyStopping, ReduceRLOnPlateau and Augmentation
* Transfer Learning Using EfficientNetB0 Pretrained Model
* Fine Tuning For EfficientNetB0 Model
* Evaluate The Best Model On the Test Data
# Main Challenge
* Challenge of the project was to increase the TB recall because High TB recall is  important in medical applications because missing TB cases can be dangerous
# Main Resource
* https://ieeexplore.ieee.org/abstract/document/9224622#algorithms
