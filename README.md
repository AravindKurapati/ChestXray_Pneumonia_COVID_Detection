# ChestXray_Pneumonia_COVID_Detection

#  Detection of Pneumonia and COVID-19 from Chest X-Ray Images using Neural Networks and Deep Learning

This project implements a deep learning-based solution for detecting **Pneumonia and COVID-19** from chest X-ray images. It builds upon both a **custom CNN architecture** and **transfer learning** using pre-trained models such as AlexNet, InceptionV3, ResNet50, and VGG19. The results and methodology of this project were published in a peer-reviewed paper:

 **[Published paper on ResearchGate](https://www.researchgate.net/publication/364821917_Detection_of_Pneumonia_and_COVID-19_from_Chest_X-Ray_Images_Using_Neural_Networks_and_Deep_Learning)**  
 Authors: Jeet Santosh Nimbhorkar, Kurapati Sreenivas Aravind, K Jeevesh, Suja Palaniswamy

---

##  Objectives

- Early detection of pneumonia and COVID-19 using chest radiographs
- Compare the performance of different pre-trained CNN architectures
- Propose a custom lightweight CNN model with reduced parameters
- Evaluate precision, recall, and accuracy across multiple public datasets

---

##  Datasets Used

The models were trained and validated on three publicly available chest X-ray datasets:

1. **Pneumonia Chest X-ray Dataset**  
    [Kaggle Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)  
   ➤ 2-class: Pneumonia vs. Normal

2. **Chest X-ray – COVID-19 and Pneumonia**  
    [Kaggle Dataset](https://www.kaggle.com/prashant268/chest-xray-covid19-pneumonia)  
   ➤ 3-class: COVID-19, Pneumonia, Normal

3. **COVID-19 Radiography Database**  
    [Kaggle Dataset](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)  
   ➤ 4-class: COVID-19, Pneumonia, Lung Opacity, Normal

---

##  Model Architectures

###  Pre-trained Models via Transfer Learning
- **AlexNet**  
- **InceptionV3**  
- **ResNet50**  
- **VGG19**
   Each model was fine-tuned with additional dense layers (e.g., 1024, 512 neurons) for improved classification.

###  Proposed Custom CNN
- 4 Convolutional layers (filters: 32, 64, 64, 128)
- MaxPooling after each conv layer
- Flatten → Dense(64) → Output
- Input size: 150×150×3
- Activation: ReLU, Output: Sigmoid/Softmax

---

##  Training Setup

| Parameter        | Value                         |
|------------------|-------------------------------|
| Epochs           | 20                            |
| Optimizers       | Adam / SGD (compared)         |
| Loss Functions   | Binary / Categorical CrossEntropy |
| Platforms        | Kaggle GPU (P100, 16GB)       |

---

##  Evaluation Metrics

| Metric     | Description                       |
|------------|-----------------------------------|
| Accuracy   | Overall correct predictions       |
| Precision  | Correct positive predictions      |
| Recall     | Detection sensitivity             |

---

##  Results Summary

| Dataset      | Best Model            | Accuracy (%) |
|--------------|-----------------------|--------------|
| Pneumonia    | VGG19 + Adam          | **93.58**    |
| COVID+Pneu   | InceptionV3 (2048)    | **94.25**    |
| COVID Radiog.| InceptionV3 (original)| **84.32**    |

Our custom CNN achieved **competitive performance** with significantly **fewer trainable parameters** than pre-trained models.

---

## Key Observations

- Adam optimizer improved accuracy across all models (up to **34.5%** for VGG19)
- Adding **dense layers** to ResNet and InceptionV3 improved generalization
- **ResNet50 underperformed**, likely due to vanishing gradient issues despite skip connections
- The **custom CNN model** provided a lightweight alternative with high accuracy and low complexity

---

## Parameter Comparison

| Model      | Total Params | Trainable Params | Reduction (%) |
|------------|--------------|------------------|----------------|
| AlexNet    | 58M          | 58M              | ~0%            |
| InceptionV3| 21.8M        | 2K               | **99.99%**     |
| ResNet50   | 23.5M        | 2K               | **99.99%**     |
| VGG19      | 139M         | 119M             | 15%            |

---

## Citation

If you use this codebase or build upon the ideas, please cite the original publication:

> Nimbhorkar, J.S., Kurapati, S.A., Jeevesh, K., & Palaniswamy, S. (2022).  
> **Detection of Pneumonia and COVID-19 from Chest X-Ray Images Using Neural Networks and Deep Learning.**  
> *International Journal of Computer Science and Applications, 2022.*  
> [Available on ResearchGate](https://www.researchgate.net/publication/364821917_Detection_of_Pneumonia_and_COVID-19_from_Chest_X-Ray_Images_Using_Neural_Networks_and_Deep_Learning)

---

