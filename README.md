# Detection of Pneumonia and COVID-19 from Chest X-Ray Images

**Built in 2021. Published in 2022. Before the AI boom, before ChatGPT, before everyone had a computer vision tutorial in their feed.**

This was one of my first serious ML projects....built during COVID, debugged with Stack Overflow, Kaggle forums and a lot of trial and error. No LLMs to explain error messages. No Copilot to autocomplete the architecture. Just documentation, forum threads and persistence.

It got published. But more importantly, it taught me how models actually fail... which turned out to be more useful than the final accuracy numbers.

**[Published paper on ResearchGate](https://www.researchgate.net/publication/364821917_Detection_of_Pneumonia_and_COVID-19_from_Chest_X-Ray_Images_Using_Neural_Networks_and_Deep_Learning)**  
Authors: Jeet Santosh Nimbhorkar, Kurapati Sreenivas Aravind, K Jeevesh, Suja Palaniswamy

---

## Context: Why This Exists

In 2021, COVID-19 diagnosis was still heavily reliant on manual chest X-ray review. The idea that a CNN could flag pneumonia or COVID from a radiograph — and potentially help in under-resourced settings felt like a problem worth working on.

The tooling available then was a fraction of what exists today. Kaggle's free GPU tier (P100, 16GB) was genuinely exciting to have access to. HuggingFace didn't have the ecosystem it does now. Transfer learning from ImageNet was still considered clever, not obvious.

This project would be considered straightforward today. In 2021, getting InceptionV3 to fine-tune correctly on medical images without it immediately overfitting took some level of debugging.

---

## The Notebooks - Including the Ones That Failed

This repo includes 3-4 Kaggle notebooks, not just the final working version. **The failed ones are in there on purpose.**

Early runs included:

- **VGG19 with no frozen layers** — the model immediately overfit. Training accuracy hit 99%, validation stayed at 60%. Classic mistake, learned from it.
- **ResNet50 without learning rate adjustment** — vanishing gradients made it essentially untrainable. Dropping the LR and adding a custom head fixed it, but it took several broken runs to get there.
- **Wrong normalization on the custom CNN** — outputs were saturating, loss wasn't moving. Took an afternoon and a Stack Overflow thread to diagnose.
- **Data leakage in an early split** — caught it when validation numbers looked suspiciously good.

The failed runs aren't embarrassing — they're the actual record of how the working version got built. Anyone who tells you their first run worked is leaving something out.

---

## What This Project Actually Is

A benchmarking study across multiple CNN architectures for multi-class chest X-ray classification, with a custom lightweight CNN proposed as a low-parameter alternative to heavy pretrained models.

Three datasets, four pretrained architectures, one custom model, three classification tasks of increasing complexity (2-class → 3-class → 4-class).

---

## Datasets

| Dataset | Classes | Source |
|---|---|---|
| Pneumonia Chest X-ray | 2 (Pneumonia, Normal) | [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) |
| COVID-19 + Pneumonia | 3 (COVID, Pneumonia, Normal) | [Kaggle](https://www.kaggle.com/prashant268/chest-xray-covid19-pneumonia) |
| COVID-19 Radiography | 4 (COVID, Pneumonia, Lung Opacity, Normal) | [Kaggle](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database) |

---

## Architectures

### Pretrained (Transfer Learning)
AlexNet, InceptionV3, ResNet50, VGG19 - each fine-tuned with additional dense layers (1024 → 512 neurons) on top of frozen ImageNet weights.

### Custom CNN
4 convolutional layers (32 → 64 → 64 → 128 filters), MaxPooling after each, Flatten → Dense(64) → output. Input: 150×150×3. The goal was to see how far a lightweight model could go against the heavyweights, given Kaggle's compute constraints at the time.

---

## Results

| Dataset | Best Model | Accuracy |
|---|---|---|
| Pneumonia | VGG19 + Adam | 93.58% |
| COVID + Pneumonia | InceptionV3 (2048) | **94.25%** |
| COVID Radiography | InceptionV3 (original) | 84.32% |

### What Actually Moved the Needle
- Switching from SGD to Adam improved VGG19 accuracy by ~34.5% — optimizer choice matters more than it looks on paper
- Adding dense layers to ResNet and InceptionV3 improved generalization significantly
- ResNet50 consistently underperformed its reputation at this scale — skip connections don't save you when your dataset is small and your LR is misconfigured

---

## Parameter Comparison

| Model | Total Params | Trainable Params |
|---|---|---|
| AlexNet | 58M | 58M |
| InceptionV3 | 21.8M | ~2K (frozen) |
| ResNet50 | 23.5M | ~2K (frozen) |
| VGG19 | 139M | 119M |
| Custom CNN | << all of the above | all trainable |

---

## Training Setup

- **Platform**: Kaggle (P100 GPU, 16GB VRAM)
- **Epochs**: 20
- **Optimizers**: Adam and SGD (compared)
- **Loss**: Binary CrossEntropy (2-class), Categorical CrossEntropy (3/4-class)
- **Debugging tools**: Stack Overflow, Kaggle discussion threads, Keras docs, and a lot of patience

---

## A Note on Where This Sits in 2026

Yes, you can get better results today with a pretrained ViT and 10 lines of HuggingFace code. That's not the point.

That process is what led to everything that came after: the histopathology work at NYU Langone, the MIL pipelines, the SLURM cluster debugging. It started here.

---

## Citation

> Nimbhorkar, J.S., Kurapati, S.A., Jeevesh, K., & Palaniswamy, S. (2022).  
> **Detection of Pneumonia and COVID-19 from Chest X-Ray Images Using Neural Networks and Deep Learning.**  
> *International Journal of Computer Science and Applications, 2022.*  
> [Available on ResearchGate](https://www.researchgate.net/publication/364821917_Detection_of_Pneumonia_and_COVID-19_from_Chest_X-Ray_Images_Using_Neural_Networks_and_Deep_Learning)
