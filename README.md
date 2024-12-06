# Adversarial Attacks on Pretrained Models Using FGSM

## Overview
This project examines the susceptibility of deep learning models to adversarial attacks and explores strategies for improving their robustness, inspired by the foundational work "Explaining and Harnessing Adversarial Examples" by Ian J. Goodfellow, Jonathon Shlens, and Christian Szegedy. Using the Fast Gradient Sign Method (FGSM), we generate adversarial examples and analyze their impact on pretrained models, including GoogleNet, VGG11, and ConvNeXt. The experiments are performed on the `Imagenette` dataset, a smaller, computationally efficient subset of ImageNet.

## Objectives
1. Evaluate the baseline performance of pretrained models on clean images.
2. Generate adversarial examples using FGSM and measure their effect on model accuracy.
3. Retrain models with adversarial examples to enhance robustness.
4. Investigate additional questions:
   - 4.1 Does a retrained model generalize to unseen classes?
   - 4.2 Are adversarial examples transferable across models?
   - 4.3 Can alternative attack methods be applied to ConvNeXt?

## Key Findings
- **Baseline Accuracy**:  
  Pretrained models (e.g., GoogleNet, VGG11, ConvNeXt) perform well on clean datasets, achieving high accuracy on the Imagenette validation set. This serves as a reference point for evaluating the impact of adversarial attacks.
- **Impact of FGSM**:  
  Adversarial examples significantly degrade model accuracy, with higher epsilon values causing greater disruption. However, modern architectures like ConvNeXt demonstrate notable robustness against FGSM attacks, reflecting advancements in model design and training strategies. This robustness likely stems from enhanced regularization techniques and architectural improvements.
- **Robustness Training**:  
  Retraining models with adversarial examples improves their resilience to attacks, though with trade-offs. Key observations include:
- **Generalization to Unseen Classes**:  
    Retrained models do not generalize effectively to unseen classes. This aligns with expectations, as fine-tuning on a specific dataset often leads to a loss of capabilities on data not included in retraining. This highlights a trade-off between robustness and generalization.
- **Transferability of Adversarial Examples**:  
    Adversarial examples generated for one model (e.g., VGG11) were transferable and effective against another (e.g., GoogleNet). This behavior supports findings in Goodfellow et al.'s literature, where adversarial perturbations often align with model features in high-dimensional spaces, enabling cross-model transferability.
- **Alternative Attacks on ConvNeXt**:  
    None of the one-shot adversarial attacks tested (e.g., FGSM) were effective against ConvNeXt. This suggests that ConvNeXt’s robustness may require iterative or multi-step attack methods for evaluation. Due to computational limitations, such iterative attacks were not explored in this study but remain a promising direction for future research.


## Datasets
### 1. Imagenette
- **Description**: A subset of ImageNet with 10 classes, designed for faster experimentation.
- **Use in This Project**:  
  - Primary dataset for evaluating pretrained models.
  - Used extensively in baseline evaluation, adversarial example generation, and robustness testing.
- **Source**: [Imagenette GitHub Repository](https://github.com/fastai/imagenette)

### 2. ImageNet Mini (1000 Classes)
- **Description**: A mini version of the full ImageNet dataset containing 1000 samples from 1000 classes, providing a diverse but computationally feasible subset.
- **Use in This Project**:  
  - Used in **Step 4.1** of Additional Investigations to evaluate whether the retrained VGG11 model generalizes to unseen classes.
- **Source**: [ImageNet Mini Kaggle Dataset](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000)


## Methods
- **Fast Gradient Sign Method (FGSM)**:
  FGSM perturbs input images by adding gradients scaled by a small factor ($\epsilon$) to maximize model loss. This is expressed mathematically as:

  $` \text{Perturbed Image} = \text{Original Image} + \epsilon \cdot \text{sign}(\nabla_x J(\theta, x, y)) `$

- **Models Used**:
The following pretrained models were obtained from PyTorch’s model repository:
  - GoogleNet
  - VGG11
  - Wide Resnet
  - ConvNeXt

## Results
- FGSM with with very minimal perturbation that are not noticeable to the human eye, $\epsilon = 0.007$, caused an 84% misclassification rate on GoogleNet and 93% on VGG11.
- Models more robust to FGSM, are still fooled with sufficient perturbations.
- Retraining VGG11 with adversarial examples increased its robustness but introduced significant accuracy trade-offs on clean data.

## Repository Contents
- `Adversarial_Attacks_Final.ipynb`: The main notebook containing all experiments.
- `README.md`: This file.
- `data/`: Placeholder or link for the Imagenette dataset (not included due to size constraints).
- `Presentation.pdf`: The presentation slides for this project.
