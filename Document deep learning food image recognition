# Deep Learning for Food Image Recognition and Nutrition Estimation

## Project Overview

This project applies deep learning and computer vision techniques to recognize food items from images and estimate their nutritional information. Using transfer learning, a pretrained convolutional neural network is fine-tuned to classify food images and link predictions to a nutrition reference table, enabling calorie and macronutrient lookup.

The goal of the project is to demonstrate how deep learning models can be used in real-world scenarios such as dietary tracking, health applications, and food-related analytics.

---

## Problem Statement

Given an image of a food item, the system should:

1. Correctly classify the food category
2. Return estimated nutritional values (e.g., calories, protein, carbohydrates, fats)

This problem combines **image classification** with **data integration**, simulating how computer vision models can be embedded into practical decision-support tools.

---

## Dataset

* **Source:** Public food image dataset (accessed via Kaggle)
* **Classes:** 9 food categories (e.g., ramen, cheesecake, caesar salad, spaghetti bolognese)
* **Images per class:** ~1,000
* **Split:** 80% training / 10% validation / 10% test

Due to size and licensing considerations, the raw image dataset is not included in this repository.

---

## Methodology

### Data Preparation

* Automated train/validation/test split
* Robust image loading with corrupted-file handling
* Image resizing to 224 × 224
* Data augmentation using Albumentations (random crops, flips, color jitter, dropout)

### Model Architecture

* **Backbone:** ResNet-18 (pretrained on ImageNet)
* **Approach:** Transfer learning

  * Initial training with frozen backbone (classifier head only)
  * Full fine-tuning of all layers

### Training Setup

* Loss function: Cross-entropy loss
* Optimizer: Adam / AdamW
* Learning rate scheduling: Cosine annealing
* Mixed precision training (when GPU available)

---

## Evaluation & Results

### Model Performance

* **Validation Accuracy:** ~93%
* **Test Accuracy:** ~94.5%

### Metrics

* Precision, recall, and F1-score computed for each class
* Confusion matrix used to analyze class-level performance

The model performs consistently across most food categories, with minor confusion between visually similar classes.

---

## Inference & Nutrition Lookup

After classification, the predicted food label is matched to a nutrition reference table to retrieve:

* Calories
* Protein
* Carbohydrates
* Fats
* Fiber
* Sugars
* Sodium

This demonstrates how model predictions can be integrated with structured data to produce actionable outputs.

---

## Tools & Technologies

* Python
* PyTorch & Torchvision
* NumPy, Pandas
* Scikit-learn
* Matplotlib
* KaggleHub

---

## Repository Structure

```
notebooks/
  food_image_recognition.ipynb

results/
  confusion_matrix.png
  training_curves.png
  sample_prediction.png

report/
  final_report.pdf
```

---

## Notes on Scalability

This project was developed and validated in a notebook environment. In a production setting, the training and inference pipelines could be deployed using:

* GPU-backed cloud infrastructure
* Model serving frameworks (e.g., TorchServe, FastAPI)
* Larger and more diverse food datasets for improved generalization

---

## Academic Context

This project was completed as part of a graduate-level analytics course and reflects both technical implementation and analytical reasoning.

---

## Key Takeaway

This project demonstrates how deep learning models can move beyond standalone classification tasks to support real-world applications by integrating predictions with external data sources, enabling practical insights such as nutrition estimation from food images.
