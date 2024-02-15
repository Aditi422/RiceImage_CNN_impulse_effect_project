Dataset: https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset

# Multiclass Image Classification with Noise Reduction using Deep Learning

## Project Overview

### Introduction
This project focuses on utilizing Convolutional Neural Networks (CNNs) for multiclass image classification, specifically in the context of identifying different rice varieties. The primary innovation lies in evaluating the model's resilience to impulse noise, a real-world challenge that can affect image quality.

### Objective
The primary goal is to develop a model capable of accurately classifying various rice varieties, even in the presence of impulse noise. The project involves meticulous dataset preprocessing, model compilation, training, and evaluation of both clean and noisy test images.

### Dataset
The dataset comprises images of five rice varieties: Arborio, Basmati, Ipsala, Jasmine, and Karacadag. Label encoding is employed for effective model training, and a strategic train-test split ensures robust evaluation.

### Model Architecture
The CNN architecture consists of Conv2D layers for spatial feature extraction, MaxPooling2D layers for dimension reduction, Flatten layers for data preparation, and Dense layers for intricate pattern recognition and classification.

### Evaluation Metrics
The model's performance is assessed using standard metrics, including accuracy, precision, recall, and F1-score. The classification report provides a comprehensive overview of the model's capabilities.

### Noise Testing
A distinctive aspect involves testing the model on noisy images with impulse noise at three levels (20%, 40%, 60%). The model demonstrates remarkable adaptability and robustness, maintaining high performance even in challenging conditions.

### Results
The model showcases impressive accuracy, precision, recall, and F1-scores across clean and noisy test images. The low loss at each noise level indicates efficient learning and adaptation. Insights, including confusion matrices, provide a detailed understanding of the model's performance..........................................

#########################For clean images#########################
Confusion Matrix:
[[2973    0    0   14   13]
 [   0 2931    0   69    0]
 [   3    0 2987    9    1]
 [   1   10    0 2989    0]
 [  62    0    0    0 2938]]

Classification Report:
              precision    recall  f1-score   support

     Arborio       0.98      0.99      0.98      3000
     Basmati       1.00      0.98      0.99      3000
      Ipsala       1.00      1.00      1.00      3000
     Jasmine       0.97      1.00      0.98      3000
   Karacadag       1.00      0.98      0.99      3000

    accuracy                           0.99     15000
   macro avg       0.99      0.99      0.99     15000
weighted avg       0.99      0.99      0.99     15000

Performance Score:
Loss: 0.09057945013046265, Accuracy: 0.9878666400909424
..................................................................................................................

#########################For noisy images#########################
Confusion Matrix:
[[2972    0    0   13   15]
 [   0 2942    0   58    0]
 [   5    0 2986    9    0]
 [   1   16    0 2983    0]
 [  58    0    0    0 2942]]

Classification Report:
              precision    recall  f1-score   support

     Arborio       0.98      0.99      0.98      3000
     Basmati       0.99      0.98      0.99      3000
      Ipsala       1.00      1.00      1.00      3000
     Jasmine       0.97      0.99      0.98      3000
   Karacadag       0.99      0.98      0.99      3000

    accuracy                           0.99     15000
   macro avg       0.99      0.99      0.99     15000
weighted avg       0.99      0.99      0.99     15000

Performance Score:
Loss: 0.0900530144572258, Accuracy: 0.9883333444595337
....................................................................................................................

## Noise Level Analysis

### Noise Level: 20%
- The model achieves accuracy of 98.69%, demonstrating robustness to a 20% noise level.
- High precision, recall, and F1-scores across all classes indicate accurate and reliable predictions.
- Minimal loss suggests efficient learning and adaptation to noise.

  Confusion Matrix:
[[2963    0    0    7   30]
 [   1 2979    0   20    0]
 [   2    0 2995    3    0]
 [  18   41   18 2922    1]
 [  55    0    0    0 2945]]

Classification Report:
              precision    recall  f1-score   support

     Arborio       0.97      0.99      0.98      3000
     Basmati       0.99      0.99      0.99      3000
      Ipsala       0.99      1.00      1.00      3000
     Jasmine       0.99      0.97      0.98      3000
   Karacadag       0.99      0.98      0.99      3000

    accuracy                           0.99     15000
   macro avg       0.99      0.99      0.99     15000
weighted avg       0.99      0.99      0.99     15000


Performance Score:
Loss: 0.11871158331632614, Accuracy: 0.9869333505630493

### Noise Level: 40%
- Exceptional accuracy at 98.80% showcases resilience to increased noise.
- Consistently high precision, recall, and F1-scores suggest reliable performance.
- Low loss indicates effective learning and adaptation to higher noise levels.

  Confusion Matrix:
[[2965    0    0    7   28]
 [   0 2982    0   18    0]
 [   2    0 2994    4    0]
 [  14   40   10 2935    1]
 [  56    0    0    0 2944]]

Classification Report:
              precision    recall  f1-score   support

     Arborio       0.98      0.99      0.98      3000
     Basmati       0.99      0.99      0.99      3000
      Ipsala       1.00      1.00      1.00      3000
     Jasmine       0.99      0.98      0.98      3000
   Karacadag       0.99      0.98      0.99      3000

    accuracy                           0.99     15000
   macro avg       0.99      0.99      0.99     15000
weighted avg       0.99      0.99      0.99     15000

Performance Score:
Loss: 0.09435475617647171, Accuracy: 0.9879999756813049

### Noise Level: 60%
- Admirable performance at a 60% noise level showcases model adaptability.
- Consistent high precision, recall, and F1-scores indicate reliable predictions.
- Robustness demonstrated with low loss even in substantial noise.

  Confusion Matrix:
[[2960    0    0    6   34]
 [   1 2971    1   27    0]
 [   2    0 2994    4    0]
 [  18   49   16 2917    0]
 [  64    0    0    0 2936]]

Classification Report:
              precision    recall  f1-score   support

     Arborio       0.97      0.99      0.98      3000
     Basmati       0.98      0.99      0.99      3000
      Ipsala       0.99      1.00      1.00      3000
     Jasmine       0.99      0.97      0.98      3000
   Karacadag       0.99      0.98      0.98      3000

    accuracy                           0.99     15000
   macro avg       0.99      0.99      0.99     15000
weighted avg       0.99      0.99      0.99     15000


Performance Score:
Loss: 0.09869148582220078, Accuracy: 0.9851999878883362

### Overall
The model exhibits remarkable performance across various noise levels, highlighting its suitability for real-world scenarios. High accuracy, coupled with consistent precision, recall, and F1-scores, underscores the model's robustness and efficient learning in challenging conditions.

## Conclusion
This project not only advances our understanding of CNNs but also emphasizes their potential for practical deployment in image classification tasks, even when images are affected by unpredictable and disruptive noise. The comprehensive README provides an overview of the project, its objectives, methodology, and key findings.
