# Intel Image Classification using PyTorch

## Overview

This project implements an image classification model using **PyTorch** to classify images from the **Intel Image Classification Dataset**.

The model is trained to identify different natural scenes and classify them into the following categories:

* Buildings
* Forest
* Glacier
* Mountain
* Sea
* Street

The project covers the complete deep learning workflow, including data loading, preprocessing, model building, training, validation, testing, and performance visualization.

---

## Dataset

The Intel Image Classification Dataset contains images belonging to six different scene categories:

| Class | Description |
| ----- | ----------- |
| 0     | Buildings   |
| 1     | Forest      |
| 2     | Glacier     |
| 3     | Mountain    |
| 4     | Sea         |
| 5     | Street      |

The dataset is divided into training, validation, and testing data.

---

## Tech Stack

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* PIL

---

## Project Workflow

```text
Image Dataset
      ↓
Data Loading
      ↓
Data Preprocessing
      ↓
Data Augmentation
      ↓
PyTorch DataLoader
      ↓
CNN Model
      ↓
Training
      ↓
Validation
      ↓
Model Evaluation
      ↓
Accuracy & Loss Visualization
      ↓
Image Classification
```

---

## Data Preprocessing and Augmentation

Images are transformed before being passed to the model.

The preprocessing pipeline includes:

* Image resizing
* Data augmentation
* Conversion to PyTorch tensors
* Image normalization

Example transformations include:

```python
transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
```

Data augmentation helps improve model generalization and reduces overfitting.

---

## Model Architecture

The CNN model consists of multiple convolutional layers for feature extraction followed by pooling layers and fully connected layers for classification.

General architecture:

```text
Input Image
     ↓
Convolution Layer
     ↓
ReLU Activation
     ↓
Max Pooling
     ↓
Convolution Layer
     ↓
ReLU Activation
     ↓
Max Pooling
     ↓
Flatten
     ↓
Fully Connected Layer
     ↓
Output Layer
     ↓
6 Scene Classes
```

---

## Training

The model is trained using:

* **Loss Function:** CrossEntropyLoss
* **Optimizer:** Adam
* **Framework:** PyTorch

Training process:

```text
Batch of Images
      ↓
Forward Pass
      ↓
Model Predictions
      ↓
Calculate Loss
      ↓
Backward Propagation
      ↓
Optimizer Updates Weights
      ↓
Repeat for Multiple Epochs
```

Basic training loop:

```python
for images, labels in train_loader:

    optimizer.zero_grad()

    outputs = model(images)

    loss = criterion(outputs, labels)

    loss.backward()

    optimizer.step()
```

---

## Model Evaluation

After training, the model is evaluated on unseen test data.

The evaluation process calculates:

* Test Loss
* Test Accuracy

```text
Test Images
     ↓
Trained CNN Model
     ↓
Predicted Classes
     ↓
Compare with Actual Labels
     ↓
Calculate Accuracy
```

---

## Results Visualization

The project visualizes the training performance using:

* Training Loss
* Validation Loss
* Training Accuracy
* Validation Accuracy

These plots help analyze:

* Model learning progress
* Overfitting
* Underfitting
* Generalization performance

---

## Sample Prediction

The trained model can be used to predict the class of an unseen image.

```text
Input Image
     ↓
Preprocessing
     ↓
CNN Model
     ↓
Prediction Scores
     ↓
Highest Probability Class
     ↓
Final Predicted Scene
```

Example output:

```text
Predicted Class: Mountain
```

---

## Key Learnings

Through this project, I gained hands-on experience with:

* Building CNN models using PyTorch
* Working with `Dataset` and `DataLoader`
* Image preprocessing using Torchvision transforms
* Data augmentation
* Training and validation loops
* Forward propagation and backpropagation
* Loss calculation and optimization
* Model evaluation
* Accuracy and loss visualization
* Image classification using deep learning

---

## Future Improvements

Possible improvements include:

* Using transfer learning with pretrained models such as ResNet
* Hyperparameter tuning
* Adding a learning rate scheduler
* Using early stopping
* Building a web application for image classification
* Deploying the trained model using FastAPI or Streamlit

---

## Repository Structure

```text
intel-image-classification-pytorch/
│
├── data/
│
├── intel_image_classification.ipynb
│
├── model.py
│
├── requirements.txt
│
├── README.md
│
└── results/
    ├── accuracy_plot.png
    ├── loss_plot.png
    └── sample_predictions.png
```

## Author

**Shravan Kundap**

Electronics & Telecommunication Engineering student exploring:

**Data Analytics | Machine Learning | Deep Learning | AI | IoT**
