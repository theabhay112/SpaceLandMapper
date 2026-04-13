# SpaceLandMapper

**Integrating AI and Earth Observation for Land Classification**

SpaceLandMapper is a discipline-specific Artificial Intelligence project that uses deep learning and satellite imagery to classify land-use types. The project combines a baseline Convolutional Neural Network (CNN), an EfficientNetB0 transfer-learning pipeline, and a Streamlit prototype for image prediction and grid-based land mapping.

## Project Overview

Land-use classification is important in urban planning, agriculture, environmental monitoring, and resource management. Traditional analysis of satellite imagery can be slow and difficult to scale. This project explores how AI can automate classification from Earth observation data and support decision-making through a practical prototype.

## Objectives

- Build a baseline CNN model for land-use classification
- Improve performance using EfficientNetB0 with transfer learning
- Evaluate results using validation metrics and confusion matrices
- Develop a working prototype for image prediction and grid-based mapping

## Dataset

This project uses the **EuroSAT** dataset.

- Around **27,000** labelled RGB satellite image patches
- **10 land-use classes**
- Based on **Sentinel-2** Earth observation imagery

Example classes include:

- AnnualCrop
- Forest
- HerbaceousVegetation
- Highway
- Industrial
- Pasture
- PermanentCrop
- Residential
- River
- SeaLake

Reference:
Helber, P., Bischke, B., Dengel, A. and Borth, D. (2019) *EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification*.

## Methodology

### 1. Data Preparation
- Organised labelled image paths
- Checked class distribution
- Split data into **70% training**, **15% validation**, and **15% test**
- Resized images to **224 x 224**
- Applied augmentation to training data:
  - rotation
  - zoom
  - horizontal flip

### 2. Baseline CNN
A custom baseline CNN was built as the first benchmark model. It includes:
- 3 convolution + pooling stages
- flattening
- dense layers
- dropout

This model provides an initial performance baseline before using transfer learning.

### 3. EfficientNetB0 Transfer Learning
A stronger model was developed using **EfficientNetB0** with **ImageNet pretrained weights**.

Process:
- freeze the backbone first
- train the top classification layers
- fine-tune upper layers with a lower learning rate

This helps the model reuse previously learned visual features while adapting to land-use classification.

### 4. Training Support Methods
The following techniques were used:
- transfer learning
- data augmentation
- early stopping
- ReduceLROnPlateau
- model checkpointing

### 5. Evaluation
Models were evaluated using:
- training and validation accuracy
- training and validation loss
- confusion matrix
- classification report
- class-wise performance analysis

## Prototype

A **Streamlit** application was developed as the final prototype.

### Features
- Upload a satellite image
- Predict the land-use class
- Generate a grid-based land classification map
- Visualise land distribution using colour overlays

This allows the project to move beyond model training into a usable demonstration tool.

## Project Structure

```text
SpaceLandMapper/
│
├── app.py                     # Streamlit prototype
├── test.ipynb                    # Model development / training script
├── BaselineModel.ipynb        # Notebook version (if used)
├── saved_model/               # Saved trained model files
├── assets/                    # Images / outputs / charts
├── README.md                  # Project documentation
```

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit
- OpenCV 

## How to Run

### 1. Clone the repository
```bash
git clone <your-repository-link>
cd SpaceLandMapper
```

### 2. Install dependencies


 install the main libraries :
```bash
pip install tensorflow streamlit  pandas matplotlib scikit-learn 
```

### 3. Run the training script
```bash
python test.ipynb
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

## Example Workflow

1. Prepare the EuroSAT dataset
2. Train the baseline CNN
3. Train EfficientNetB0 frozen model
4. Fine-tune the upper layers
5. Evaluate with confusion matrix and reports
6. Launch Streamlit prototype
7. Upload an image and generate predictions or grid mapping outputs


## Ethical and Responsible AI Considerations

- The system should support, not replace, expert judgement
- Land-use predictions should be interpreted carefully
- Responsible AI use requires transparency, validation, and awareness of limitations
- Outputs should not be used blindly for critical planning decisions


## Authors

- Abhay Kumar
- Mashrafi Azad
- Robuil

