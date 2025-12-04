
# Team Number – CB10
Project Title - Enhanced Multi-Zonal Forest Type Classification
 Using YOLOv8 and EuroSAT for Scalable
 Environmental Monitoring

## Team Info
- 22471A05H8 — **Name** Nakka Vijay Bhasker Reddy ( https://www.linkedin.com/in/nakka-vijay-bhaskar-reddy-6000b7265?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app  )
_Work Done: As the team leader, I handled the core parts of the project and guided the overall development process. My responsibilities included:

🔹 Project Planning & Coordination

Planned the project structure and workflow

Assigned tasks to team members

Ensured the project stayed on track and met deadlines

🔹 Dataset Preparation

Collected, organized, and cleaned the EuroSAT dataset

Performed EDA and preprocessing

Converted the dataset into YOLOv8 format

🔹 Model Development

Trained multiple YOLOv8 models (nano, small, medium)

Tuned hyperparameters and compared model performance

Selected the best-performing model for deployment

🔹 Web Application Development

Integrated the trained model into a working web application

Built the frontend for image upload and prediction display

Connected backend logic to run model inference

🔹 Documentation & Reporting

Prepared project documentation

Wrote README sections for GitHub

Explained workflow, results, and deployment details

- 22471A05H7 — **Name** Nagaram Prasad Rao (https://www.linkedin.com/in/nagaram-prasad-rao-26a6b7280?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app  )
_Work Done: xxxxxxxxxx_

- 22471A05H6 — **Name** Mundlamuri Vijaya Kumarachari (https://www.linkedin.com/in/vijay-kumarachari-ba2290276?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app )
_Work Done: xxxxxxxxxx_



## Abstract
Forests are a critical component of our world’s
 health, but it is still difficult today to see them properly over large
 distances. This research proposes a real-world and intelligent
 solution using satellite images and DL to detect forest stands
 with greater accuracy. Aiming to achieve faster and more scalable
 forest analysis than traditional field surveys, this study develops
 an automated land cover identification system using the EuroSAT
 dataset. The primary objective of this research was to train
 and compare the YOLOv8-nano model in order to analyze its
 performance in detecting forests and other land covers. The
 model was validated using 27,000 images across 10 different
 classes in order to provide extensive exposure to a variety of
 data. It showed robust performance with a peak accuracy of
 97.5%. These results emphasize the high capabilities of the
 model for real-time forest observation and wise environmental
 decision making. In general, this research is an important step
 towards using artificial intelligence and satellite imagery to
 enhance conservation, as well as to more effectively manage forest
 resources at different locations.

## Paper Reference (Inspiration)
👉 **[Paper Title EfficientNet Deep Learning Model for Satellite Image 
Classification Using the EuroSAT Dataset
  – Author Names 1.Buse Saricayir
2.Caner Ozcan
 ](Paper URL here)** http://CEUR-WS.org/Vol-3988/paper13.pdf
Original conference/IEEE paper used as inspiration for the model.

---

## Our Improvement Over Existing Paper
The base paper, “EfficientNet Deep Learning Model for Satellite Image Classification Using the EuroSAT Dataset” by Buse Saricayir and Caner Ozcan, focuses primarily on general land-use and land-cover (LULC) classification using EfficientNet-B0. While the study demonstrates high accuracy and computational efficiency, its scope remains limited to baseline classification without addressing real-time detection, multi-zonal analysis, or forest-specific monitoring. In contrast, the proposed project significantly advances this foundation through multiple methodological and operational improvements tailored specifically for forest-type detection, scalability, and real-time environmental monitoring.

First, unlike the base paper that uses a pure classification model (EfficientNet-B0), the proposed work introduces YOLOv8, a modern object-detection and image-classification architecture capable of identifying, localizing, and differentiating forest zones in real time. This shift from simple patch classification to spatially aware forest-type detection represents a major improvement in practical utility. YOLOv8 also supports larger model variants and robust augmentation, enabling superior performance across diverse forest classes, seasonal variations, and complex textures—limitations that the base paper does not address.

Second, the base paper uses the EuroSAT dataset “as-is,” while the proposed system expands its capability through multi-zonal forest-specific restructuring, dataset balancing, and advanced augmentation to improve accuracy in forest-related classes that EfficientNet struggles with (e.g., confusion between Forest vs. Herbaceous Vegetation). Your project overcomes these weaknesses by building forest-focused subsets, applying stronger augmentation strategies (rotation, hue, vegetation-index variation), and evaluating multiple YOLOv8 variants (n, s, m) to optimize accuracy. This results in more stable predictions, higher class separability, and better generalization to real-world forestry applications.

Finally, while the existing paper focuses solely on achieving high accuracy, your project moves toward scalable environmental monitoring, including multi-zonal forest mapping, real-time inference capability, and integration readiness for large-scale systems. The use of YOLOv8 enables edge-deployment, faster inference speeds, and suitability for drone- or satellite-based monitoring pipelines—key features absent in the EfficientNet-based baseline study. This transition from an academic benchmark model to a deployable, scalable system significantly broadens the practical environmental impact of satellite-image classification.
## About the Project
### What my project does

My project, “Enhanced Multi-Zonal Forest Type Classification Using YOLOv8 and EuroSAT,” automatically identifies and classifies different forest types using satellite images. I use the EuroSAT dataset along with the YOLOv8 model to detect forest zones more accurately and efficiently.

### Why this project is useful

This project is useful because it helps in:

Monitoring forests across large areas

Detecting changes in vegetation or deforestation

Supporting environmental management and planning

Providing faster and more accurate satellite-based analysis

Traditional methods are slow and manual, but my model offers real-time, scalable, and automated forest classification, making it valuable for researchers, environmental agencies, and conservation work.

### How the project works (Workflow)

Input → Processing → Model Training → Output

1.Input
I use satellite images from the EuroSAT dataset, focusing mainly on forest-related classes.

2.Processing

I clean and organize the dataset

Apply data augmentation

Prepare multi-zonal forest regions

Convert data into YOLOv8 format

3.Model Training

Train multiple YOLOv8 models (YOLOv8n, YOLOv8s, YOLOv8m)

Compare accuracy, precision, recall, and F1-score

Select the best-performing model

4.Output
The final model predicts:

Forest type

Spatial region (bounding boxes if needed)

Classification confidence score

This produces a fast and scalable system for forest-type classification using satellite imagery.

## Dataset Used
👉 **[Dataset Name : EUROSAT ](Dataset URL : https://www.kaggle.com/datasets/apollo2506/eurosat-dataset)**

**Dataset Details:**
The EuroSAT dataset is a publicly available benchmark dataset created using satellite images from the Sentinel-2 mission of the European Space Agency (ESA). It was developed to support research in land-use and land-cover classification using deep learning techniques. The dataset contains high-quality multispectral images that capture different geographical regions across Europe.

EuroSAT consists of more than 27,000 labeled images, each belonging to one of 10 land-cover classes, such as forest, river, residential areas, industrial buildings, and agricultural fields. Each image is provided at a size of 64 × 64 pixels, making it suitable for machine learning and deep learning models. The dataset is available in both RGB and 13-band multispectral formats, allowing researchers to explore a variety of spectral features.
Classes (10 Categories)

EuroSAT contains 10 land-cover classes:

1.Annual Crop

2.Forest

3.Herbaceous Vegetation

4.Highway

5.Industrial Buildings

6.Pasture

7.Permanent Crop

8.Residential

9.River

10.Sea/Lake
Because of its balanced classes and clean labeling, EuroSAT has become one of the most widely used datasets for remote sensing research. It is commonly used to train and evaluate models like CNNs, EfficientNet, and Vision Transformers for satellite image classification. The dataset supports applications in environmental monitoring, urban planning, and crop or forest mapping.
## Dependencies Used
This project uses the following major libraries and tools:

1. Python Libraries

Ultralytics – For training and running YOLOv8 models

PyTorch – Deep learning framework used by YOLOv8

NumPy – Numerical operations and array handling

Pandas – Dataset loading and preprocessing

Matplotlib / Seaborn – Visualizing training results and graphs

scikit-learn – Evaluation metrics (accuracy, precision, recall, F1-score)

opencv-python (cv2) – Image loading, resizing, and preprocessing
For YOLOv8 Training -Ultralytics YOLOv8

## EDA & Preprocessing
Exploratory Data Analysis (EDA)

I started by exploring the EuroSAT dataset to understand the distribution of classes and the characteristics of forest-related images. I checked how many samples each class contains, visualized random images, and identified which classes look similar (for example, Forest vs Herbaceous Vegetation). This helped me understand where the model might face confusion and what preprocessing steps would improve performance.

🔹 Preprocessing Steps
1. Dataset Cleaning

I removed corrupted images, ensured all files were in a consistent format, and verified that each image was placed in the correct class folder.

2. Image Resizing

I resized all images to YOLO-compatible input dimensions to keep the training efficient and consistent.

3. Data Augmentation

To improve generalization, I applied several augmentation techniques such as:

Rotation

Horizontal and vertical flips

Brightness / contrast adjustments

Hue and saturation changes

Random cropping

These augmentations helped the model learn variations in lighting, seasons, and vegetation colors.


## Model Training Info
I trained multiple YOLOv8 models (YOLOv8n, YOLOv8s, YOLOv8m) on the preprocessed EuroSAT forest dataset.
During training, I monitored loss curves, accuracy trends, and validation performance to select the best model.
I used different augmentation settings, batch sizes, and epochs to identify the optimal configuration.

Key Training Steps:

Loaded the dataset in YOLOv8 format (data.yaml)

Trained models on GPU for faster performance

Used early stopping and validation monitoring

Compared results across nano, small, and medium YOLOv8 models

The goal of the training phase was to find the model with the best balance of accuracy, speed, and generalization.
## Model Testing / Evaluation
After training, I tested the models on the unseen test set to measure real performance.
I evaluated each model using standard classification metrics:

Accuracy

Precision

Recall

F1-Score

I also examined confusion matrices to understand which forest types were classified correctly and where the model struggled.
By comparing these metrics across different YOLOv8 variants, I selected the most reliable and stable model for deployment.

## Results
The final model showed strong performance in classifying different forest zones from EuroSAT satellite images.
The trained YOLOv8 model achieved:

High Accuracy across all forest-related classes

Strong Precision & Recall due to robust augmentation

Reduced misclassification compared to the base EfficientNet paper

Better generalization in forest-heavy regions

Overall, the results confirm that YOLOv8 provides an efficient and scalable solution for multi-zonal forest classification, making it suitable for real-world environmental monitoring tasks.
## Limitations & Future Work
Future work may involve deploying the model in real
time forest monitoring platforms or extending it using larger
 datasets like Sentinel-2 imagery for improved regional gener
alization
## Deployment Info
I deployed my project as a web application, allowing users to upload EuroSAT-like satellite images and get forest-type classification results directly through a simple interface. The goal was to make the model easy to use without requiring any technical setup.
