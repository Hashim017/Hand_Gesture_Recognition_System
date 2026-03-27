# Hand Gesture Recognition System

An AI-powered computer vision application that recognizes and classifies hand gestures in real-time using machine learning. Built with Python, this system captures hand images, trains a classifier model, and enables gesture-based control for media operations.

## Features
- Real-time hand gesture detection and recognition
- Multi-gesture classification model
- Webcam-based image capture and processing
- Dataset creation and validation
- Trained machine learning classifier
- Media control through hand gestures
- Camera compatibility checking
- Dataset visualization and analysis

## Tech Stack
- **Language:** Python (100%)
- **Computer Vision:** OpenCV, MediaPipe
- **Machine Learning:** Scikit-learn, NumPy
- **Image Processing:** PIL, NumPy
- **Data Handling:** Pickle serialization

## Project Structure
Hand_Gesture_Recognition_System/ ├── Collect_Images.py # Capture gesture images from webcam ├── Create_Dataset.py # Process images into training dataset ├── Train_Classifier.py # Train ML model on dataset ├── Inference_Classifier.py # Real-time gesture recognition ├── media_control.py # Control media with gestures ├── check_dataset.py # Analyze and visualize dataset ├── check_cameras.py # Test camera compatibility ├── model.p # Trained classifier model ├── data.pickle # Processed dataset └── README.md


## Quick Start
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run camera check: `python check_cameras.py`
4. Collect gesture images: `python Collect_Images.py`
5. Create dataset: `python Create_Dataset.py`
6. Train model: `python Train_Classifier.py`
7. Test recognition: `python Inference_Classifier.py`

## Workflow

### Phase 1: Data Collection
- `Collect_Images.py` - Captures hand gesture images using webcam
- Organized by gesture type for easy dataset management

### Phase 2: Dataset Preparation
- `Create_Dataset.py` - Processes images and extracts features
- `check_dataset.py` - Validates and visualizes collected data

### Phase 3: Model Training
- `Train_Classifier.py` - Trains classification model
- Generates `model.p` for inference

### Phase 4: Real-time Recognition
- `Inference_Classifier.py` - Live gesture detection and classification
- `media_control.py` - Performs actions based on recognized gestures

## Core Functionality
- **Gesture Detection:** Real-time hand tracking using MediaPipe
- **Classification:** ML-based gesture identification
- **Media Control:** Play, pause, volume, and navigation
- **Dataset Management:** Create, validate, and analyze training data
- **Accuracy:** High-performance gesture recognition

## System Requirements
- Python 3.7+
- Webcam
- 4GB+ RAM
- OpenCV compatible system

## Getting Help
For issues or questions, please open a GitHub issue.

**License:** MIT  

**Project Type:** AI/ML Final Project

**License:** MIT  
**Project Type:** AI/ML Final Project
