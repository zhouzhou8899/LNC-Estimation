# Leaf Nitrogen Content Estimation
This repository contains model weights for testing and documentation related to the tests.
1. Introduction

This repository provides files for ensemble models and progressive transfer learning models used to predict leaf nitrogen content (LNC), along with inference and prediction code.

2. Model File Description
The `models/` directory in this repository contains pre-trained model files (in `.pkl` format).
File Format: Python `joblib` serialization files.
File Contents: Includes the complete model object, preprocessing parameters (standardization processors), feature selection lists, and ensemble weights.
Regarding Model Weights: In this project, machine learning model weights are encapsulated together with the model structure within the `.pkl` file; loading this file restores all model parameter states.

3. Testing Instructions
After cloning this repository, proceed to configure the local environment: pip install -r requirements.txt
Then run the Python test script: python predict.py
