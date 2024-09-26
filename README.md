# Classifying-Walking-and-Jumping-Accel-Data

## Requirements
- Python
  - sklearn
  - pandas
  - numpy
  - joblib
  - tkinter
  - matplotlib
  - h5py

## Overview
This project aimed to create a desktop application that, given a CSV  file with 3D acceleration and time, could label the data as walking or jumping and display the results readably.

## Project Components

### 1. Data Collection
Data was collected via the mobile app PhyPphox. This was chosen because it did not require any physical technology that the team members did not already have and because the data could be saved as a CSV. Data was collected multiple times in different ways for both the walking and jumping data.

### 2. Data Storing
The data started as many CSV files labeled as either walking or jumping. Before the data was stored it was manipulated in several ways.
1. **Labeling:** CSV files are renamed.
2. **Concatenation:** CSV files are combined. Two files remain after this, one walking and one jumping.
3. **Shuffling:** To train the model the data should not be in chronological order, so shuffling is needed.
4. **Splitting:** Each file is split in two, one for training and the other for testing.
After these four steps, the CSV files are saved in the HDF5 format.

### 3. Visualization
To see information about this part see the Final Report pdf. 

### 4. Preprocessing
To reduce noise a moving average filter was used. To normalize the data sklearn's StandardScaler was used. 

### 5. Feature Extraction & Normalization
With the help of Pandas a list of features were extracted. For details see the Final Report pdf

### 6. Training the Classifier
Not much to say here. See report or code if interested

### 7. Model Deployment
With Tkinter, a basic GUI was implemented which allows for file upload.
