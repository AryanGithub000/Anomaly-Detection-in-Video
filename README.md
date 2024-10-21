# Anomaly Detection in Surveillance Videos 

## Dataset

This project uses a model that has been pre trained on fog and haze datasets on roughly 700 videos, furthermore more than 10 Lakh frames were generated for which this model has been trained on.
The download link for the model is attached below:

[Download Model](https://drive.google.com/file/d/1wM52gaDvSl5QxoDSKs8SeqjwUIjCZuF3/view?usp=sharing)

## Abstract

This project introduces an approach to simulate real-world situations like fog and haze in surveillance videos for anomaly detection. We used the UCF dataset and created a synthetic dataset based on several classes of anomalies (e.g., Arson, Fighting, Road Accident, Arrest, Abuse). The project includes normalization techniques, a CNN + LSTM model trained on the synthetic dataset, and a Gradio UI frontend for real-world applications.

## Features

- Simulation of fog and haze effects on surveillance videos
- Preprocessing pipeline for video data
- CNN + LSTM model for anomaly detection
- Gradio UI for frontend visualization
- Performance evaluation using confusion matrices

## Technologies Used

- Python
- OpenCV (cv2)
- NumPy
- Pandas
- scikit-learn
- TensorFlow and Keras
- OS Library
- Gradio

## Installation

Run the Gradio_UI notebook and download the model from the link 
Run in IDE 

## Dataset

The project uses a synthetic dataset created from the following:
1. UCF Crime dataset
2. Augmented with simulated fog and haze effects.
3. The original UCF-Crime dataset can be found [here](https://www.crcv.ucf.edu/projects/real-world/).

## Model Architecture

The model uses a CNN + LSTM architecture:
- Three convolutional layers with increasing filter sizes (32, 64, 128) and ReLU activation
- MaxPooling layers after each convolutional layer
- A Flatten layer
- Two Dense layers: one hidden layer with 128 neurons and an output layer with sigmoid activation

## Results

The model achieved a training accuracy of 0.8909 and a test accuracy of 0.9506.

## Future Work

- Expansion to other environmental conditions (rain, snow, varying lighting levels)
- Incorporation of more datasets based on environmental changes

## Ouput
![Demo Result](https://github.com/user-attachments/assets/b8194f18-e385-4371-aa56-e0eb61922232)



