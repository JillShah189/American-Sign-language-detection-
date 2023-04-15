# American-Sign-language-detection-
Sign Language Gesture Recognition Using Resnet

Sign Language Alphabet Recognition
This project uses deep learning techniques to recognize American Sign Language (ASL) alphabet signs using a convolutional neural network (CNN) based on the ResNet architecture.

Dataset
The dataset used in this project consists of 27 classes representing each letter in the ASL alphabet. The data was split into training and validation sets using a 80:20 split ratio. The dataset was preprocessed using the ImageDataGenerator function from Keras to perform data augmentation techniques such as random rotations, shifts, and flips. 
Dataset - https://www.kaggle.com/datasets/muhammadkhalid/sign-language-for-alphabets

Model
The model used is based on the ResNet architecture, which consists of residual blocks that allow for better convergence and training of deep neural networks. The architecture is composed of convolutional layers, batch normalization layers, ReLU activation functions, and a dense output layer with a softmax activation function. The model was trained using the Adam optimizer and the sparse categorical crossentropy loss function.

Results
The trained model achieved a validation accuracy of 97.28% after 20 epochs of training. The training and validation accuracy and loss were plotted for each epoch, showing good convergence and generalization of the model.

Files
Code: This contains main.ipynb: Jupyter notebook containing the code for training and evaluating the model
Reports: contains mid sem and end sem reports.
Results - This file contains the screenshots of results of the alphaet prediction. 
Weekly reports- this contains our weekly progress. 
README.md: This file, containing information about the project

Requirements
Python 3.7+
TensorFlow 2.x
Keras 2.x
Matplotlib
Numpy

Usage
Download the dataset from the Kaggle website given above. 
Open the Jupyter notebook main.ipynb and run the cells to train and evaluate the model.
