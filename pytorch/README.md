# <img src="https://pytorch.org/assets/images/pytorch-logo.png" alt="Image" style="width:3.5%;"> Pytorch Projects

### Pytorch Convolutional Neural Network (CNN) ~ [PytorchCNN.py](pytorch/PytorchCNN.py)
This project implements a Convolutional Neural Network (CNN) using PyTorch for image classification tasks. The CNN model architecture consists of multiple layers, including convolutional layers, activation functions, batch normalization, pooling layers, and fully connected layers. The model is designed to work with images of size 256x256 and has been optimized using the Adam optimizer with specific hyperparameters.

#### Architecture
The CNN architecture consists of multiple convolutional layers with different filter sizes, kernel sizes, and strides, 
followed by PReLU activation functions and batch normalization layers. It also includes average pooling and max pooling layers, 
a flatten layer to convert the output, dropout layers with a specified dropout rate, fully connected layers with different units, and a 
final output layer with a sigmoid activation function for binary classification.


#### Usage
To use this script, follow these steps:

Ensure that PyTorch, torchvision, and numpy are installed on your system.
Replace the --train and --dev arguments in the script with the paths to your training and development datasets, respectively.
Adjust the hyperparameters, such as batch size, number of epochs, learning rate, dropout rate, and others, as per your requirements.
Run the script and observe the training progress, which will print the loss and accuracy for each epoch.
After training, the script loads a pre-trained model from a file named model.pt. Make sure to save your trained model using the same filename if you plan to use it for predictions later.
To make predictions, create a text file (script.txt) with the paths of the images you want to predict, with each path on a new line. Update the script_file variable in the script to point to this file.
Run the script to generate predictions for the provided images. The predictions will be saved as a numpy array in a file named predictions.npy.

#### References
This implementation was inspired by the following article:

[*Recent Advances in Convolutional Neural Networks*](https://arxiv.org/pdf/1512.07108.pdf)

Please refer to the article for detailed information on the CNN architecture and related concepts.

#
