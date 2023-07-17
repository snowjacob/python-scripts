## Python-Scripts

This is my Python-Scripts folder, a collection of code dedicated to machine learning and numerical optimization. Modules used in these programs include numpy, pytorch, and tensorflow. 

#### About

In this repository, I leverage widely adopted machine learning libraries such as PyTorch and TensorFlow to build powerful models. Additionally, you'll find mathematical Python scripts that make use of NumPy, providing efficient computations and numerical analysis.

#### Scripts

🔆[Methods in Nonlinear Optimization](optimization/NonLinOptipy.py): The Nonlinear Optimization project focuses on implementing Quasi-Newton and Conjugate Gradient methods for finding the maximum or minimum of specific equations. This project explores the application of these methods on two distinct functions: the Hump Function and the Rosenbrock Function. The Hump Function is a complex equation involving multiple variables, while the Rosenbrock Function is a well-known function used to evaluate optimization algorithms. By leveraging the power of these nonlinear methods, this project aims to provide efficient and accurate solutions to optimization problems. Through the exploration of different techniques and functions, this project serves as a valuable resource for understanding and implementing nonlinear optimization algorithms in Python.

📷[Pytorch CNN Development](pytorch/PytorchCNN.py): The Convolutional Neural Network (CNN) project in PyTorch focuses on building and training a powerful model for image classification tasks. The project involves implementing a CNN architecture with multiple convolutional and fully connected layers to learn complex patterns and features from input images. The model utilizes popular PyTorch modules such as Conv2d, PReLU, BatchNorm2d, and MaxPool2d to extract meaningful representations from the input data. With the help of data augmentation techniques like random rotation, horizontal and vertical flips, and random cropping, the model is trained on a labeled dataset. The training process involves optimizing the model parameters using the Adam optimizer and minimizing the BCELoss (Binary Cross Entropy Loss) criterion. Additionally, the project includes evaluation on a separate validation dataset to measure the model's performance. Finally, the trained model can be used for making predictions on new images by loading the saved model and applying it to the test dataset.

📡[Wildfire Prediction Network using Tensorflow](wildfire_prediction/fire_predict.ipynb): The project is a classification model built using TensorFlow and Keras in a Jupyter Notebook. The model architecture consists of multiple Conv2D layers with ReLU activation, followed by BatchNormalization, AveragePooling2D, MaxPooling2D, Flatten, and Dense layers. Dropout layers are added to prevent overfitting. The model is trained using the Adam optimizer with specified learning rate, beta values, epsilon, and weight decay. The loss function used is BinaryCrossentropy, and accuracy is chosen as the metric. In addition to the model development, this project also showcases the utilization of TensorFlow.js, Node.js, and Express to create a web application version of the trained model. The full project is located [here](https://github.com/snowjacob/FirePredictionJS).

#### Getting Started
To run any of the projects in this repository, follow the instructions provided in their respective README files. Make sure you have a proper Python environment configured, as well as any modules / packages mentioned in the README files.

#### Contributing
If you'd like to contribute to any of the projects or have ideas for new projects, feel free to submit a pull request. I'm always open to collaboration and exploring new perspectives.
