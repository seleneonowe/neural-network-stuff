## What is this?
A general dense feedforward neural network c++ implementation! Currently only a few types of activation, initialization, and cost functions are supported but the project can easily be expanded to include more.

## How can I build the project?
### Dependencies:
The project relies on [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page), a c++ template library for linear algebra; specifically the implementations for dynamically-sized matrices and vectors and associated common operations. Eigen can be downloaded from links on [the wiki](https://eigen.tuxfamily.org/index.php?title=Main_Page) or directly from [their gitlab](https://gitlab.com/libeigen/eigen).

### Building:
To build the project, clone Eigen and this repository. Ensure Eigen is present in your include path.

Then simply run the command `make` to build the project, and `./bin.out` to run the resulting executable.

You may also run `make clean` to clear unwanted object/executable files etc.

## What can I do with it?
You can solve any problems that can be solved with this type of neural network. In principle, anyway. Bear in mind that this project was something I made as an exercise in learning c++ and learning some of the concepts involved in constructing neural networks, not for solving modern problems. 

Genuine research or industrial applications of machine learning that want to use an algorithm like the ones implemented here ought to be doing the required matrix- and vector-operations involved in training a network in parallel on many GPUs.

To use this would be slower but technically possible.

## What happens when I run it as-is?
Right now, it procedurally generates a dataset (see `DataSetGenerator.cpp`), and then creates a neural network and tries to train it on that data. 

By editing `main.cpp` one can change all the paramaters used, like the number of classifications in the dataset generated, the number of layers and neurons in the neural network, the activation/initialization/loss functions used in the network, the number of iterations to learn for, and the learning rate.

By doing so one can create little sample models that encounter various common issues neural networks used in machine learning on real data may encounter. For example: excessive computation time due to low learning rate; instability and oscillations due to high learning rate; overfitting; local optima; the problem of poorly conditioned curvature in the solution space; etc. 