#include "NeuralNetwork.hpp"
#include "MathUtils.hpp"
#include <iostream>

using namespace std;

int main()
{
	// seed rng
	srand(time(0));

	vector<unsigned> shape = {2, 3, 3, 4};
	vector<ActivationFunction> activationFunctions = {none, ReLU, ReLU, softmax};
	vector<InitFunction> weightInitFunctions = {zeros, heInitialization, heInitialization, heInitialization};
	vector<InitFunction> biasInitFunctions = {zeros, zeros, zeros, zeros};

	NeuralNetwork net(shape, activationFunctions, weightInitFunctions, biasInitFunctions, CategoricalCrossEntropy);

	MatrixXd inputs(2, 3);
	inputs.setRandom();

	MatrixXd y_true(3, 1);
	for (int i = 0; i < y_true.rows(); i++) {
		y_true(i,0) = (int) rand() % 3;
	} 

	// cout << "y to be passed: \n"
	// 	 << y_true << endl;

	MathUtils::convertToOneHotEncoded(y_true, 4);

	// cout << "y in one hot encoded form: \n"
	// 	 << y_true << endl;

	net.forward(inputs, y_true);

	cout << "******************" << endl;

	double learningRate = 0.1;

	net.backward(learningRate);

	cout << "******************" << endl;
	cout << "mean loss: " << net.meanLoss << endl;
	cout << "******************" << endl;

	for (int i = 0; i < 10000; i++)
	{

		net.forward(inputs, y_true);
		net.backward(learningRate);
	}

	cout << "******************" << endl;
	cout << "mean loss: " << net.meanLoss << endl;
	cout << "******************" << endl;
	
	return 0;
}