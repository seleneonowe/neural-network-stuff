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

	int batchSize = 4;
	MatrixXd inputs(2, batchSize);
	inputs.setRandom();

	MatrixXd y_true(batchSize, 1);

	for (int i = 0; i < y_true.rows(); i++) {
		y_true(i,0) = (int) rand() % 3;
	} 

	MathUtils::convertToOneHotEncoded(y_true, 4);

	net.forward(inputs, y_true);

	cout << "******************" << endl;

	double learningRate = 0.1;

	net.backward(learningRate);

	cout << "******************" << endl;
	cout << "mean loss: " << net.meanLoss << endl;
	cout << "******************" << endl;

	for (int i = 0; i < 1000; i++)
	{

		net.forward(inputs, y_true);
		net.backward(learningRate);
	}

	cout << "******************" << endl;
	cout << "mean loss: " << net.meanLoss << endl;
	cout << "******************" << endl;

	return 0;
}