#include "NeuralNetwork.hpp"
#include "MathUtils.hpp"
#include <iostream>

using namespace std;

int main() {
	//seed rng
	srand(time(0));

	vector<unsigned> shape = {2,3,3,4};
	vector<ActivationFunction> activationFunctions = {none,ReLU,ReLU,softmax};
	vector<InitFunction> weightInitFunctions = {zeros,heInitialization,heInitialization,heInitialization};
	vector<InitFunction> biasInitFunctions = {zeros,zeros,zeros,zeros};

	NeuralNetwork net(shape,activationFunctions,weightInitFunctions,biasInitFunctions, CategoricalCrossEntropy);

	MatrixXd inputs(2,3);
	inputs.setRandom();

	cout << inputs << endl;

	MatrixXd y_true(3,1);
	y_true(0,0) = 0;
	y_true(1,0) = 1;
	y_true(2,0) = 2;

	cout << "y to be passed: \n" << y_true << endl;

	MathUtils::convertToOneHotEncoded(y_true,4);

	cout << "y in one hot encoded form: \n" << y_true << endl;

	net.forward(inputs,y_true);

	cout << "mean loss: " << net.meanLoss << endl;

	cout << "******************" << endl;

	net.backward();

	cout << "layer 3 error: \n" << net.layers.at(2).error << endl;
	return 0;
}