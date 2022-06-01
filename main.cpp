#include "NeuralNetwork.hpp"
#include <iostream>

using namespace std;

int main() {
	//seed rng
	srand(time(0));

	vector<unsigned> shape = {2,3,3,2};
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
	y_true(2,0) = 1;

	cout << "y to be passed: \n" << y_true << endl;

	net.forward(inputs,y_true);

	cout << "mean loss: " << net.meanLoss << endl;
	return 0;
}