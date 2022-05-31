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

	NeuralNetwork net(shape,activationFunctions,weightInitFunctions,biasInitFunctions);

	return 0;
}