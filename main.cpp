#include "NeuralNetwork.hpp"
#include <iostream>

using namespace std;

int main() {
	//seed rng
	srand(time(0));

	vector<unsigned> shape = {2,3,3,2};
	vector<ActivationFunction> activationFunctions = {none,ReLU,ReLU,softmax};
	vector<InitFunction> initFunctions = {zeros,heInitialization,heInitialization,heInitialization};

	NeuralNetwork net(shape,activationFunctions,initFunctions);

	return 0;
}