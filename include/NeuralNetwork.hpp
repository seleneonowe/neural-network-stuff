#pragma once

#include <vector>
#include <Eigen/Dense>
#include "DenseLayer.hpp"

using Eigen::MatrixXd;
using std::vector;

class NeuralNetwork
{
public:
	NeuralNetwork(const vector<unsigned> shape, const vector<ActivationFunction> activationFunctions, const vector<InitFunction> initFunction);
	void forward(const MatrixXd inputBatch, const MatrixXd y);
	void backward();

	const unsigned numLayers;
	vector<DenseLayer> layers;
	const vector<ActivationFunction> activationFunctions;
	const vector<InitFunction> initFunctions;

	double loss;
};