#pragma once

#include <vector>
#include <Eigen/Dense>
#include "DenseLayer.hpp"
#include "LossFunction.hpp"

using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using std::vector;

class NeuralNetwork
{
public:
	NeuralNetwork(const vector<unsigned> shape, const vector<ActivationFunction> activationFunctions, const vector<InitFunction> weightInitFunctions, const vector<InitFunction> biasInitFunctions, const LossFunction lossFunction);
	void forward(const MatrixXd &inputBatch, const MatrixXd &y);
	void backward();

	void calculateLoss(const MatrixXd &y);

	const unsigned numLayers;
	vector<DenseLayer> layers;
	const vector<ActivationFunction> activationFunctions;
	const vector<InitFunction> weightInitFunctions;
	const vector<InitFunction> biasInitFunctions;
	const LossFunction lossFunction;

	MatrixXd outputs;
	RowVectorXd losses;
	double meanLoss;

	private: 
		unsigned batchSize;
};