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
	void backward(double &learningRate);

	void calculateLoss();

	void computeGradiantOfLossWRTOutput();

	const unsigned numLayers;
	vector<DenseLayer> layers;
	const vector<ActivationFunction> activationFunctions;
	const vector<InitFunction> weightInitFunctions;
	const vector<InitFunction> biasInitFunctions;
	const LossFunction lossFunction;

	// has size rows = number of output neurons in final layer and columns = batchSize
	MatrixXd outputs;

	// aka y
	MatrixXd expectedOutputs;
	RowVectorXd losses;
	double meanLoss;

	MatrixXd gradiantOfLossWRTOutput;

	private: 
		unsigned batchSize;
};