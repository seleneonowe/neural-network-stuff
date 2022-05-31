#pragma once

#include <Eigen/Dense>
#include "LayerType.hpp"
#include "ActivationFunction.hpp"
#include "InitFunction.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class DenseLayer
{
public:
	DenseLayer(unsigned layerNum, unsigned previousLayerSize, unsigned mySize, LayerType, ActivationFunction, InitFunction);

	void forward(const MatrixXd inputs);
	void backward();

	void initialize();
	void applyActivationFunction();

	unsigned layerNum;
	unsigned previousLayerSize;
	unsigned mySize;
	LayerType type;
	ActivationFunction activationFunction;
	InitFunction initFunction;

	// has rows = previous layer neuron number + 1 and cols = this layer neuron number + 1
	MatrixXd weightsAndBiases;
	MatrixXd output;
	MatrixXd outputBeforeActivation;
};