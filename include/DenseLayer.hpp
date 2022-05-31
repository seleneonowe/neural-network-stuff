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
	DenseLayer(unsigned layerNum, unsigned previousLayerSize, unsigned mySize, LayerType, ActivationFunction, InitFunction weightInitFunction, InitFunction biasInitFunction);

	void forward(const MatrixXd inputs);
	void backward();

	void initializeWeights();
	void initializeBiases();

	// applies activationFunction to outputBeforeActivation and sets output equal to the result.
	void applyActivationFunction();

	// sets batch size, then sets the biases matrix columns equal to the batch size, then calls fixBiasesMatrix()
	void setBatchSizeAndResizeBiasesMatrix(unsigned size);

	unsigned layerNum;
	unsigned previousLayerSize;
	unsigned mySize;
	LayerType type;
	ActivationFunction activationFunction;
	InitFunction weightInitFunction;
	InitFunction biasInitFunction;

	MatrixXd weights;
	VectorXd biases;

	// has number of columns equal to batch size
	MatrixXd biasesMatrix;
	MatrixXd output;
	MatrixXd outputBeforeActivation;

private:
	unsigned batchSize;

	// sets each column of the biases matrix equal to the biases vector
	void fixBiasesMatrix();
};