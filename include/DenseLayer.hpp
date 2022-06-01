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
	void backward(const MatrixXd &errorInNextLayer, const MatrixXd &weightsInNextLayer);

	void initializeWeights();
	void initializeBiases();

	// Applies activationFunction to outputBeforeActivation and sets output equal to the result.
	void applyActivationFunction();

	// Applies the derivative of activationFunction to outputBeforeActivation and sets gradOutputBeforeActivation equal to the result.
	void applyDActivationFunction();

	// Sets batch size, then sets the biases matrix columns equal to the batch size, then calls fixBiasesMatrix()
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

	MatrixXd inputs;

	// has number of columns equal to batch size
	MatrixXd biasesMatrix;
	MatrixXd output;
	MatrixXd outputBeforeActivation;

	// note error happens to be the same as the gradient of the biases. rows = number of neurons; cols = batchSize.
	MatrixXd error;
	MatrixXd gradWeights;
	VectorXd gradBiases;
	MatrixXd gradOutputBeforeActivation;

private:
	unsigned batchSize;

	// sets each column of the biases matrix equal to the biases vector
	void fixBiasesMatrix();
};