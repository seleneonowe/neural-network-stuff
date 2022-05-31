#include "DenseLayer.hpp"
#include "InitFunctions.hpp"
#include "ActivationFunctions.hpp"

// Constructor
DenseLayer::DenseLayer(unsigned layerNum, unsigned previousLayerSize, unsigned mySize, LayerType type, ActivationFunction activationFunction, InitFunction weightInitFunction, InitFunction biasInitFunction)
	: layerNum(layerNum), previousLayerSize(previousLayerSize), mySize(mySize), type(type), activationFunction(activationFunction), weightInitFunction(weightInitFunction), biasInitFunction(biasInitFunction)
{
	weights.resize(previousLayerSize, mySize);
	biases.resize(mySize);
	biasesMatrix.resize(mySize, 1);
	output.resize(mySize, 1);
	outputBeforeActivation.resize(mySize, 1);
	initializeWeights();
	initializeBiases();
}

void DenseLayer::initializeWeights()
{
	// allow for different initialization functions to be used in future
	switch (weightInitFunction)
	{
	case heInitialization:
		InitFunctions::HeInitialize(weights);
		break;
	case zeros:
		weights.setZero();
		break;
	}
}

void DenseLayer::initializeBiases() {
	switch (biasInitFunction) {
		case heInitialization:
			InitFunctions::HeInitialize(biases);
			break;
		case zeros:
			biases.setZero();
			break;
	}
}

void DenseLayer::forward(const MatrixXd inputs)
{
	if (type == input)
	{
		output = inputs;
	}
	else
	{
		if (batchSize != inputs.cols())
		{
			setBatchSizeAndResizeBiasesMatrix(inputs.cols());
		}
		outputBeforeActivation = weights * inputs + biasesMatrix;
		applyActivationFunction();
	}
}

void DenseLayer::applyActivationFunction()
{

	// copy output before activation to output
	output = outputBeforeActivation;

	switch (activationFunction)
	{
	case ReLU:

		// apply ReLU
		ActivationFunctions::forward_ReLU(output);
		break;

	case softmax:

		// apply softmax
		ActivationFunctions::forward_softmax(output);
		break;

	case none:
		break;
	}
}

void DenseLayer::setBatchSizeAndResizeBiasesMatrix(unsigned size)
{
	batchSize = size;
	biasesMatrix.resize(mySize, batchSize);
	fixBiasesMatrix();
}

void DenseLayer::fixBiasesMatrix()
{
	for (int j = 0; j < biasesMatrix.cols(); j++)
	{
		for (int i = 0; i < biasesMatrix.rows(); i++)
		{
			biasesMatrix(i, j) = biases(i);
		}
	}
}