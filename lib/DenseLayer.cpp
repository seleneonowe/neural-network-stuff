/*
 *   Copyright (c) 2022 seleneonowe
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *   
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *   
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */


#include "DenseLayer.hpp"
#include "InitFunctions.hpp"
#include "ActivationFunctions.hpp"
#include "MathUtils.hpp"

// Constructor
DenseLayer::DenseLayer(unsigned layerNum, unsigned previousLayerSize, unsigned mySize, LayerType type, ActivationFunction activationFunction, InitFunction weightInitFunction, InitFunction biasInitFunction)
	: layerNum(layerNum), previousLayerSize(previousLayerSize), mySize(mySize), type(type), activationFunction(activationFunction), weightInitFunction(weightInitFunction), biasInitFunction(biasInitFunction)
{
	weights.resize(mySize, previousLayerSize);
	biases.resize(mySize);
	biasesMatrix.resize(mySize, 1);
	output.resize(mySize, 1);
	outputBeforeActivation.resize(mySize, 1);
	gradOutputBeforeActivation.resize(mySize, 1);
	batchSize = 0;
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

void DenseLayer::initializeBiases()
{
	switch (biasInitFunction)
	{
	case heInitialization:
		InitFunctions::HeInitialize(biases);
		break;
	case zeros:
		biases.setZero();
		break;
	}
}

void DenseLayer::forward(const MatrixXd &inputs)
{
	this->inputs = inputs;

	if (type == input)
	{
		output = inputs;
	}
	else // hidden or output layer
	{
		if (batchSize != inputs.cols())
		{
			setBatchSizeAndResizeBiasesMatrix(inputs.cols());
		}

		outputBeforeActivation = weights * inputs + biasesMatrix;
		applyActivationFunction();
	}
}

void DenseLayer::backward(const MatrixXd &errorInNextLayer, const MatrixXd &weightsInNextLayer, double &learningRate)
{

	if (type == LayerType::output)
	{ // CALCULATE THE OUTPUT ERROR IN NEURAL NETWORK CLASS THEN PASS IT HERE
		// there is no previous layer here so errorInNextLayer is the gradient of the cost function wrt the output of the network
		// the error in the final layer depends only on grad(z) and grad(cost)wrt a
		
		error = errorInNextLayer;
	}
	else if (type == LayerType::hidden)
	{
		applyDActivationFunction();

		error = (weightsInNextLayer.transpose() * errorInNextLayer).cwiseProduct(gradOutputBeforeActivation);
	}
	
	gradWeights = error * inputs.transpose() / batchSize;

	gradBiases.resize(error.rows());
	for (int i = 0; i < error.rows(); i++) {
		double sum = 0;
		for (int j = 0; j < error.cols(); j++) {
			sum+= error(i,j);
		}
		gradBiases(i) = sum / error.cols();
	}

	updateWeightsAndBiases(learningRate);
}

void DenseLayer::updateWeightsAndBiases(double &learningRate) {
	weights-= learningRate * gradWeights;
	biases -= learningRate * gradBiases;
	fixBiasesMatrix();
}

void DenseLayer::applyDActivationFunction()
{
	// copy z to grad(z) before taking grad
	gradOutputBeforeActivation = outputBeforeActivation;

	// take gradient
	switch (activationFunction)
	{
	case ReLU:
		ActivationFunctions::backward_ReLU(gradOutputBeforeActivation);
		break;
	// // TODO: write backwards softmax function
	// case softmax:
	// 	ActivationFunctions::backward_softmax(gradOutputBeforeActivation);
	// 	break;
	case none:
		// derivative of g(z)=z is 1
		gradOutputBeforeActivation.setConstant(1);
		break;
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

const MatrixXd& DenseLayer::getError() {
	return error;
}
const MatrixXd& DenseLayer::getWeights() {
	return weights;
}
const MatrixXd& DenseLayer::getOutput() {
	return output;
}

