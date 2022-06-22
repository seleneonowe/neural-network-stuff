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


#include "NeuralNetwork.hpp"
#include "MathUtils.hpp"
#include <iostream>

NeuralNetwork::NeuralNetwork(const vector<unsigned> shape, const vector<ActivationFunction> activationFunctions, const vector<InitFunction> weightInitFunctions, const vector<InitFunction> biasInitFunctions, const LossFunction lossFunction)
	: numLayers(shape.size()), activationFunctions(activationFunctions), weightInitFunctions(weightInitFunctions), biasInitFunctions(biasInitFunctions), lossFunction(lossFunction)
{

	// create layers
	for (unsigned i = 0; i < numLayers; i++)
	{
		LayerType type;
		unsigned previousLayerSize;

		if (i == 0)
		{
			type = input;
			previousLayerSize = 0;
		}
		else if (i + 1 == numLayers)
		{
			type = output;
			previousLayerSize = shape.at(i - 1);
		}
		else
		{
			type = hidden;
			previousLayerSize = shape.at(i - 1);
		}

		DenseLayer layer(i, previousLayerSize, shape.at(i), type, activationFunctions.at(i), weightInitFunctions.at(i), biasInitFunctions.at(i));

		layers.insert(layers.begin() + i, layer);
	}
}

void NeuralNetwork::forward(const MatrixXd &inputBatch, const MatrixXd &y)
{
	batchSize = inputBatch.cols();

	// pass the inputs to the first layer
	layers.at(0).forward(inputBatch);

	// pass each layer the output of the previous layer
	for (long unsigned i = 1; i < layers.size(); i++)
	{
		layers.at(i).forward(layers.at(i - 1).getOutput());
	}

	// get the output of the final layer and set the network output to this.
	outputs = layers.at(layers.size() - 1).getOutput();

	expectedOutputs = y.transpose();
	calculateLoss();
}

void NeuralNetwork::backward(double &learningRate)
{
	computeGradiantOfLossWRTOutput();
	// for the output layer, we pass the gradiant of the loss wrt the network output, and it doesn't matter what we pass as the second argument.
	layers.at(numLayers - 1).backward(gradiantOfLossWRTOutput, gradiantOfLossWRTOutput, learningRate);

	for (int i = numLayers - 2; i > 0; i--)
	{
		// for each other layer, we feed the error of the previous layer back)
		layers.at(i).backward(layers.at(i + 1).getError(), layers.at(i + 1).getWeights(), learningRate);
	}
}

void NeuralNetwork::computeGradiantOfLossWRTOutput()
{
	switch (lossFunction)
	{
	case CategoricalCrossEntropy:
		if (!MathUtils::isShapeEqual(expectedOutputs, outputs))
		{
			MathUtils::convertToOneHotEncoded(expectedOutputs, outputs.rows());
		}
		gradiantOfLossWRTOutput = outputs - expectedOutputs;
	}
}

// if using CCE, require y to be an array of one-hot encoded vectors, or a vector of integer values.
void NeuralNetwork::calculateLoss()
{
	losses.resize(batchSize);
	switch (lossFunction)
	{
	case CategoricalCrossEntropy:
		// check if we have been passed y in the form of an array of one hot encoded vectors
		if (expectedOutputs.rows() == outputs.rows() && expectedOutputs.cols() == outputs.cols())
		{
			MatrixXd TESTONLY = outputs;
			// we clip so that -log(outputs) never returns infinite
			MathUtils::clip(outputs, 1e-7, 1 - 1e-7);

			// temp will be a matrix of the same shape as outputs (output neuron number * batchsize) but with only the expected output position matches nonzero
			MatrixXd temp = expectedOutputs.cwiseProduct(outputs);

			for (int j = 0; j < temp.cols(); j++)
			{
				double sum = 0;
				for (int i = 0; i < temp.rows(); i++)
				{
					sum += temp(i, j);
				}

				// CCE takes the negative log of the correct confidences
				losses(j) = -log(sum);
			}
		}
		// check if we have been passed y in the form of a row or column vector of integer values
		else if ((expectedOutputs.rows() == batchSize && expectedOutputs.cols() == 1) || (expectedOutputs.rows() == 1 && expectedOutputs.cols() == batchSize))
		{
			// we clip so that -log(outputs) never returns infinite
			MathUtils::clip(outputs, 1e-7, 1 - 1e-7);

			if (expectedOutputs.rows() == batchSize)
			{
				for (unsigned i = 0; i < batchSize; i++)
				{
					losses(i) = -log(outputs((int)expectedOutputs(i, 0), i));
				}
			}
			else
			{ // do the same thing but switch rows and columns in expected outputs
				for (unsigned i = 0; i < batchSize; i++)
				{
					losses(i) = -log(outputs((int)expectedOutputs(0, i)));
				}
			}
		}
		else
		{
			throw std::invalid_argument("y_true must be a matrix of batchSize*output nodes (one-hot encoded), or a row/column vector of class identifiers of length batchSize");
		}
	}

	meanLoss = losses.mean();
}

const double& NeuralNetwork::getMeanLoss() { 
	return meanLoss;
}