#include "NeuralNetwork.hpp"
#include "MathUtils.hpp"
#include <iostream>

using namespace std;

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
	cout << "starting forward pass" << endl;
	layers.at(0).forward(inputBatch);
	cout << "forwarded first layer, output: \n"
		 << layers.at(0).output << endl;
	for (long unsigned i = 1; i < layers.size(); i++)
	{
		cout << "forwarding layer: " << i << endl;
		layers.at(i).forward(layers.at(i - 1).output);
		cout << "forwarded layer " << i << " output: \n"
			 << layers.at(i).output << endl;
	}

	outputs = layers.at(layers.size() - 1).output;

	expectedOutputs = y.transpose();
	cout << "calculating loss" << endl;
	calculateLoss();
}

void NeuralNetwork::backward()
{
	computeGradiantOfLossWRTOutput();
	// for the output layer, we pass the gradiant of the loss wrt the network output, and it doesn't matter what we pass as the second argument.
	layers.at(numLayers - 1).backward(gradiantOfLossWRTOutput, gradiantOfLossWRTOutput);

	cout << "backward for final layer complete" << endl;

	for (int i = numLayers - 2; i > 0; i--)
	{
		// for each other layer, we feed the error of the previous layer back)
		layers.at(i).backward(layers.at(i + 1).error, layers.at(i + 1).weights);
	}
	cout << "backpropagation of errors complete." << endl;
}

void NeuralNetwork::computeGradiantOfLossWRTOutput()
{
	cout << "computing gradiant of loss wrt output" << endl;
	switch (lossFunction)
	{
	case CategoricalCrossEntropy:
		if (!MathUtils::isShapeEqual(expectedOutputs, outputs))
		{
			cout << "converting to one hot encoded" << endl;
			MathUtils::convertToOneHotEncoded(expectedOutputs, outputs.rows());
		}
		gradiantOfLossWRTOutput = outputs - expectedOutputs;
		cout << "gradiant of loss WRT output computed.." << endl;
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
			cout << "one-hot encoded passed" << endl;
			cout << "outputs = \n"
				 << outputs << "\n expectedOutputs: \n"
				 << expectedOutputs << "\n expectedOutputs transpose: \n"
				 << expectedOutputs.transpose() << endl;
			// we clip so that -log(outputs) never returns infinite
			MathUtils::clip(outputs, 1e-7, 1 - 1e-7);
			cout << "clipped outputs: \n"
				 << outputs << endl;

			// temp will be a matrix of the same shape as outputs (output neuron number * batchsize) but with only the expected output position matches nonzero
			MatrixXd temp = expectedOutputs.cwiseProduct(outputs);
			cout << "temp : \n"
				 << temp << endl;
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
			cout << "classifier passed" << endl;
			// we clip so that -log(outputs) never returns infinite
			MathUtils::clip(outputs, 1e-7, 1 - 1e-7);
			cout << "clipped outputs: \n"
				 << outputs << endl;

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
					losses(i) = -log(outputs((int)expectedOutputs(0, i), i));
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