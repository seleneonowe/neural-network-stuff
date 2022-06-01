#include "NeuralNetwork.hpp"
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
	cout << "forwarded first layer, output: \n" << layers.at(0).output << endl;
	for (long unsigned i = 1; i < layers.size(); i++)
	{
		cout << "forwarding layer: " << i << endl;
		layers.at(i).forward(layers.at(i - 1).output);
		cout << "forwarded layer " << i << " output: \n" << layers.at(i).output << endl;
	}

	outputs = layers.at(layers.size() - 1).output;
	cout << "calculating loss" << endl;
	calculateLoss(y);
}

// if using CCE, require y to be an array of one-hot encoded vectors, or a vector of integer values.
void NeuralNetwork::calculateLoss(const MatrixXd &y_true)
{
	losses.resize(batchSize);
	switch (lossFunction)
	{
	case CategoricalCrossEntropy:
		// check if we have been passed y in the form of an array of one hot encoded vectors
		if (y_true.rows() == outputs.cols() && y_true.cols() == outputs.rows())
		{
			cout << "one-hot encoded passed" << endl;
			cout << "outputs = \n" << outputs << "\n y_true: \n" << y_true << endl;
			MatrixXd temp = y_true * outputs;
			cout << "temp : \n" << temp << endl;
			for (int j = 0; j < temp.cols(); j++)
			{
				double sum = 0;
				for (int i = 0; i < temp.rows(); i++)
				{
					sum += temp(i, j);
				}
				losses(j) = sum;
			}
		}
		// check if we have been passed y in the form of a row or column vector of integer values
		else if ((y_true.rows() == batchSize && y_true.cols() == 1) || (y_true.rows() == 1 && y_true.cols() == batchSize))
		{
			cout << "classifier passed" << endl;
			if (y_true.rows() == batchSize)
			{
				for (unsigned i = 0; i < batchSize; i++)
				{
					losses(i) = outputs((int) y_true(i,0),i); 
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