#include "NeuralNetwork.hpp"

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

	layers.at(0).forward(inputBatch);

	for (int i = 1; i < layers.size(); i++)
	{
		layers.at(i).forward(layers.at(i - 1).output);
	}

	outputs = layers.at(layers.size() - 1).output;
	calculateLoss(y);
}

// if using CCE, require y to be an array of one-hot encoded vectors, or a vector of integer values.
void NeuralNetwork::calculateLoss(const MatrixXd &y_true)
{
	switch (lossFunction)
	{
	case CategoricalCrossEntropy:
		// check if we have been passed y in the form of an array of one hot encoded vectors
		if (y_true.rows() == outputs.rows() && y_true.cols() == outputs.cols())
		{
			MatrixXd temp = outputs * y_true;
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
			if (y_true.rows() == batchSize)
			{
				for (int i = 0; i < batchSize; i++)
				{
					losses(i) = outputs((int) y_true(i,0),i); 
				}
			}
		}
		else
		{
			throw std::invalid_argument("y_true must be an array of batchSize one-hot encoded vectors of length equal to number of output nodes, or a row/column vector of class identifiers of length batchSize");
		}
	}
}