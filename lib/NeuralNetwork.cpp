#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork(const vector<unsigned> shape, const vector<ActivationFunction> activationFunctions, const vector<InitFunction> weightInitFunctions, const vector<InitFunction> biasInitFunctions)
	: numLayers(shape.size()), activationFunctions(activationFunctions), weightInitFunctions(weightInitFunctions), biasInitFunctions(biasInitFunctions)
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