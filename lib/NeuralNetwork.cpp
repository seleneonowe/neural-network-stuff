#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork(const vector<unsigned> shape, const vector<ActivationFunction> activationFunctions, const vector<InitFunction> initFunctions)
	: numLayers(shape.size()), activationFunctions(activationFunctions), initFunctions(initFunctions)
{

	// create layers
	for (unsigned i = 0; i < numLayers; i++)
	{
		LayerType type;
		unsigned previousLayerSize;
		// add 1 to sizes because biases are part of weight matrices in this formulation.
		if (i == 0)
		{
			type = input;
			previousLayerSize = 0;
		}
		else if (i + 1 == numLayers)
		{
			type = output;
			previousLayerSize = shape.at(i - 1) + 1;
		}
		else
		{
			type = hidden;
			previousLayerSize = shape.at(i - 1) + 1;
		}

		DenseLayer layer(i, previousLayerSize, shape.at(i) + 1, type, activationFunctions.at(i), initFunctions.at(i));

		layers.insert(layers.begin() + i, layer);
	}
}