#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork(const VectorXi shape, const vector<ActivationFunction> activationFunctions, const InitFunction initFunction)
	: numLayers(shape.size()), activationFunctions(activationFunctions), initFunction(initFunction)
{
	// we have numLayers layers in our network
//	layers.resize(numLayers);

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
			previousLayerSize = shape(i - 1) + 1;
		}
		else
		{
			type = hidden;
			previousLayerSize = shape(i - 1) + 1;
		}

		DenseLayer layer(i, previousLayerSize, shape(i) + 1, type, activationFunctions.at(i), initFunction);

		layers.insert(layers.begin() + i, layer);
	}
}