#include "DenseLayer.hpp"
#include "InitFunctions.hpp"

// Constructor
DenseLayer::DenseLayer(unsigned layerNum, unsigned previousLayerSize, unsigned mySize, LayerType type, ActivationFunction activationFunction, InitFunction initFunction)
	: layerNum(layerNum)
	, previousLayerSize(previousLayerSize)
	, mySize(mySize)
	, type(type)
	, activationFunction(activationFunction)
	, initFunction(initFunction)
{
	weightsAndBiases.resize(previousLayerSize,mySize);
	output.resize(mySize,1);
	activations.resize(mySize,1);
	initialize();
}

void DenseLayer::initialize() {
	//allow for different initialization functions to be used in future
	switch (initFunction) {
		case heInitialization:
			InitFunctions::HeInitialize(weightsAndBiases);
			break;
	}
}
