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


#pragma once

#include <Eigen/Dense>
#include "LayerType.hpp"
#include "ActivationFunction.hpp"
#include "InitFunction.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/*
	Objects of this class represent a layer of neurons within the network. One can pass inputs forwards
	through the layer to calculate the activations ofthe neurons, and pass information about the next layer backwards
	to calculate the error and therefore the gradiant of the weights and biases.
*/
class DenseLayer
{
public:
	// DenseLayer objects store their position in the network, the number of neurons in their and the previous layers, their type (input, hidden, output), and their activation and initialization functions.
	DenseLayer(unsigned layerNum, unsigned previousLayerSize, unsigned mySize, LayerType, ActivationFunction, InitFunction weightInitFunction, InitFunction biasInitFunction);

	// passes inputs forward through this layer, computing the output for this layer.
	void forward(const MatrixXd &inputs);

	// passes error backwards through this layer, computing the error for this layer.
	void backward(const MatrixXd &errorInNextLayer, const MatrixXd &weightsInNextLayer, double &learningRate);

	// updates weights and biases by subtracting learningRate * their gradient (computed by backward pass)
	void updateWeightsAndBiases(double &learningRate);

	// getters
	const MatrixXd& getOutput();
	const MatrixXd& getWeights();
	const MatrixXd& getError();

private:
	// initializes the weights using the specified initialization function
	void initializeWeights();
	
	// initializes the biases using the specified initialization function
	void initializeBiases();

	// Applies activationFunction to outputBeforeActivation and sets output equal to the result.
	void applyActivationFunction();

	// Applies the derivative of activationFunction to outputBeforeActivation and sets gradOutputBeforeActivation equal to the result.
	void applyDActivationFunction();


	// Sets batch size, then sets the biases matrix columns equal to the batch size, then calls fixBiasesMatrix()
	void setBatchSizeAndResizeBiasesMatrix(unsigned size);

	// sets each column of the biases matrix equal to the biases vector
	void fixBiasesMatrix();

	// this layer's position in the network
	unsigned layerNum;
	// number of neurons in previous layer
	unsigned previousLayerSize;
	// number of neurons in this layer
	unsigned mySize;

	LayerType type;
	ActivationFunction activationFunction;
	InitFunction weightInitFunction;
	InitFunction biasInitFunction;

	MatrixXd weights;
	VectorXd biases;

	MatrixXd inputs;

	// has number of columns equal to batch size
	MatrixXd biasesMatrix;
	MatrixXd output;
	MatrixXd outputBeforeActivation;

	// note error happens to be the same as the gradient of the biases. rows = number of neurons; cols = batchSize.
	MatrixXd error;
	MatrixXd gradWeights;
	VectorXd gradBiases;
	MatrixXd gradOutputBeforeActivation;

	unsigned batchSize;
};