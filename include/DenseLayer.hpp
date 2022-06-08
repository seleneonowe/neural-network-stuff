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
	DenseLayer(unsigned layerNum, unsigned previousLayerSize, unsigned mySize, LayerType, ActivationFunction, InitFunction weightInitFunction, InitFunction biasInitFunction);

	void forward(const MatrixXd &inputs);
	void backward(const MatrixXd &errorInNextLayer, const MatrixXd &weightsInNextLayer, double &learningRate);

	void updateWeightsAndBiases(double &learningRate);

	const MatrixXd& getOutput();
	const MatrixXd& getWeights();
	const MatrixXd& getError();

private:
	void initializeWeights();
	void initializeBiases();

	// Applies activationFunction to outputBeforeActivation and sets output equal to the result.
	void applyActivationFunction();

	// Applies the derivative of activationFunction to outputBeforeActivation and sets gradOutputBeforeActivation equal to the result.
	void applyDActivationFunction();


	// Sets batch size, then sets the biases matrix columns equal to the batch size, then calls fixBiasesMatrix()
	void setBatchSizeAndResizeBiasesMatrix(unsigned size);

	// sets each column of the biases matrix equal to the biases vector
	void fixBiasesMatrix();

	unsigned layerNum;
	unsigned previousLayerSize;
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