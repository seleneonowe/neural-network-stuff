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

#include <vector>
#include <Eigen/Dense>
#include "DenseLayer.hpp"
#include "LossFunction.hpp"

using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using std::vector;

/*
	Neural Network objects are arbitrarily shaped (any number of neurons in any number of layers)
	 and densely connected (every neuron connected to every other neuron in the previous layer).
	Different activation functions and weight/bias initialization functions may be specified per-layer.
	Different loss functions may be specified.
*/
class NeuralNetwork
{
public:
	// NeuralNetwork objects must have specified shape (number of neurons in each layer), the activation and weight/bias initialization functions for each layer, and the loss function for the network.
	NeuralNetwork(const vector<unsigned> shape, const vector<ActivationFunction> activationFunctions, const vector<InitFunction> weightInitFunctions, const vector<InitFunction> biasInitFunctions, const LossFunction lossFunction);

	// will pass inputBatch forward through the network, and compare the output to the expected output y to calculate the loss.
	void forward(const MatrixXd &inputBatch, const MatrixXd &y);

	// will perform a backward pass through the network with the given learning rate.
	void backward(double &learningRate);

	// getters
	const double &getMeanLoss();

private:
	void calculateLoss();

	void computeGradiantOfLossWRTOutput();

	// number of layers in the network (input + hidden + output)
	const unsigned numLayers;

	// the layers themselves, stored as a vector of DenseLayer objects
	vector<DenseLayer> layers;

	const vector<ActivationFunction> activationFunctions;
	const vector<InitFunction> weightInitFunctions;
	const vector<InitFunction> biasInitFunctions;
	const LossFunction lossFunction;

	// has size rows = number of output neurons in final layer and columns = batchSize
	MatrixXd outputs;

	// aka y
	MatrixXd expectedOutputs;

	// losses for each batch
	RowVectorXd losses;

	// mean of losses
	double meanLoss;

	MatrixXd gradiantOfLossWRTOutput;

	unsigned batchSize;
};