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

class NeuralNetwork
{
public:
	NeuralNetwork(const vector<unsigned> shape, const vector<ActivationFunction> activationFunctions, const vector<InitFunction> weightInitFunctions, const vector<InitFunction> biasInitFunctions, const LossFunction lossFunction);
	void forward(const MatrixXd &inputBatch, const MatrixXd &y);
	void backward(double &learningRate);

	const double& getMeanLoss();

private:
	void calculateLoss();

	void computeGradiantOfLossWRTOutput();

	const unsigned numLayers;
	vector<DenseLayer> layers;
	const vector<ActivationFunction> activationFunctions;
	const vector<InitFunction> weightInitFunctions;
	const vector<InitFunction> biasInitFunctions;
	const LossFunction lossFunction;

	// has size rows = number of output neurons in final layer and columns = batchSize
	MatrixXd outputs;

	// aka y
	MatrixXd expectedOutputs;
	RowVectorXd losses;
	double meanLoss;

	MatrixXd gradiantOfLossWRTOutput;

	unsigned batchSize;
};