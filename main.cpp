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


#include "NeuralNetwork.hpp"
#include "MathUtils.hpp"
#include "DataSetGenerator.hpp"
#include <iostream>

using namespace std;

int main()
{
	// seed rng
	srand(time(0));

	unsigned numberOfClasses = 5;

	vector<unsigned> shape = {2, 8, 8, numberOfClasses};
	vector<ActivationFunction> activationFunctions = {none, ReLU, ReLU, softmax};
	vector<InitFunction> weightInitFunctions = {zeros, heInitialization, heInitialization, heInitialization};
	vector<InitFunction> biasInitFunctions = {zeros, zeros, zeros, zeros};

	NeuralNetwork net(shape, activationFunctions, weightInitFunctions, biasInitFunctions, CategoricalCrossEntropy);

	auto dataSet = DataSetGenerator::createBlockData(10000, numberOfClasses);

	unsigned numberOfBatches = 50;
	cout << "creating random batches..." << endl;
	auto batchedData = MathUtils::createRandomBatchesFromDataSet(dataSet, numberOfBatches);

	// int batchSize = 4;
	// MatrixXd inputs(2, batchSize);
	// inputs.setRandom();

	// MatrixXd y_true(batchSize, 1);

	// for (int i = 0; i < y_true.rows(); i++) {
	// 	y_true(i,0) = (int) rand() % 3;
	// }

	// MathUtils::convertToOneHotEncoded(y_true, 4);
	cout << "data set matrix size: rows: " << get<0>(dataSet).rows() << " cols: " << get<0>(dataSet).cols() << endl;
	cout << "data set vector size: rows: " << get<1>(dataSet).rows() << " cols: " << get<1>(dataSet).cols() << endl;
	MatrixXd y_true = get<1>(dataSet);
	MathUtils::convertToOneHotEncoded(y_true, numberOfClasses);
	cout << "y_true size: rows: " << y_true.rows() << " cols: " << y_true.cols() << endl;

	net.forward(get<0>(dataSet), y_true);

	cout << "******************" << endl;

	double learningRate = 0.005;
	unsigned epochs = 100;

	net.backward(learningRate);

	cout << "******************" << endl;
	cout << "mean loss: " << net.meanLoss << endl;
	cout << "******************" << endl;

	for (unsigned i = 0; i < numberOfBatches - 1; i++)
	{
		for (unsigned j = 0; j < epochs; j++)
		{
			MatrixXd y = get<1>(batchedData.at(i));
			MathUtils::convertToOneHotEncoded(y, numberOfClasses);
			net.forward(get<0>(batchedData.at(i)), y);
			net.backward(learningRate);
		}
		if (i % (int)(numberOfBatches / 10) == 0)
		{
			cout << "batch " << i << " processed" << endl;
		}
	}

	cout << "******************" << endl;
	cout << "mean loss: " << net.meanLoss << endl;
	cout << "******************" << endl;

	// we kept a single batch not as part of the training data so we can see how well it classifies this
	MatrixXd y = get<1>(batchedData.at(numberOfBatches - 1));
	MathUtils::convertToOneHotEncoded(y, numberOfClasses);
	net.forward(get<0>(batchedData.at(numberOfBatches - 1)), y);

	cout << "******************" << endl;
	cout << "mean loss: " << net.meanLoss << endl;
	cout << "******************" << endl;

	return 0;
}