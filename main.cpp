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

/*
Main to-do list
TODO: make program ask for the below options for parameters, dataset, etc at runtime so I don't have to recompile to do something different.
TODO: add momentum, to help with getting caught in local minima
TODO: add some more functions that are also commonly used, like tanh and sigmoid
TODO: implement learning rate decay?
TODO: do training in multiple threads
TODO: add ability to save/load network state to/from file
TODO: add ability to save/load datasets to/from file
*/
int main()
{
	// seed rng
	srand(time(0));

	// number of classes our procedurally generated data will have. = number of output neurons in the network.
	unsigned numberOfClasses = 5;

	// shape our network will have (number of neurons in each layer)
	vector<unsigned> shape = {2, 8, 8, numberOfClasses};

	// activation and initialization functions our network will have for each layer.
	vector<ActivationFunction> activationFunctions = {none, ReLU, ReLU, softmax};
	vector<InitFunction> weightInitFunctions = {zeros, heInitialization, heInitialization, heInitialization};
	vector<InitFunction> biasInitFunctions = {zeros, zeros, zeros, zeros};

	// create our neural network object, "net".
	NeuralNetwork net(shape, activationFunctions, weightInitFunctions, biasInitFunctions, CategoricalCrossEntropy);

	// procedurally generate a dataset to feed this network.
	auto dataSet = DataSetGenerator::createBlockData(10000, numberOfClasses);

	// number of mini-batches to split this dataset up into
	unsigned numberOfBatches = 50;

	cout << "creating random batches..." << endl;
	auto batchedData = MathUtils::createRandomBatchesFromDataSet(dataSet, numberOfBatches);

	// just print some info about the dataset
	cout << "data set matrix size: rows: " << get<0>(dataSet).rows() << " cols: " << get<0>(dataSet).cols() << endl;
	cout << "data set vector size: rows: " << get<1>(dataSet).rows() << " cols: " << get<1>(dataSet).cols() << endl;

	// get the expected output for the data set in one hot encoded form.
	MatrixXd y_true = get<1>(dataSet);
	MathUtils::convertToOneHotEncoded(y_true, numberOfClasses);

	// just print the size of this expected output.
	cout << "y_true size: rows: " << y_true.rows() << " cols: " << y_true.cols() << endl;

	// we do a forward pass to check the mean loss before training
	net.forward(get<0>(dataSet), y_true);

	cout << "******************" << endl;

	double learningRate = 0.005;
	unsigned epochs = 100;

	net.backward(learningRate);

	cout << "******************" << endl;
	cout << "mean loss: " << net.getMeanLoss() << endl;
	cout << "******************" << endl;

	// training: we loop over the batches, except the last one, which we save to test the network on non-training data afterwards.
	for (unsigned i = 0; i < numberOfBatches - 1; i++)
	{

		MatrixXd y = get<1>(batchedData.at(i));
		MathUtils::convertToOneHotEncoded(y, numberOfClasses);
		// for each batch we repeatedly forward and backward pass <epochs> times
		for (unsigned j = 0; j < epochs; j++)
		{
			net.forward(get<0>(batchedData.at(i)), y);
			net.backward(learningRate);
		}

		// just a progress counter
		if (i % (int)(numberOfBatches / 10) == 0)
		{
			cout << "batch " << i << " processed" << endl;
		}
	}

	cout << "******************" << endl;
	cout << "mean loss: " << net.getMeanLoss() << endl;
	cout << "******************" << endl;

	// we kept a single batch not as part of the training data so we can see how well it classifies this
	MatrixXd y = get<1>(batchedData.at(numberOfBatches - 1));
	MathUtils::convertToOneHotEncoded(y, numberOfClasses);
	net.forward(get<0>(batchedData.at(numberOfBatches - 1)), y);

	cout << "******************" << endl;
	cout << "mean loss: " << net.getMeanLoss() << endl;
	cout << "******************" << endl;

	return 0;
}