#include "NeuralNetwork.hpp"
#include "MathUtils.hpp"
#include "DataSetGenerator.hpp"
#include <iostream>

using namespace std;

int main()
{
	// seed rng
	srand(time(0));


	unsigned numberOfClasses = 3;

	vector<unsigned> shape = {2, 8, 8, numberOfClasses};
	vector<ActivationFunction> activationFunctions = {none, ReLU, ReLU, softmax};
	vector<InitFunction> weightInitFunctions = {zeros, heInitialization, heInitialization, heInitialization};
	vector<InitFunction> biasInitFunctions = {zeros, zeros, zeros, zeros};

	NeuralNetwork net(shape, activationFunctions, weightInitFunctions, biasInitFunctions, CategoricalCrossEntropy);

	auto spiralDataSet = DataSetGenerator::createSpiralData(10, numberOfClasses);

	unsigned numberOfBatches = 3;
	auto batchedData = MathUtils::createRandomBatchesFromDataSet(spiralDataSet, numberOfBatches);

	// int batchSize = 4;
	// MatrixXd inputs(2, batchSize);
	// inputs.setRandom();

	// MatrixXd y_true(batchSize, 1);

	// for (int i = 0; i < y_true.rows(); i++) {
	// 	y_true(i,0) = (int) rand() % 3;
	// }

	// MathUtils::convertToOneHotEncoded(y_true, 4);
	cout << "spiral data set matrix size: rows: " << get<0>(spiralDataSet).rows() << " cols: " << get<0>(spiralDataSet).cols() << endl;
	cout << "spiral data set vector size: rows: " << get<1>(spiralDataSet).rows() << " cols: " << get<1>(spiralDataSet).cols() << endl;
	MatrixXd y_true = get<1>(spiralDataSet);
	MathUtils::convertToOneHotEncoded(y_true, numberOfClasses);
	cout << "y_true size: rows: " << y_true.rows() << " cols: " << y_true.cols() << endl;

	net.forward(get<0>(spiralDataSet), y_true);

	cout << "******************" << endl;

	double learningRate = 0.01;
	unsigned repeats = 1000;

	net.backward(learningRate);

	cout << "******************" << endl;
	cout << "mean loss: " << net.meanLoss << endl;
	cout << "******************" << endl;

	cout << "-log(1e-7) is " << -log(1e-7) << endl;
	cout << "1-1e-7 is: " << 1-1e-7 << endl;

	for (unsigned i = 0; i < numberOfBatches; i++)
	{
		for (unsigned j = 0; j < repeats; j++)
		{
			MatrixXd y = get<1>(batchedData.at(i));
			MathUtils::convertToOneHotEncoded(y,numberOfClasses);
			net.forward(get<0>(batchedData.at(i)), y);
			net.backward(learningRate);
		}
	}

	cout << "******************" << endl;
	cout << "losses: \n" << net.losses << endl;
	cout << "mean loss: " << net.meanLoss << endl;
	cout << "******************" << endl;

	return 0;
}