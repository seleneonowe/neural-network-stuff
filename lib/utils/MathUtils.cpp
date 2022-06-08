#include "MathUtils.hpp"
#include "pyrange.hpp"
#include <tuple>
#include <bits/stdc++.h>
#include <iostream>

void MathUtils::clip(MatrixXd &inputs, const double &min, const double &max)
{
	applyToAll(inputs, min, max, &clip);
}

double MathUtils::clip(double input, double min, double max)
{
	if (input < min)
	{
		return min;
	}
	else if (input > max)
	{
		return max;
	}
	else
	{
		return input;
	}
}

bool MathUtils::isShapeEqual(MatrixXd &matrix1, MatrixXd &matrix2)
{
	return (matrix1.cols() == matrix2.cols() && matrix1.rows() == matrix2.rows());
}

void MathUtils::applyToAll(MatrixXd &inputs, double (*func)(double))
{
	for (unsigned j = 0; j < inputs.cols(); j++)
	{
		for (unsigned i = 0; i < inputs.rows(); i++)
		{
			inputs(i, j) = func(inputs(i, j));
		}
	}
}
void MathUtils::applyToAll(MatrixXd &inputs, double arg1, double arg2, double (*func)(double, double, double))
{
	for (unsigned j = 0; j < inputs.cols(); j++)
	{
		for (unsigned i = 0; i < inputs.rows(); i++)
		{
			inputs(i, j) = func(inputs(i, j), arg1, arg2);
		}
	}
}

void MathUtils::convertToOneHotEncoded(MatrixXd &inputs, unsigned numberOfClasses)
{
	if (inputs.cols() == 1)
	{
		MatrixXd output;
		output.resize(inputs.rows(), numberOfClasses);
		output.setZero();
		for (int i = 0; i < inputs.rows(); i++)
		{
			output(i, (int)inputs(i, 0)) = 1.0;
		}
		inputs = output;
	}
	else if (inputs.rows() == 1)
	{
		MatrixXd output;
		output.resize(inputs.rows(), numberOfClasses);
		output.setZero();
		for (int i = 0; i < inputs.cols(); i++)
		{
			output(i, (int)inputs(0, i)) = 1.0;
		}
		inputs = output;
	}
	else
	{
		throw std::invalid_argument("must pass a class identifier matrix (row or column vector) to be able to convert to one-hot encoding");
	}
}

std::tuple<VectorXd, double> MathUtils::linspace(double start, double stop, unsigned int num, bool endpoint)
{
	double stepSize;

	if (endpoint)
	{
		stepSize = (stop - start) / num;
	}
	else
	{
		stepSize = (stop - start) / (num + 1);
	}

	VectorXd outVector(num);

	for (unsigned int i = 0; i < num; i++)
	{
		outVector(i) = start + i * stepSize;
	}

	return std::tuple<VectorXd, double>(outVector, stepSize);
}

std::vector<std::tuple<MatrixXd, VectorXd>> MathUtils::createRandomBatchesFromDataSet(std::tuple<MatrixXd, VectorXd> &dataSet, unsigned numberOfBatches)
{
	unsigned dataSetSize = std::get<1>(dataSet).size();
	std::unordered_set<unsigned> unusedElements;

	for (int i : pyrange(dataSetSize))
	{
		unusedElements.insert(i);
	}

	std::unordered_set<unsigned>::iterator itr;

	std::vector<std::tuple<MatrixXd, VectorXd>> output;
	output.resize(numberOfBatches);

	for (unsigned i = 0; i < numberOfBatches; i++)
	{
		unsigned j = 0;
		while (j < dataSetSize / numberOfBatches && unusedElements.size() != 0)
		{
			int elemNumber = rand() % unusedElements.size();
			itr = unusedElements.begin();
			for (int i = 0; i < elemNumber; i++)
			{
				itr++;
			}

			// get matrix at i
			int currentRows = std::get<0>(output.at(i)).rows();
			int currentCols = std::get<0>(output.at(i)).cols();

			if (currentRows == 0)
			{
				std::get<0>(output.at(i)).resize(std::get<0>(dataSet).rows(), currentCols + 1);
				std::get<0>(output.at(i)) = std::get<0>(dataSet).col(*itr);

				// vector at i
				std::get<1>(output.at(i)).resize(currentCols + 1);
				// stack vertically
				std::get<1>(output.at(i)) = std::get<1>(dataSet).row(*itr);
			}
			else
			{
				MatrixXd temp(currentRows, currentCols + 1);
				// stack horizontally
				temp << std::get<0>(output.at(i)), std::get<0>(dataSet).col(*itr);

				std::get<0>(output.at(i)) = temp;
				// vector at i
				VectorXd temp2(currentCols + 1);
				// stack vertically
				temp2 << std::get<1>(output.at(i)), std::get<1>(dataSet).row(*itr);
				std::get<1>(output.at(i)) = temp2;
			}

			unusedElements.erase(itr);
			j++;
		}
	}
	return output;
}
