#include "MathUtils.hpp"

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

bool MathUtils::isShapeEqual(MatrixXd &matrix1, MatrixXd &matrix2) {
	return (matrix1.cols()==matrix2.cols() && matrix1.rows() == matrix2.rows());
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