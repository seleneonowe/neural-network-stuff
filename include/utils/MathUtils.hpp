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

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

/*
	Class containing a library of useful math functions, stored as static methods.
	Done this way for convenience but it ought to be split up if it becomes much larger.
*/
class MathUtils
{
public:
	// clips each value in the matrix between min and max.
	static void clip(MatrixXd &inputs, const double &min, const double &max);

	// clips input between min and max
	static double clip(double input, double min, double max);

	// checks if two matrices have the same shape
	static bool isShapeEqual(MatrixXd &matrix1, MatrixXd &matrix2);

	// converts class label vector to one hot encoded form.
	static void convertToOneHotEncoded(MatrixXd &inputs, unsigned);

	// TODO: Make this not terrible because it's way too slow rn
	// takes a data set and splits it into random batches.
	static std::vector<std::tuple<MatrixXd, VectorXd>> createRandomBatchesFromDataSet(std::tuple<MatrixXd, VectorXd> &dataSet, unsigned numberOfBatches);

	// applies func to each entry of the input matrix.
	static void applyToAll(MatrixXd &inputs, double (*func)(double));

	static void applyToAll(MatrixXd &inputs, double, double, double (*func)(double, double, double));

	// very basic numpy.linspace like implementation
	static std::tuple<VectorXd, double> linspace(double start, double stop, unsigned int num = 10, bool endpoint = false);
};