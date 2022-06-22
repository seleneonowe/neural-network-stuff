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
#include <tuple>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::tuple;

/*
	Class used as a library of functions that can procedurally generate datasets, stored as static methods.
	May be expanded.
	The created data sets are intended to be used as example datasets to be passed to a neural network, which
	should try to classify the data correctly.
*/
class DataSetGenerator {
	public:
		// makes <classes> sets of spiral-shaped data with <samples> samples.
		static tuple<MatrixXd,VectorXd> createSpiralData(unsigned int samples, unsigned int classes);

		// makes <classes> sets of sinusoidal-shaped data with <samples> samples.
		static tuple<MatrixXd,VectorXd> createSinusoidalData(unsigned int samples, unsigned int classes);

		// makes <classes> sets of block-shaped data with <samples> samples.
		static tuple<MatrixXd,VectorXd> createBlockData(unsigned int samples, unsigned int classes);
};