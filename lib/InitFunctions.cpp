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


#include "InitFunctions.hpp"
#include <random>

void InitFunctions::HeInitialize(MatrixXd &inputs)
{
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0, sqrt(2.0 / (inputs.cols())));

	for (int j = 0; j < inputs.cols(); j++)
	{
		for (int i = 0; i < inputs.rows(); i++)
		{
			inputs(i, j) = distribution(generator);
		}
	}
}

void InitFunctions::HeInitialize(VectorXd &inputs)
{
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0, sqrt(2.0));

	for (int i = 0; i < inputs.size(); i++)
	{
		inputs(i) = distribution(generator);
	}
}