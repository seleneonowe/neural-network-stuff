#include "InitFunctions.hpp"
#include <random>

void InitFunctions::HeInitialize(MatrixXd &inputs)
{
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0, sqrt(2.0 / (inputs.cols())));

	for(int j = 0; j < inputs.cols(); j++) {
		for(int i=0; i < inputs.rows(); i++) {
			inputs(i,j) = distribution(generator);
		}
	}
}