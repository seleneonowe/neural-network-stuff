#pragma once
#include <Eigen/Dense>

using Eigen::MatrixXd;

class MathUtils {
	public:
	static void clip(MatrixXd &inputs, const double &min, const double &max);

	static double clip(double input, double min, double max);

	static bool isShapeEqual(MatrixXd &matrix1, MatrixXd &matrix2);

	static void convertToOneHotEncoded(MatrixXd &inputs, unsigned );
	
	static void applyToAll(MatrixXd &inputs, double (*func)(double));

	static void applyToAll(MatrixXd &inputs, double, double, double (*func)(double,double,double));
};