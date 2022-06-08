#include "DataSetGenerator.hpp"
#include "MathUtils.hpp"
#include "pyrange.hpp"
#include <random>

tuple<MatrixXd, VectorXd> DataSetGenerator::createSpiralData(unsigned int samples, unsigned int classes)
{
	std::random_device rd{};
	std::mt19937 mt{rd()};
	std::normal_distribution<> dist{};

	// X contains all data samples for all classes. Data samples are a pair of doubles.
	MatrixXd X(2, samples * classes);
	X.setZero();

	// y indexes the data in X by class.
	VectorXd y(samples * classes);
	y.setZero();

	for (int class_number : pyrange(classes)) // MathUtils::range(1,classes))
	{
		// radii of points
		VectorXd r = std::get<0>(MathUtils::linspace(0, 1, samples));

		// angle of points
		VectorXd t = std::get<0>(MathUtils::linspace(class_number * 4, (class_number + 1) * 4, samples));

		for (int ix : pyrange(samples * class_number, samples * (class_number + 1)))
		{
			// so as not to overrun:
			int iw = ix - samples * class_number;

			// add random noise to angle
			t(iw) += 0.2 * dist(mt);

			// calculate points
			X(0, ix) = r(iw) * sin(t(iw) * 2.5);
			X(1, ix) = r(iw) * cos(t(iw) * 2.5);

			// index this data point
			y(ix) = class_number;
		}
	}

	return tuple<MatrixXd, VectorXd>(X, y);
}

tuple<MatrixXd, VectorXd> DataSetGenerator::createSinusoidalData(unsigned samples, unsigned classes)
{
	std::random_device rd{};
	std::mt19937 mt{rd()};
	std::normal_distribution<> dist{};

	// X contains all data samples for all classes. Data samples are a pair of doubles.
	MatrixXd X(2, samples * classes);
	X.setZero();

	// y indexes the data in X by class.
	VectorXd y(samples * classes);
	y.setZero();

	for (int class_number : pyrange(classes))
	{
		for (int ix : pyrange(samples * class_number, samples * (class_number + 1)))
		{
			// calculate points
			X(0, ix) = 1*dist(mt);
			X(1, ix) = sin(X(0,ix)+class_number);

			// index this data point
			y(ix) = class_number;
		}
	}

	return tuple<MatrixXd, VectorXd>(X, y);
}

tuple<MatrixXd,VectorXd> DataSetGenerator::createBlockData(unsigned int samples, unsigned int classes) {
	std::random_device rd{};
	std::mt19937 mt{rd()};
	std::normal_distribution<> dist{};

	// X contains all data samples for all classes. Data samples are a pair of doubles.
	MatrixXd X(2, samples * classes);
	X.setZero();

	// y indexes the data in X by class.
	VectorXd y(samples * classes);
	y.setZero();

	for (int class_number : pyrange(classes))
	{
		for (int ix : pyrange(samples * class_number, samples * (class_number + 1)))
		{
			// calculate points
			X(0, ix) = dist(mt) + class_number*4;
			X(1, ix) = dist(mt);

			// index this data point
			y(ix) = class_number;
		}
	}

	return tuple<MatrixXd, VectorXd>(X, y);
}