#ifndef __SELECT_PROPAGATER_H__
#define __SELECT_PROPAGATER_H__

#include <vector>
#include <iostream>

#include <opencv2/opencv.hpp>
#include "nnls.h"

using namespace std;
using namespace cv;

enum ColorSpace {BGR, YCrCb, Lab};

enum SampleMethod {Uniform, Importance, NoSample};

class SelectPropagater {

private:	// private variables

	Size size;

	Mat img;

	Mat selects;

	// extracted select info
	vector<Vec3b> selectedColors;
	vector<double> selectedStrenths;
	vector<Point> selectedPositions;

	// basis samples
	vector<Vec3b> basisColors;
	vector<double> basisStrenths;
	vector<Point> basisPositions;

	// equation samples
	vector<Vec3b> equColors;
	vector<double> equStrenths;
	vector<Point> equPositions;

	// least squares
	Mat A, b;
	Mat a;	// RBF coefficients

private:	// private methods

	void convertColorSpace();

	void extractSelect();

	void sampleBasis();

	void sampleEquation();

	void basisUniformSampling();

	void equUniformSampling();

	void prepareSampleShow();

	void solve();

	void constructEquations();

	void nnlsCall();

	void calculateSimilarityMap();

	double interpolate(const Vec3b& f);

	void calculateSelectsMSE();

public:	// public variables

	bool debug;

	ColorSpace colorSpace;

	// sample method
	SampleMethod basisSampleMethod;
	SampleMethod equSampleMethod;

	// number of samples
	int numBasisSamples;
	int numEquSamples;

	double sigma;

	Mat sMap;

	double selectsMSE;

	// actual numbers
	int numSelects;
	int numBasis;
	int numEquations;

	// sample illustration
	Mat basisShow;
	Mat equShow;

public:	// public methods

	// img must be CV_8UC3 and in BGR color space
	// selects is a matrix (CV_8S) with
	// >0 select
	// 0  not select
	// <0 select as exclusive
	SelectPropagater(const Mat& img, const Mat& selects);

	void apply(Mat& sMap);

};

#endif
