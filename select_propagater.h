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

	Mat strenMap;

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

	void calculateSimilarityMap(Mat& sMap);

	double interpolate(const Vec3b& f);

	void calculateSelectsMSE(const Mat& sMap);

	// void matReshape(const Mat& src, Mat& dst, int numRows);

	// void imgKMeans();

public:	// public variables

	bool debug;

	ColorSpace colorSpace;

	// sample method
	SampleMethod sampleMethod;

	// number of samples
	int numBasisSamples;
	int numEquSamples;

	double sigma;

	double selectsMSE;

	// actual numbers
	int numSelects;
	int numBasis;
	int numEquations;

	// sample illustration
	Mat basisShow;
	Mat equShow;

	vector<Point> selectedPositions;
	
	vector<bool> basisSelects;
	vector<size_t> basisSelectIndices;	// efficiency consideration
	vector<bool> equSelects;

	// clustering
	size_t K;
	Mat labelMap;

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
