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

	// importance sampling
	vector<double> PDF;	// size K
	vector<double> CDF;	// size numSelects
	vector<double> weights;

private:	// private methods

	void convertColorSpace();

	void extractSelect();

	void sampleBasis();

	void sampleEquation();

	void basisUniformSampling();

	void equUniformSampling();

	void basisImportanceSampling();

	void equImportanceSampling();

	void prepareForImportanceSampling();

	void imgKMeans();

	void calculateWeights();

	void calculatePDF();

	void calculateCDF();

	void prepareSampleShow();

	void solve();

	void constructEquations();

	void nnlsCall();

	void calculateSimilarityMap(Mat& sMap);

	double interpolate(const Vec3b& f);

	void calculateSelectsMSE(const Mat& sMap);

	void rSamplingInt(vector<int>& samples, int a, int b, size_t S);

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

	// importance sampling
	size_t K;
	Mat labelMap;	// single column matrix, CV_32SC1

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
