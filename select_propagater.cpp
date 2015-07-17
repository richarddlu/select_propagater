#include "select_propagater.h"

SelectPropagater::SelectPropagater(const Mat& img, const Mat& selects) {
	debug = false;
	size = img.size();
	this->img = img.clone();
	this->selects = selects.clone();
	colorSpace = BGR;
	sigma = 0.003;
}

void SelectPropagater::apply(Mat& sMap) {
	// convert color space
	convertColorSpace();

	// extract select info
	extractSelect();

	// sample basis and equations
	sampleBasis();
	sampleEquation();

	// solve RBF coefficients
	solve();

	// calculate similarity map
	calculateSimilarityMap();
	sMap = this->sMap.clone();

	// calculate selects MSE
	if(debug)
		calculateSelectsMSE();
}

void SelectPropagater::convertColorSpace() {
	if(colorSpace == YCrCb)
		cvtColor(img, img, CV_BGR2YCrCb);
	else if(colorSpace == Lab)
		cvtColor(img, img, CV_BGR2Lab);
}

void SelectPropagater::extractSelect() {
	for(int h = 0; h < size.height; h++) {
		for(int w = 0; w < size.width; w++) {
			if(selects.at<char>(h,w) != 0) {
				selectedColors.push_back(img.at<Vec3b>(h,w));
				if(selects.at<char>(h,w) > 0) {	// inclusive
					selectedStrenths.push_back(1.0);
				} else {	// exclusive
					selectedStrenths.push_back(0.0);
				}
				if(debug)
					selectedPositions.push_back(Point(w,h));
			}
		}
	}
	numSelects = selectedColors.size();
}

void SelectPropagater::sampleBasis() {
	basisColors = selectedColors;
	basisStrenths = selectedStrenths;
	if(debug)
		basisPositions = selectedPositions;
	numBasis = basisColors.size();
}

void SelectPropagater::sampleEquation() {
	equColors = selectedColors;
	equStrenths = selectedStrenths;
	if(debug)
		equPositions = selectedPositions;
	numEquations = equColors.size();
}

void SelectPropagater::SelectPropagater::solve() {
	constructEquations();
	nnlsCall();
}

void SelectPropagater::constructEquations() {
	// Construct matrix A row by row
	for(int i = 0; i < numEquations; i++) {
		Mat row(1, numBasis, CV_64F);
		Vec3b f = equColors[i];
		for(int j = 0; j < numBasis; j++) {
			Vec3b fi = basisColors[j];
			double r = norm(f, fi, NORM_L2);
			double rf = exp(-sigma * r * r);
			row.at<double>(0,j) = rf;
		}
		// row *= q[selectedColorLabels[i]];
		A.push_back(row);
	}
	
	// Construct matrix B
	b = Mat::zeros(numEquations, 1, CV_64F);
	for(int i = 0; i < numEquations; i++) {
		b.at<double>(i,0) = equStrenths[i];
	}
}

void SelectPropagater::nnlsCall()
{
	// Construct mda, m and n
	int m = A.size().height;
	int n = A.size().width;
	int mda = m;

	// Construct array for A
	double* arrA;
	if(A.isContinuous()) {
		arrA = (double*)A.data;
	}
	else {
		arrA = (double*)malloc(m*n*8);
		for(int h = 0; h < m; h++)
			memcpy(&(arrA[h*n]), A.ptr<double>(h), n*8);
	}

	// Contruct array for b
	double* arrb;
	if(b.isContinuous()) {
		arrb = (double*)b.data;
	}
	else {
		arrb = (double*)malloc(m*8);
		for(int h = 0; h < m; h++)
			arrb[h] = b.at<double>(h,0);
	}

	// Construct array for a
	double* arra;
	arra = (double*)malloc(n*8);

	// COnstruct working space array
	double rnorm;
	double* w = (double*)malloc(n*8);
	double* zz = (double*)malloc(m*8);
	int* index = (int*)malloc(n*4);

	// Call nnls
	int mode;
	nnls(arrA, mda, m, n, arrb, arra, &rnorm, w, zz, index, &mode, 1000000);
	// cout<<mode<<endl;

	// Construct return matrix a
	a = *(new Mat(n, 1, CV_64F, arra));
}

void SelectPropagater::calculateSimilarityMap() {
	sMap = Mat::zeros(size, CV_64F);
	for(int h = 0; h < size.height; h++) {
		for(int w = 0; w < size.width; w++) {
			double temp = interpolate(img.at<Vec3b>(h,w));
			sMap.at<double>(h,w) = temp;
		}
	}
}

double SelectPropagater::interpolate(const Vec3b& f) {
	double result = 0;
	for(int i = 0; i < numBasis; i++) {
		double r = norm(f, basisColors[i], NORM_L2);
		double rf = a.at<double>(i,0) * exp(-sigma * r * r);
		result += rf;
	}
	return result;
}

void SelectPropagater::calculateSelectsMSE()
{
	selectsMSE = 0;
	for(int i = 0; i < numSelects; i++) {
		double rbf = sMap.at<double>(selectedPositions[i]);
		selectsMSE += pow(rbf - selectedStrenths[i], 2);
	}
	selectsMSE /= numSelects;
}
