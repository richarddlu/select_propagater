#include "select_propagater.h"

SelectPropagater::SelectPropagater(const Mat& img, const Mat& selects) {
	debug = false;
	size = img.size();
	this->img = img.clone();
	this->selects = selects.clone();
	colorSpace = BGR;
	basisSampleMethod = NoSample;
	equSampleMethod = NoSample;

	// default number of samples, -1 means let system decide them
	numSelects = 0;
	numBasisSamples = 2000;
	numEquSamples = 6000;

	sigma = 0.003;
}

void SelectPropagater::apply(Mat& sMap) {
	// convert color space
	convertColorSpace();

	// extract select info
	extractSelect();

	// // if no pixel selected
	if(numSelects == 0)
		return;
	
	// // sample basis and equations
	sampleBasis();
	sampleEquation();

	// // show samples
	// if(debug)
	// 	prepareSampleShow();

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
	strenMap = Mat::zeros(size, CV_64F);
	for(int h = 0; h < size.height; h++) {
		for(int w = 0; w < size.width; w++) {
			if(selects.at<char>(h,w) != 0) {
				selectedPositions.push_back(Point(w,h));
				if(selects.at<char>(h,w) > 0) {	// inclusive
					strenMap.at<double>(h,w) = 1.0;
				}
				numSelects++;
			}
		}
	}
}

// void SelectPropagater::sampleBasis() {
// 	// sample number validate
// 	if(basisSampleMethod != NoSample) {
// 		numBasis = numBasisSamples;
// 		if(numBasis <= 0)
// 			numBasis = 1;
// 		if(numBasis > numSelects)
// 			numBasis = numSelects;
// 	}

// 	if(basisSampleMethod == Uniform)	// uniform sampling
// 		basisUniformSampling();
// 	else {	// no sampling
// 		basisColors = selectedColors;
// 		if(debug) {
// 			basisStrenths = selectedStrenths;
// 			basisPositions = selectedPositions;
// 		}
// 		numBasis = basisColors.size();
// 	}
// }

void SelectPropagater::sampleBasis() {
	numBasis = numSelects;
	basisSelects.resize(numBasis, true);
}

// void SelectPropagater::sampleEquation() {
// 	// sample number validate
// 	if(equSampleMethod != NoSample) {
// 		numEquations = numEquSamples;
// 		if(numEquations <= 0)
// 			numEquations = 1;
// 		if(numEquations > numSelects)
// 			numEquations = numSelects;
// 	}

// 	if(equSampleMethod == Uniform)	// uniform sampling
// 		equUniformSampling();
// 	else {	// no sampling
// 		equColors = selectedColors;
// 		equStrenths = selectedStrenths;
// 		if(debug)
// 			equPositions = selectedPositions;
// 		numEquations = equColors.size();
// 	}
// }
void SelectPropagater::sampleEquation() {
	numEquations = numSelects;
	equSelects.resize(numEquations, true);
}

void SelectPropagater::basisUniformSampling() {
	RNG rng(getTickCount());
	vector<Vec3b> selectedColorsTemp = selectedColors;
	vector<double> selectedStrenthsTemp;
	vector<Point> selectedPositionsTemp;
	if(debug) {
		selectedStrenthsTemp = selectedStrenths;
		selectedPositionsTemp = selectedPositions;
	}
	for(int i = 0; i < numBasis; i++) {
		int rn = rng.uniform(0, numSelects-i);
		basisColors.push_back(selectedColorsTemp[rn]);
		if(debug) {
			basisStrenths.push_back(selectedStrenthsTemp[rn]);
			basisPositions.push_back(selectedPositionsTemp[rn]);
		}

		// erase selected element
		selectedColorsTemp.erase(selectedColorsTemp.begin()+rn);
		if(debug) {
			selectedStrenthsTemp.erase(selectedStrenthsTemp.begin()+rn);
			selectedPositionsTemp.erase(selectedPositionsTemp.begin()+rn);
		}
	}
}

void SelectPropagater::equUniformSampling() {
	RNG rng(getTickCount());
	vector<Vec3b> selectedColorsTemp = selectedColors;
	vector<double> selectedStrenthsTemp = selectedStrenths;
	vector<Point> selectedPositionsTemp;
	if(debug)
		selectedPositionsTemp = selectedPositions;
	for(int i = 0; i < numEquations; i++) {
		int rn = rng.uniform(0, numSelects-i);
		equColors.push_back(selectedColorsTemp[rn]);
		equStrenths.push_back(selectedStrenthsTemp[rn]);
		if(debug)
			equPositions.push_back(selectedPositionsTemp[rn]);

		// erase selected element
		selectedColorsTemp.erase(selectedColorsTemp.begin()+rn);
		selectedStrenthsTemp.erase(selectedStrenthsTemp.begin()+rn);
		if(debug)
			selectedPositionsTemp.erase(selectedPositionsTemp.begin()+rn);
	}
}

void SelectPropagater::prepareSampleShow() {
	basisShow = img.clone();
	equShow = img.clone();
	for(int i = 0; i < numBasis; i++) {
		if(basisStrenths[i] < 0.5)
			circle(basisShow, basisPositions[i], 2.0, Scalar(0,0,255), 1, 8);
		else
			circle(basisShow, basisPositions[i], 2.0, Scalar(0,255,0), 1, 8);
	}
	for(int i = 0; i < numEquations; i++) {
		if(equStrenths[i] < 0.5)
			circle(equShow, equPositions[i], 2.0, Scalar(0,0,255), 1, 8);
		else
			circle(equShow, equPositions[i], 2.0, Scalar(0,255,0), 1, 8);
	}
}

void SelectPropagater::SelectPropagater::solve() {
	constructEquations();
	nnlsCall();
}

void SelectPropagater::constructEquations() {
	// Construct matrix A row by row
	for(int i = 0; i < numSelects; i++) {
		if(equSelects[i]) {
			Mat row(1, numBasis, CV_64F);
			Vec3b f = img.at<Vec3b>(selectedPositions[i]);
			int count = 0;
			for(int j = 0; j < numSelects; j++) {
				if(basisSelects[j]) {
					Vec3b fi = img.at<Vec3b>(selectedPositions[j]);
					double r = norm(f, fi, NORM_L2);
					double rf = exp(-sigma * r * r);
					row.at<double>(0,count) = rf;
					count++;
				}
			}
			// row *= q[selectedColorLabels[i]];
			A.push_back(row);
		}
	}
	
	// Construct matrix B
	b = Mat::zeros(numEquations, 1, CV_64F);
	int count = 0;
	for(int i = 0; i < numSelects; i++) {
		if(equSelects[i]) {
			b.at<double>(count,0) = strenMap.at<double>(selectedPositions[i]);
			count++;
		}
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
	int count = 0;
	for(int i = 0; i < numSelects; i++) {
		if(basisSelects[i]) {
			Vec3b fi = img.at<Vec3b>(selectedPositions[i]);
			double r = norm(f, fi, NORM_L2);
			double rf = a.at<double>(count,0) * exp(-sigma * r * r);
			result += rf;
			count++;
		}
	}
	return result;
}

void SelectPropagater::calculateSelectsMSE()
{
	selectsMSE = 0;
	for(int i = 0; i < numSelects; i++) {
		double rbf = sMap.at<double>(selectedPositions[i]);
		selectsMSE += pow(rbf - strenMap.at<double>(selectedPositions[i]), 2);
	}
	selectsMSE /= numSelects;
}
