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
	numBasis = 0;
	numEquations = 0;
	numBasisSamples = 54;
	numEquSamples = 100;

	sigma = 0.003;
}

void SelectPropagater::apply(Mat& sMap) {
	// Initialize sMap whatever happened
	sMap = Mat::zeros(size, CV_64F);

	// convert color space
	convertColorSpace();

	// extract select info
	extractSelect();

	// // if no pixel selected
	if(numSelects == 0)
		return;
	
	// sample basis and equations
	if(basisSampleMethod == Importance || equSampleMethod == Importance)
		prepareForImportanceSampling();
	sampleBasis();
	sampleEquation();

	if(numBasis <= 0 || numEquations <= 0)
		return;
	if(numEquations < numBasis)	// this case is not likely to happen
		return;

	// show samples
	if(debug)
		prepareSampleShow();

	// solve RBF coefficients
	solve();

	// calculate similarity map
	calculateSimilarityMap(sMap);

	// calculate selects MSE
	if(debug)
		calculateSelectsMSE(sMap);
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

void SelectPropagater::sampleBasis() {
	// sample number validate
	if(basisSampleMethod != NoSample) {
		numBasis = numBasisSamples;
		if(numBasis <= 0)
			numBasis = 1;
		if(numBasis > numSelects)
			numBasis = numSelects;
	}

	if(basisSampleMethod == Uniform)	// uniform sampling
		basisUniformSampling();
	else if(basisSampleMethod == Importance)
		basisImportanceSampling();
	else {	// no sampling
		numBasis = numSelects;
		basisSelects.resize(numBasis, true);
	}
}

void SelectPropagater::sampleEquation() {
	// sample number validate
	if(equSampleMethod != NoSample) {
		numEquations = numEquSamples;
		if(numEquations <= 0)
			numEquations = 1;
		if(numEquations > numSelects)
			numEquations = numSelects;
	}

	if(equSampleMethod == Uniform)	// uniform sampling
		equUniformSampling();
	else if(equSampleMethod == Importance)
		equImportanceSampling();
	else {	// no sampling
		numEquations = numSelects;
		equSelects.resize(numEquations, true);
	}
}

void SelectPropagater::basisUniformSampling() {
	basisSelects.resize(numSelects, false);

	vector<int> indices;
	rSamplingInt(indices, 0, numSelects, numBasis);
	for(size_t i = 0; i < numBasis; i++)
		basisSelects[indices[i]] = true;
}

void SelectPropagater::equUniformSampling() {
	equSelects.resize(numSelects, false);

	vector<int> indices;
	rSamplingInt(indices, 0, numSelects, numEquations);
	for(size_t i = 0; i < numEquations; i++)
		equSelects[indices[i]] = true;
}

void SelectPropagater::basisImportanceSampling() {
	RNG rng(getTickCount());
	basisSelects.resize(numSelects, false);

	size_t actualNumBasis = 0;
	for(size_t i = 0; i < numBasis; i++) {
		double rn = rng.uniform(0.0, 1.0);
		for(size_t j = 0; j < numSelects; j++) {
			if(rn <= CDF[j]) {
				if(!basisSelects[j]) {
					basisSelects[j] = true;
					actualNumBasis++;
				}
				break;
			}
		}
	}
	numBasis = actualNumBasis;
}

void SelectPropagater::equImportanceSampling() {
	RNG rng(getTickCount());
	equSelects.resize(numSelects, false);

	size_t actualNumEquations = 0;
	for(size_t i = 0; i < numEquations; i++) {
		double rn = rng.uniform(0.0, 1.0);
		for(size_t j = 0; j < numSelects; j++) {
			if(rn <= CDF[j]) {
				if(!equSelects[j]) {
					equSelects[j] = true;
					actualNumEquations++;
				}
				break;
			}
		}
	}
	numEquations = actualNumEquations;
}

void SelectPropagater::prepareForImportanceSampling() {
	imgKMeans();
	calculateWeights();
	calculatePDF();
	calculateCDF();
}

void SelectPropagater::imgKMeans() {
	Mat points(numSelects, 1, CV_32FC3);
	for(size_t i = 0; i < numSelects; i++)
		points.at<Vec3f>(i,0) = img.at<Vec3b>(selectedPositions[i]);
	kmeans(points, K, labelMap, TermCriteria(TermCriteria::EPS+TermCriteria::COUNT,1000,0.1), 1, KMEANS_PP_CENTERS);
}

void SelectPropagater::calculateWeights() {
	weights.resize(K);

	vector<size_t> counts(K, 0);
	for(size_t i = 0; i < numSelects; i++) {
		counts[labelMap.at<int>(i,0)]++;
	}
	for(size_t i = 0; i < K; i++) {
		weights[i] = (double)(counts[i]) / numSelects;
	}
}

void SelectPropagater::calculatePDF() {
	PDF.resize(K);
	for(size_t i = 0; i < K; i++) {
		if(weights[i] == 0)
			PDF[i] = 0;
		else
			PDF[i] = 1 / weights[i];
	}
}

void SelectPropagater::calculateCDF() {
	CDF.resize(numSelects);
	for(size_t i = 0; i < numSelects; i++)
		if(i == 0)
			CDF[i] = PDF[labelMap.at<int>(i,0)];
		else
			CDF[i] = CDF[i-1] + PDF[labelMap.at<int>(i,0)];

	// normalize CDF
	double ratio = 1.0 / CDF[numSelects-1];	// assert CDF[numSelects-1] is not equal to zero
	for(size_t i = 0; i < numSelects; i++)
		CDF[i] *= ratio;
}

void SelectPropagater::prepareSampleShow() {
	basisShow = img.clone();
	equShow = img.clone();
	for(int i = 0; i < numSelects; i++) {
		if(basisSelects[i]) {
			if(strenMap.at<double>(selectedPositions[i]) < 0.5)
				circle(basisShow, selectedPositions[i], 1.0, Scalar(0,0,255), 1, 8);
			else
				circle(basisShow, selectedPositions[i], 1.0, Scalar(0,255,0), 1, 8);
		}
	}
	for(int i = 0; i < numSelects; i++) {
		if(equSelects[i]) {
			if(strenMap.at<double>(selectedPositions[i]) < 0.5)
				circle(equShow, selectedPositions[i], 1.0, Scalar(0,0,255), 1, 8);
			else
				circle(equShow, selectedPositions[i], 1.0, Scalar(0,255,0), 1, 8);
		}
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
	// arrA is column major, while Mat in OpenCV is row major. So a transpose is performed
	Mat ATrans;
	transpose(A, ATrans);
	double* arrA;
	if(A.isContinuous()) {
		arrA = (double*)ATrans.data;
	}
	else {
		arrA = (double*)malloc(m*n*8);
		for(int w = 0; w < n; w++)
			memcpy(&(arrA[w*m]), ATrans.ptr<double>(w), m*8);
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

void SelectPropagater::calculateSimilarityMap(Mat& sMap) {
	// efficiency consideration
	for(size_t i = 0; i < numSelects; i++)
		if(basisSelects[i])
			basisSelectIndices.push_back(i);

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
		size_t index = basisSelectIndices[i];
		Vec3b fi = img.at<Vec3b>(selectedPositions[index]);
		double r = norm(f, fi, NORM_L2);
		double rf = a.at<double>(i,0) * exp(-sigma * r * r);
		result += rf;
	}
	return result;
}

void SelectPropagater::calculateSelectsMSE(const Mat& sMap)
{
	selectsMSE = 0;
	for(int i = 0; i < numSelects; i++) {
		double rbf = sMap.at<double>(selectedPositions[i]);
		selectsMSE += pow(rbf - strenMap.at<double>(selectedPositions[i]), 2);
	}
	selectsMSE /= numSelects;
}

void SelectPropagater::rSamplingInt(vector<int>& samples, int a, int b, size_t S) {
	// parameter validation
	if(!(S > 0))
		return;
	if(b - a < S)
		return;

	// clear dst
	samples.clear();

	for(size_t i = 0; i < S; i++) {
		samples.push_back(a + i);
	}

	size_t n = b - a;
	size_t num_seen = S;
	srand(time(NULL));
	while(num_seen < n) {
		num_seen++;
		int rn = rand() % num_seen;
		if(rn < S)
			samples[rn] = a + num_seen - 1;
	}
}
