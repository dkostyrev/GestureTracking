#include "IntegralOrientationHistogram.h"
//file structure:
//label,val1,val2,.....,val(total_sectors)
IntegralOrientationHistogram::IntegralOrientationHistogram()
{
	this->filterType = INTEGRAL_ORIENTATION_HISTOGRAM_SOBEL;
	this->total_sectors = 16;
	roi = new cv::Mat(480,640,CV_8U,data);
	dx = new cv::Mat(roi->size(),CV_32F);
	dy = new cv::Mat(roi->size(),CV_32F);
    //circularHistogram = cv::Mat(400,400,CV_8UC3);
	this->threshold = 100;
	this->histogram = std::vector<sector>();
}
//--------------------------------------------------------------
IntegralOrientationHistogram::IntegralOrientationHistogram(int sectors,unsigned char *data,int w,int h,int threshold)
{
	this->total_sectors = sectors;
	roi = new cv::Mat(h, w, CV_8U, data);
	dx = new cv::Mat(roi->size(),CV_32F);
	dy = new cv::Mat(roi->size(),CV_32F);
    //circularHistogram = cv::Mat(400,400,CV_8UC3);
	this->threshold = threshold;
	this->histogram = std::vector<sector>();
	this->filterType = INTEGRAL_ORIENTATION_HISTOGRAM_SOBEL;
}
//--------------------------------------------------------------
IntegralOrientationHistogram::IntegralOrientationHistogram(int sectors,integramOrientationHistogramFilterType filterType,unsigned char *data,int w,int h,int threshold)
{
	this->filterType = filterType;
	this->total_sectors = sectors;
	roi = new cv::Mat(h, w, CV_8U, data);
	dx = new cv::Mat(roi->size(),CV_32F);
	dy = new cv::Mat(roi->size(),CV_32F);
    //circularHistogram = cv::Mat(400,400,CV_8UC3);
	this->threshold = threshold;
	this->histogram = std::vector<sector>();
}
//--------------------------------------------------------------
IntegralOrientationHistogram::IntegralOrientationHistogram(int sectors,integramOrientationHistogramFilterType filterType,cv::Mat *roi,int w,int h,int threshold)
{
	this->filterType = filterType;
	this->total_sectors = sectors;
	this->roi = roi;
	dx = new cv::Mat(roi->size(),CV_32F);
	dy = new cv::Mat(roi->size(),CV_32F);
    //circularHistogram = cv::Mat(400,400,CV_8UC3);
	this->threshold = threshold;
	this->histogram = std::vector<sector>();
}
//--------------------------------------------------------------
IntegralOrientationHistogram::IntegralOrientationHistogram(int sectors, integramOrientationHistogramFilterType filterType, cv::Mat src, int threshold) {
    this->total_sectors = sectors;
    this->filterType = filterType;
    this->roi = &src;
    this->threshold = threshold;
    this->histogram = std::vector<sector>();
    dx = new cv::Mat(roi->size(),CV_32F);
    dy = new cv::Mat(roi->size(),CV_32F);
    circularHistogram = cv::Mat(400,400,CV_8UC3);
}
//--------------------------------------------------------------
IntegralOrientationHistogram::IntegralOrientationHistogram(int sectors, integramOrientationHistogramFilterType filterType, cv::Mat src, cv::Rect roirect, int threshold) {
	this->total_sectors = sectors;
	this->filterType = filterType;
	cv::Point offset = cv::Point(10,10);
	if (roirect.x - offset.x > -1 && roirect.x + roirect.width + offset.x < src.cols && roirect.y - offset.y > -1 && roirect.y + roirect.height + offset.y < src.rows){
		this->roi = new cv::Mat(src,cv::Rect(roirect.x - offset.x, roirect.y - offset.y, roirect.width + 2*offset.x, roirect.height + 2*offset.y));
	}
	else {
		this->roi = new cv::Mat(src,roirect);
	}
	this->threshold = threshold;
	this->histogram = std::vector<sector>();
	dx = new cv::Mat(roi->size(),CV_32F);
	dy = new cv::Mat(roi->size(),CV_32F);
    circularHistogram = cv::Mat(400,400,CV_8UC3);
}
//--------------------------------------------------------------
IntegralOrientationHistogram::IntegralOrientationHistogram(std::vector<sector> histogram, bool compareOnly) {
	this->total_sectors = histogram.size();
	this->filterType = INTEGRAL_ORIENTATION_HISTOGRAM_SOBEL;
	this->threshold = 100;
	this->histogram = histogram;
	
	if (!compareOnly){
		dx = new cv::Mat(roi->size(),CV_32F);
		dy = new cv::Mat(roi->size(),CV_32F);
		
	}
    //circularHistogram = cv::Mat(400, 400, CV_8UC3);
}
//--------------------------------------------------------------
IntegralOrientationHistogram::IntegralOrientationHistogram( std::vector<std::vector<sector> > histograms )
{
	circularHistogram = cv::Mat(400,400,CV_8UC3);
	this->histogram = std::vector<sector>();
	if (histograms.size() > 0){
        std::vector<std::vector<double> > sorted_sectors(histograms[0].size());
		for (size_t i = 0; i < histograms.size(); ++i){
			for (size_t a = 0; a < histograms[i].size(); ++a){
				sorted_sectors[a].push_back(histograms.at(i).at(a).value);
			}
		}
		for (size_t i = 0; i < sorted_sectors.size(); ++i){
			std::sort(sorted_sectors[i].begin(), sorted_sectors[i].end());
			sector new_sector;
			new_sector.value = sorted_sectors[i][sorted_sectors[i].size()/2];
            new_sector.angle = (2*CV_PI/sorted_sectors.size())*i;
			this->histogram.push_back(new_sector);
		}
	}
	this->total_sectors = this->histogram.size();
}
//--------------------------------------------------------------
void IntegralOrientationHistogram::initializeHistogram() {
	this->histogram.clear();
	dx->setTo(cv::InputArray(0));
	dy->setTo(cv::InputArray(0));
	for (int i = 0; i < this->total_sectors; i++){
		sector current = sector();
		current.value = 0;
        current.angle = (2*CV_PI/this->total_sectors)*i;
		this->histogram.push_back(current);
	}
	this->total_sectors = this->histogram.size();
}
//--------------------------------------------------------------
void IntegralOrientationHistogram::normalizeHistogram() {
	int max = 0;
	int maxi = -1;
	for (size_t i = 0; i < histogram.size(); i++){
		if (histogram.at(i).value > max){
			max = histogram.at(i).value;
			maxi = i;
		}
	}
	if (max > 0){
		double norm = 1.0/(double) max;
		for (size_t i = 0; i < histogram.size(); i++){
			histogram.at(i).value = ((histogram.at(i).value*norm));
		}
	}
}
//--------------------------------------------------------------
double IntegralOrientationHistogram::calculateDifference(IntegralOrientationHistogram *cmpr) {
	double d = 0;
	if (cmpr->histogram.size() == this->histogram.size()){
		for (size_t i = 0; i < histogram.size(); i++){
			if (histogram.at(i).value+cmpr->histogram.at(i).value > 0) {
				d += pow(histogram.at(i).value-cmpr->histogram.at(i).value, 2.0)/(histogram.at(i).value + cmpr->histogram.at(i).value);
			}
		}
	}
	return d;
}
//--------------------------------------------------------------
void IntegralOrientationHistogram::calculate() {
	initializeHistogram();
	filter();
	normalizeHistogram();
	delete dx;
	delete dy;
	delete roi;
}
//--------------------------------------------------------------
void IntegralOrientationHistogram::filterMask(cv::Mat *mask, cv::Mat *angle, cv::Mat *magn) {

	switch (this->filterType){
		case INTEGRAL_ORIENTATION_HISTOGRAM_SOBEL:
			cv::Sobel(cv::InputArray(*mask), cv::OutputArray(*dx), CV_32F, 1, 0, 5);
			cv::Sobel(cv::InputArray(*mask), cv::OutputArray(*dy), CV_32F, 0, 1, 5);
			break;
		case INTEGRAL_ORIENTATION_HISTOGRAM_SCHARR:
			cv::Scharr(cv::InputArray(*mask), cv::OutputArray(*dx), CV_32F, 1, 0);
			cv::Scharr(cv::InputArray(*mask), cv::OutputArray(*dy), CV_32F, 0, 1);
			break;
	}
	cv::phase(cv::InputArray(*dx), cv::InputArray(*dy), cv::OutputArray(*angle));
	cv::magnitude(cv::InputArray(*dx), cv::InputArray(*dy), cv::OutputArray(*magn));
	float ang = 0;
	for (int y = 0; y < magn->rows; y++){
		for (int x = 0; x < magn->cols; x++){
			if (magn->at<float>(cv::Point(x,y)) > this->threshold){
				addToHistogram(angle->at<float>(cv::Point(x,y)));
			}
		}
	}
	
}
//--------------------------------------------------------------
void IntegralOrientationHistogram::filter() {
	cv::Mat *angle = new cv::Mat(dx->size(),dx->type());
	cv::Mat *magn = new cv::Mat(dx->size(),dx->type());
	filterMask(roi, angle, magn);
	angle->release();
	magn->release();
}
//--------------------------------------------------------------
void IntegralOrientationHistogram::addToHistogram(float angle) {
	if (angle < 0){
        angle += (CV_PI*2);
	}

	for (size_t i = 0; i < histogram.size(); i++){
		if (histogram.at(i).angle > angle){
			if (i-1 < -1){
				histogram.at(i-1).value++;
			}
			else {
				histogram.at(histogram.size()-1).value++;
			}
			break;
		}
	}
}
//--------------------------------------------------------------
void IntegralOrientationHistogram::drawCircularHistogram() {
	int height = 400; int width = 400;
	circularHistogram.setTo(cv::InputArray(cv::Scalar(255,255,255)));
	for (size_t i = 0; i < this->total_sectors; i++){
		float angle = histogram.at(i).angle;
		double current_value = histogram.at(i).value * 200;
		cv::Point endpoint = cv::Point((current_value*cos(angle))+width/2,height/2+(current_value*sin(angle)));
		cv::line(circularHistogram,cv::Point(width/2,height/2),endpoint,cv::Scalar(0,0,255),1,8,0);
	}
}
//--------------------------------------------------------------
void IntegralOrientationHistogram::rotateToMaximum() {
	//this function "rotates" histogram so it's maximum is located at 0 degrees
	int max = 0, max_index = -1;
	for (size_t i = 0; i < histogram.size(); i++) {
		if (histogram.at(i).value > max) {
			max = histogram.at(i).value;
			max_index = i;
		}
	}
	if (max_index > -1){
		std::rotate(histogram.begin(),histogram.begin()+max_index,histogram.end());
	}
	for (int i = 0; i < this->total_sectors; i++){
        this->histogram.at(i).angle = (2*CV_PI/this->total_sectors)*i;
	}
}

//--------------------------------------------------------------
double IntegralOrientationHistogram::calculateAngle(IntegralOrientationHistogram *src) {
	if (src->histogram.size() != this->histogram.size()) return 0;
	std::vector<int> src_max = getMaximums(src->getHistogram(), 3);
	std::vector<int> this_max = getMaximums(this->getHistogram(), 3);
	std::sort(src_max.begin(), src_max.end());
	std::sort(this_max.begin(), this_max.end());
	std::vector<double> angles = std::vector<double>();
	int maxdistance = 3;
	for (size_t i = 0; i < this_max.size(); ++i){
		if (std::abs(src_max.at(i) - this_max.at(i)) < maxdistance){
			angles.push_back(this->getHistogram().at(this_max.at(i)).angle - src->getHistogram().at(src_max.at(i)).angle);
		}
	}
	std::sort(angles.begin(), angles.end());
	if (angles.size() > 0){
		return angles.at(angles.size()/2);
	}
	else {
		return 0;
	}
}
//--------------------------------------------------------------
std::vector<int> IntegralOrientationHistogram::getMaximums(std::vector<sector> values, int totalmaximums) {
	int max = -1; int maxi = -1;
	std::vector<int> maximums = std::vector<int>();
	std::vector<sector> buff = std::vector<sector>();
	buff.assign(values.begin(), values.end());
	while (maximums.size() != totalmaximums){
		max = -1; maxi = -1;
		for (size_t i = 0; i < buff.size(); ++i){
			if (buff.at(i).value > max){
				max = buff.at(i).value;
				maxi = i;
			}
		}
		maximums.push_back(maxi);
		buff.at(maxi).value = 0;
	}
	return maximums;
}
//--------------------------------------------------------------
std::vector<sector> IntegralOrientationHistogram::getHistogram() {
	return histogram;
}
//--------------------------------------------------------------
cv::Mat IntegralOrientationHistogram::getHistogramMat() {
	return circularHistogram;
}

//--------------------------------------------------------------
IntegralOrientationHistogram::~IntegralOrientationHistogram(void)
{
	circularHistogram.release();
}
//--------------------------------------------------------------
std::string IntegralOrientationHistogram::ToString()
{
	std::stringstream ss;
	for (size_t i = 0; i < this->getHistogram().size(); ++i){
		ss << this->getHistogram()[i].value << ',';
	}
    return ss.str();
}
//--------------------------------------------------------------
cv::Mat IntegralOrientationHistogram::convertHistogramVectorToMat()
{
    cv::Mat result = cv::Mat(1, this->histogram.size(), CV_32FC1);
    for (size_t i = 0; i < this->histogram.size(); ++i){
        result.at<float>(cv::Point(i, 0)) = this->histogram.at(i).value;
    }
}

