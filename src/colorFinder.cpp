#include "ColorFinder.h"

int channels[] = { 0 };
int histsize[] = { 45 };
float hranges[] = { 0, 180 };
const float* ranges[] = { hranges };
cv::Point maxloc;
int elementsPerBar;
ColorFinder::ColorFinder(void) {
	hsvPlanes = std::vector<cv::Mat>();
	plottedMat = cv::Mat::ones(300, histsize[0]*10, CV_8U);
	elementsPerBar = hranges[1] / histsize[0];
}
//--------------------------------------------------------------
void ColorFinder::find( cv::Scalar& lowBound, cv::Scalar& highBound, cv::Mat &hsv, int bars /*= 90*/ )
{
	histsize[0] = bars;
	elementsPerBar = hranges[1] / histsize[0];
	cv::split(hsv, hsvPlanes);
	cv::calcHist(&hsv, 1, channels, cv::Mat(), histogram, 1, histsize, ranges, true, false); 
	cv::normalize(histogram, histogram);
	cv::minMaxLoc(histogram, 0, 0, 0, &maxloc, cv::Mat());
	lowBound.val[0] = (maxloc.y * elementsPerBar) - 3;
	highBound.val[0] = (maxloc.y * elementsPerBar) + 3;
	if (lowBound.val[0] < 0){
		lowBound.val[0] += 180;
	}
	if (highBound.val[0] > 180){
		highBound.val[0] -= 180;
	}
} 
//--------------------------------------------------------------
int ColorFinder::getIntervalsCount( cv::Mat &hsv, int bars )
{
    histsize[0] = bars;
    elementsPerBar = hranges[1] / histsize[0];
    cv::split(hsv, hsvPlanes);
    cv::calcHist(&hsv, 1, channels, cv::Mat(), histogram, 1, histsize, ranges, true, false);
    cv::normalize(histogram, histogram);
    cv::minMaxLoc(histogram, 0, 0, 0, &maxloc, cv::Mat());
    return cv::countNonZero(histogram);
}
//--------------------------------------------------------------
void ColorFinder::plot() {
	plottedMat.setTo(cv::Scalar(0));
	for (int i = 1; i < histogram.rows; i++){
        float prevVal = 300 - (300 * histogram.at<float>(i-1, 0));//300
        float binVal =  300 - (300 * histogram.at<float>(i, 0));//300
		cv::line(plottedMat,cv::Point((i-1)*10, prevVal), cv::Point(i*10,binVal), cv::Scalar(255));
	}  
}
//--------------------------------------------------------------
ColorFinder::~ColorFinder() {

}
