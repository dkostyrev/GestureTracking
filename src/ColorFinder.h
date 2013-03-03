#pragma once
#include "opencv2/opencv.hpp"
class ColorFinder {
	public:
		ColorFinder(void);
        int getIntervalsCount(cv::Mat &hsv, int bars);
		void find(cv::Scalar& lowBound, cv::Scalar& highBound, cv::Mat &hsv, int bars = 90);
		void plot();
		cv::Mat getPlottedMat() { return plottedMat; }
        ~ColorFinder();

private:
		cv::MatND histogram;
		std::vector<cv::Mat> hsvPlanes;
		cv::Mat plottedMat;
};
