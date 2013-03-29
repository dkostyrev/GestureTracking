#pragma once
#include "opencv2/opencv.hpp"
class ColorFinder {
	public:
		ColorFinder(void);
        int getIntervalsCount(cv::Mat &hsv, int bars);
        void find(cv::Scalar& lowBound, cv::Scalar& highBound, int channel, cv::Mat &mat, int delta = 5, int bars = 90, bool overlap = false);
		void plot();
		cv::Mat getPlottedMat() { return plottedMat; }
        ~ColorFinder();

private:
		cv::MatND histogram;
        std::vector<cv::Mat> matPlanes;
		cv::Mat plottedMat;
};
