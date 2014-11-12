#pragma once
#include "opencv2/opencv.hpp"
class ColorFinder {
	public:
        ColorFinder();
        int GetIntervalsCount(cv::Mat &hsv, int bars);
        void Find(cv::Scalar& lowBound, cv::Scalar& highBound, int channel, cv::Mat &mat, int delta = 5, int bars = 90, bool overlap = false);
        void Plot();
        cv::Mat GetPlottedMat() { return plottedMat; }
private:
		cv::MatND histogram;
        std::vector<cv::Mat> matPlanes;
		cv::Mat plottedMat;
        cv::Point maxloc;
        int elementsPerBar;

};
