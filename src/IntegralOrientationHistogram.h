#pragma once
#include "opencv2/opencv.hpp"
enum integramOrientationHistogramFilterType {INTEGRAL_ORIENTATION_HISTOGRAM_SOBEL,INTEGRAL_ORIENTATION_HISTOGRAM_SCHARR};
struct sector {
	float angle; double value;
};
class IntegralOrientationHistogram
{
	public:
		IntegralOrientationHistogram();
		IntegralOrientationHistogram(int sectors, unsigned char *data, int w, int h, int threshold);
		IntegralOrientationHistogram(int sectors, integramOrientationHistogramFilterType filterType, unsigned char *data, int w, int h, int threshold);
		IntegralOrientationHistogram(int sectors, integramOrientationHistogramFilterType filterType, cv::Mat src, cv::Rect roirect, int threshold);
		IntegralOrientationHistogram(int sectors, integramOrientationHistogramFilterType filterType, cv::Mat *roi, int w, int h, int threshold);
		IntegralOrientationHistogram(std::vector<sector> histogram, bool compareOnly = true);
        IntegralOrientationHistogram(std::vector<std::vector<sector> > histograms);
		~IntegralOrientationHistogram();
		void calculate();
		double calculateDifference(IntegralOrientationHistogram *cmpr);
		double calculateAngle(IntegralOrientationHistogram *src);
		void drawCircularHistogram();
		void rotateToMaximum();
		std::vector<sector> getHistogram();
		cv::Mat getHistogramMat();
		std::vector<sector> histogram;
		std::string ToString();
        cv::Mat convertHistogramVectorToMat();
	protected:
		void filter();
		//globals
		int total_sectors;
		int filterType;
	private:
		std::vector<int> getMaximums(std::vector<sector> values, int totalmaximums);
		void filterMask(cv::Mat *mask, cv::Mat *angle, cv::Mat *magn);
		void addToHistogram(float angle);
		void initializeHistogram();
		void normalizeHistogram();
		unsigned char *data;
		int w;
		int h;
		int threshold;
		cv::Mat *dx;
		cv::Mat *dy;
		cv::Mat *roi;
		unsigned char *histogramData;
		cv::Mat circularHistogram;
		//globals
};

