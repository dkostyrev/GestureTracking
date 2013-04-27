#ifndef BLOBINTEGRALHISTOGRAM_H
#define BLOBINTEGRALHISTOGRAM_H
#include "opencv2/opencv.hpp"
#include "math.h"
#include "IntegralOrientationHistogram.h"

class BlobIntegralHistogram
{
public:
    BlobIntegralHistogram(size_t sectors, cv::Mat blobMask);
    void Calculate();
    void Plot();
    cv::Mat getPlottedMat() { return circularHistogram; }
    std::vector<sector> getHistogram() { return histogram; }
private:
    void InitializeHistogram();
    void NormalizeHistogram();
    void AddToHistogram(double angle);
    cv::Mat blobMask;
    std::vector<sector> histogram;
    cv::Mat circularHistogram;
    size_t sectors;
    size_t area;
};

#endif // BLOBINTEGRALHISTOGRAM_H
