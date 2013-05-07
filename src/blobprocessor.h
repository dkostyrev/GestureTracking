#ifndef BLOBPROCESSOR_H
#define BLOBPROCESSOR_H
#include "opencv2/opencv.hpp"
#include "IntegralOrientationHistogram.h"
#include "classifier.h"
#include <iostream>
#include <fstream>
#include <math.h>

class BlobProcessor
{
public:
    BlobProcessor();
    cv::Point getCenterOfMasses(std::vector<cv::Point> contour);
    cv::Point getCenterOfMasses(cv::Mat blobMask);
    double getEccentricity(std::vector<cv::Point> contour);
};





#endif // BLOBPROCESSOR_H
