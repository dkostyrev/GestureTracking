#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#define DEBUG = true;
#include "opencv2/opencv.hpp"
#include "IntegralOrientationHistogram.h"
#include "opencv2/ml/ml.hpp"
#include <iostream>
#include <fstream>
struct FeatureVector {
    int label;
    double ecc;
    std::vector<double> moments;
    std::vector<double> histogram;
    int VECTOR_SIZE;
};
class Classifier
{
public:
    Classifier();
    void AddToTrainSet(std::vector<cv::Point> contour, int label);
    void Train();
    int Recognize(std::vector<cv::Point> contour);
    size_t GetTrainSetSize();
private:

    double getEccentricity(std::vector<cv::Point> contour);

    FeatureVector getFeatures(std::vector<cv::Point> contour);
    std::vector<FeatureVector> trainVectors;
    CvNormalBayesClassifier classifier;
};

#endif // CLASSIFIER_H
