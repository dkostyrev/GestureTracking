#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#define DEBUG = true;
#include "opencv2/opencv.hpp"
#include "IntegralOrientationHistogram.h"
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
    Classifier(std::string model);
    void AddToTrainSet(cv::Size matSize, std::vector<cv::Point> contour, int label);
    void Train(bool save);
    bool IsLoadedModel();
    size_t GetTrainSetSize();
    int Recognize(std::vector<cv::Point> contour, cv::Size matSize);
private:
    bool isLoadedModel;
    double getEccentricity(std::vector<cv::Point> contour);
    void serializeTrainingVectors(std::string filename);
    void deserializeTrainingVectors(std::string filename);
    FeatureVector getFeatures(std::vector<cv::Point> contour, cv::Size matSize);
    std::vector<FeatureVector> trainVectors;
    //cv::NormalBayesClassifier classifier;
    CvANN_MLP classifier;
};

#endif // CLASSIFIER_H
