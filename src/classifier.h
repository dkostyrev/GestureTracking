#pragma once
#define DEBUG = false;
#include "opencv2/opencv.hpp"
#include "IntegralOrientationHistogram.h"
#include <iostream>
#include <fstream>
#include "time.h"
#include <memory>


class Classifier
{
public:
    struct FeatureVector {
        int label;
        std::vector<float> a; //from zero to 1/4
        std::vector<float> b; //from 1/4 to 1/2
        std::vector<float> c; //from 1/2 to 3/4
        std::vector<float> d; //from 3/4 to end
        int VECTOR_SIZE;
    };


    Classifier();
    Classifier(std::string model);
    void AddToTrainSet(int label, std::vector<std::vector<double> > histograms);
    void Train(bool generateTrain);
    bool IsTrained();
    size_t GetTrainSetSize();
    int Recognize(std::vector<std::vector<double> > histograms);
    void serializeTrainingVectors(std::string filename);
private:
    float CalculateChiSqrDistances(FeatureVector vector, int label);
    FeatureVector GetFeatureVector(int label, std::vector<std::vector<double> > histograms);
    std::vector<float> GetMedianHistogram(std::vector<std::vector<double> > histograms, int begin, int end);
    double GetEccentricity(std::vector<cv::Point> contour);
    void DeserializeTrainingVectors(std::string filename);
    FeatureVector GetFeatures(std::vector<cv::Point> contour, cv::Size matSize);

    std::vector<FeatureVector> trainVectors;
    std::shared_ptr<CvKNearest> classifier;
    cv::Mat trainMat, responsesMat;
    bool isTrained;
};

