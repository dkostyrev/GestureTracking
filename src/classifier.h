#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#define DEBUG = true;
#include "opencv2/opencv.hpp"
#include "IntegralOrientationHistogram.h"
#include <iostream>
#include <fstream>
/*struct FeatureVector {
    int label;
    std::vector<double> u;
    std::vector<double> v;
    int VECTOR_SIZE;
};
struct FeatureVector {
    int label;
    std::vector<std::vector<double> > histograms;
};
*/
struct FeatureVector {
    int label;
    std::vector<double> a;
    std::vector<double> b;
    std::vector<double> c;
    int VECTOR_SIZE;
};

class Classifier
{
public:
    Classifier();
    Classifier(std::string model);
    void AddToTrainSet(int label, std::vector<std::vector<double> > histograms);
    void Train(bool save);
    bool IsTrained();
    size_t GetTrainSetSize();
    int Recognize(std::vector<std::vector<double> > histograms);
    void serializeTrainingVectors(std::string filename);
private:
    bool isTrained;
    double getEccentricity(std::vector<cv::Point> contour);
    void deserializeTrainingVectors(std::string filename);
    FeatureVector getFeatures(std::vector<cv::Point> contour, cv::Size matSize);
    std::vector<FeatureVector> trainVectors;
    //cv::NormalBayesClassifier classifier;
    CvANN_MLP classifier;

};

#endif // CLASSIFIER_H
