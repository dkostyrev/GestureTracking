#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#define DEBUG = false;
#include "opencv2/opencv.hpp"
#include "IntegralOrientationHistogram.h"
#include <iostream>
#include <fstream>
#include "time.h"
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
    std::vector<float> a; //from zero to 1/4
    std::vector<float> b; //from 1/4 to 1/2
    std::vector<float> c; //from 1/2 to 3/4
    std::vector<float> d; //from 3/4 to end
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
    FeatureVector getFeatureVector(int label, std::vector<std::vector<double> > histograms);
    std::vector<float> getMedianHistogram(std::vector<std::vector<double> > histograms, int begin, int end);
    bool isTrained;
    double getEccentricity(std::vector<cv::Point> contour);
    void deserializeTrainingVectors(std::string filename);
    FeatureVector getFeatures(std::vector<cv::Point> contour, cv::Size matSize);
    std::vector<FeatureVector> trainVectors;
    //CvANN_MLP classifier;

};

#endif // CLASSIFIER_H
