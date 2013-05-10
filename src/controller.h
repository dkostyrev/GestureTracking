#ifndef CONTROLLER_H
#define CONTROLLER_H
#include "opencv2/opencv.hpp"
#include "blobgetter.h"
#include "blobprocessor.h"
#include "classifier.h"
#include "motionestimator.h"
#include <string>
#include "time.h"
#include "winsock2.h"
enum MatchedClassifier {
    PALM,
    FIST,
    FINGER,
    NONE
};

class Controller
{
public:
    Controller();
    void Process(cv::Mat frame);
    void sendSocket(std::string payload);
private:
    void checkKeys();
    std::vector<std::vector<std::vector<double> > > resizeToMax(std::vector<std::vector<std::vector<double> > > data, std::vector<std::vector<double> > max);
    std::vector<std::vector<double> > getMedianVector(std::vector<std::vector<std::vector<double> > > data);
    void getEuclidianDistance(std::vector<std::vector<std::vector<double> > > saved, std::vector<std::vector<std::vector<double> > > queryInput);
    void labelAndTrain();
    void classify();
    BlobGetter blobgetter;
    Classifier classifier;
};

#endif // CONTROLLER_H
