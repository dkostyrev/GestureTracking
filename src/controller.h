#ifndef CONTROLLER_H
#define CONTROLLER_H
#include "opencv2/opencv.hpp"
#include "blobgetter.h"
#include "blobprocessor.h"
#include "classifier.h"
#include "motionestimator.h"
#include <string>
const std::string CASCADE_PATH_PALM = "C:\\Projects\\GestureTracking\\xml\\palm.xml";
const std::string CASCADE_PATH_FIST = "C:\\Projects\\GestureTracking\\xml\\fist.xml";
const std::string CASCADE_PATH_HAND = "C:\\Projects\\GestureTracking\\xml\\hand.xml";
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

    void filterRois(std::vector<cv::Rect> input, std::vector<cv::Rect> &output);
    void classifyRegions(cv::Size matSize, std::vector<std::vector<cv::Point> > contours, std::vector<int> &classes);
    void Approach1(cv::Mat frame);
    void fillRect(cv::RotatedRect &rrect, cv::Rect startRect, cv::Mat mat);
    void Approach2(cv::Mat frame);
private:
    void resizeRegions(std::vector<cv::Rect> regions, cv::Size frameSize, size_t delta, std::vector<cv::Rect> &resized);
    void checkKeys(cv::Mat frame, std::vector<std::vector<cv::Point> > contours);
    MatchedClassifier currentClassifier;
    MatchedClassifier acquireHandRegion(cv::Mat frame, std::vector<cv::Rect> &candidates);
    cv::Rect handRegion;
    std::vector<cv::Rect> currentCandidates;
    std::vector<cv::Rect> handRegionHistory;
    cv::CascadeClassifier cascadeClassifier;
    void classifyRegions(std::vector<std::vector<cv::Point> > contours, std::vector<int> &classes);
    bool ifRegionInHistory(cv::Mat frame, std::vector<cv::Rect> regions);
    BlobGetter blobgetter;
    BlobProcessor blobprocessor;
    Classifier classifier;
};

#endif // CONTROLLER_H
