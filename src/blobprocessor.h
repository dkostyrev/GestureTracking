#ifndef BLOBPROCESSOR_H
#define BLOBPROCESSOR_H
#include "opencv2/opencv.hpp"
#include "IntegralOrientationHistogram.h"
#include "classifier.h"
#include <iostream>
#include <fstream>

class BlobProcessor
{
public:
    BlobProcessor();
    void Process(cv::Mat input, cv::Mat skinMap, cv::Mat foregroundMap);

    void DispMap(cv::Mat input, cv::Mat &output, int threshold);
    void getRects(cv::Mat input, std::vector<cv::Rect> &rects);
    void getMask(cv::Mat input, cv::Mat &mask);

    void serializeContour(std::string filename, std::vector<cv::Point> contour);

    void deserializeContours(std::string filename, std::vector<std::vector<cv::Point> > &contours);
    cv::Point getCenterOfMasses(std::vector<cv::Point> contour);
    double getEccentricity(std::vector<cv::Point> contour);
    void cutRoi(std::vector<cv::Point> contour, std::vector<cv::Rect> &clusters);
    std::vector<cv::Rect> getRectsFromLabeledPoints(int totalLabels, std::vector<int> labels, std::vector<cv::Point> points);
    void getCuttedRoisFromMap(cv::Mat map, std::vector<cv::Rect> &rois, int extra);
    void getContours(cv::Mat input, std::vector<std::vector<cv::Point> > &contours, int method = CV_RETR_EXTERNAL, cv::Point offset = cv::Point());
    void trainProcedure(cv::Mat map, std::vector<cv::Rect> rois);
    std::vector<cv::Point> getMaxContourFromRoi(cv::Mat map, cv::Rect roi);
    void MedianFilter(cv::Mat input, cv::Mat &output, size_t times);
private:
    std::vector<std::vector<cv::Point> > trainContours;
    Classifier classifier;
    bool trainInProgress;
    cv::Mat medianKernel;
};





#endif // BLOBPROCESSOR_H
