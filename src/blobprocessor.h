#ifndef BLOBPROCESSOR_H
#define BLOBPROCESSOR_H
#include "opencv2/opencv.hpp"
#include "IntegralOrientationHistogram.h"
#include "colorFinder.h"
#include <iostream>
#include <fstream>

class BlobProcessor
{
public:
    BlobProcessor();
    void Process(cv::Mat skinMap, cv::Mat foregroundMap, cv::Mat input);

    void DispMap(cv::Mat input, cv::Mat &output, int threshold);
    void RecognizeClusters(cv::Mat input, std::vector<cv::Rect> clusters);
    std::vector<cv::Point> GetGoodContour(cv::Mat input, cv::Rect rect);
    void getRects(cv::Mat input, std::vector<cv::Rect> &rects);
    void getMask(cv::Mat input, cv::Mat &mask);
    void GetGoodClusters(cv::Mat input, std::vector<cv::Rect> clusters, std::vector<cv::Rect> &goodClusters);
    void trainClassifier(cv::Mat dispInput, cv::Mat skinMap, cv::Rect rect, std::vector<cv::Point> &contour);
    void serializeContour(std::string filename, std::vector<cv::Point> contour);

    void getContoursFromRoi(cv::Mat dispInput, cv::Mat skinMap, cv::Rect rect, std::vector<cv::Point> &contour);
    std::vector<cv::Point> getCommonContour(std::vector<std::vector<cv::Point> > bigger, std::vector<std::vector<cv::Point> > smaller);
    std::vector<cv::Point> getMaxContour(std::vector<std::vector<cv::Point> > contours);
    void deserializeFileToContours(std::string filename, std::vector<std::vector<cv::Point> > &contours);
    void getContours(cv::Mat input, std::vector<std::vector<cv::Point> > &contours);
private:
    std::vector<std::vector<cv::Point> > trainContours;

};





#endif // BLOBPROCESSOR_H
