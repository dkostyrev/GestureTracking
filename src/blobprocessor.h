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
    void Process(cv::Mat input, cv::Mat skinMap, cv::Mat foregroundMap);

    void getRects(cv::Mat input, std::vector<cv::Rect> &rects);

    void serializeContour(std::string filename, std::vector<cv::Point> contour);

    void deserializeContours(std::string filename, std::vector<std::vector<cv::Point> > &contours);
    cv::Point getCenterOfMasses(std::vector<cv::Point> contour);
    double getEccentricity(std::vector<cv::Point> contour);
    void cutRoi(std::vector<cv::Point> contour, std::vector<cv::Rect> &clusters);
    std::vector<cv::Rect> getRectsFromLabeledPoints(int totalLabels, std::vector<int> labels, std::vector<cv::Point> points);
    void getCuttedRoisFromMap(cv::Mat map, std::vector<cv::Rect> &rois, int extra);
    void getContours(cv::Mat input, std::vector<std::vector<cv::Point> > &contours, int method = CV_RETR_EXTERNAL, cv::Point offset = cv::Point());
    void trainProcedure(Classifier &classifier, cv::Mat map, std::vector<std::vector<cv::Point> > contours);
    std::vector<cv::Point> getMaxContourFromRoi(cv::Mat map, cv::Rect roi);
    void resizeRegions(std::vector<cv::Rect> regions, cv::Size frameSize, size_t deltax, size_t delta_y, std::vector<cv::Rect> &resized);
    void growRegions(bool useGray, cv::Mat input, cv::Point start, std::vector<cv::Point> &contour);
    void growRegions(bool useGray, cv::Mat input, cv::Rect roi, std::vector<cv::Point> &contour);
    void checkTopBottom(bool &bottom, bool &top, cv::Mat hsv, cv::Point point, std::vector<cv::Point> &points, int x);
    std::vector<cv::Point> getMaxContour(cv::Mat map, cv::Point offset);
    void contourRefine(std::vector<cv::Point> contour, cv::Mat blobMask, std::vector<cv::Point> &refinedContour);
    void filterContours(std::vector<std::vector<cv::Point> > contours, std::vector<std::vector<cv::Point> > filtered);
private:
    std::vector<uchar> sampleLine(cv::Mat mat, cv::Point p1, cv::Point p2);
    cv::Point2f normalAtPoint(cv::Point prev, cv::Point current, cv::Point next, bool inOut);
    cv::Point2f normalizePoint(cv::Point2f point);

    bool trainInProgress;
    cv::Mat medianKernel;
};





#endif // BLOBPROCESSOR_H
