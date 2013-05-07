#include "blobprocessor.h"

BlobProcessor::BlobProcessor() {}

cv::Point BlobProcessor::getCenterOfMasses(cv::Mat blobMask) {
    cv::Moments moments = cv::moments(blobMask);
    return cv::Point(moments.m10 / moments.m00, moments.m01 / moments.m00);
}

cv::Point BlobProcessor::getCenterOfMasses(std::vector<cv::Point> contour) {
    cv::Moments moments = cv::moments(contour);
    return cv::Point(moments.m10 / moments.m00, moments.m01 / moments.m00);
}

double BlobProcessor::getEccentricity(std::vector<cv::Point> contour) {
    cv::Moments moments = cv::moments(contour);
    double a20 = moments.mu20 / moments.m00;
    double a02 = moments.mu02 / moments.m00;
    double a11 = moments.mu11 / moments.m00;
    double l1 = (a20 + a02) / 2 + sqrt(4 * a11 * a11 + (a20 - a02) * (a20 - a02)) / 2;
    double l2 = (a20 + a02) / 2 - sqrt(4 * a11 * a11 + (a20 - a02) * (a20 - a02)) / 2;
    return 1 - l2 / l1;

}
