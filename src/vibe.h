#ifndef VIBE_H
#define VIBE_H
#include "opencv2/opencv.hpp"
class ViBe
{
public:

    ViBe();
    ViBe(int width, int height);
    void Initialize(cv::Mat firstframe);
    void Process(cv::Mat input, cv::Mat &segMap);
private:
    int EuclidianDistance(uchar a, uchar b);
    int N, width, height, R, smin, r_s;
    std::vector<cv::Mat> samples;
    cv::RNG rng;

};

#endif // VIBE_H
