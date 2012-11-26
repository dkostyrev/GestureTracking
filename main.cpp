#include <iostream>
#include "opencv2/opencv.hpp"
#include "src/tracker.h"
#include "src/gaussianaverage.h"
#include "src/structures.h"
using namespace std;

int main()
{
    cv::VideoCapture cap;
    cap.open("C:\\Projects\\SampleVids\\Walk1.mpg");
    cv::Mat frame;
    tracker::Tracker tracker = tracker::Tracker(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    int key = 0;
    //GaussianAverage ga = GaussianAverage(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    cv::Mat output = cv::Mat(cap.get(CV_CAP_PROP_FRAME_HEIGHT),cap.get(CV_CAP_PROP_FRAME_WIDTH), CV_8UC1);
    while (cap.read(frame)){
        //ga.Process(frame, output);
        std::vector<tracker::ROI> rois = std::vector<tracker::ROI>();
        tracker.Track(frame, rois);

        for (size_t i = 0; i < rois.size(); ++i){
            cv::rectangle(frame, rois.at(i).boundingRect, cv::Scalar(0, 255, 0), 1, CV_AA);
            cv::line(frame, cv::Point(rois.at(i).center.x, rois.at(i).center.y - 5),
                     cv::Point(rois.at(i).center.x, rois.at(i).center.y + 5), cv::Scalar(0, 0, 255), 1, CV_AA );
            cv::line(frame, cv::Point(rois.at(i).center.x - 5, rois.at(i).center.y),
                     cv::Point(rois.at(i).center.x + 5, rois.at(i).center.y), cv::Scalar(0, 0, 255), 1, CV_AA );
        }
        cv::imshow("Input", frame);
        key = cv::waitKey(1);
        switch (key){
            case 27:
                return(0);
        }
    }
    return 0;
}

