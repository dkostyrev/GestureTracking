#include <iostream>
#include "opencv2/opencv.hpp"
using namespace std;

int main()
{
    cv::VideoCapture cap;
    cap.open("C:\\Projects\\SampleVids\\Walk1.mpg");
    cv::Mat frame;
    cv::Mat prevFrame;
    while (cap.read(frame)){
        cv::imshow("input", frame);
        if (prevFrame.)
        int key = cv::waitKey(1);
        switch (key){
            case 27:
                return(0);
        }
    }
    return 0;
}

