#include "opencv2/opencv.hpp"
#include "src/controller.h"
#include "src/blobintegralhistogram.h"
using namespace std;

int main()
{
    cv::VideoCapture cap;
    cap.open(0);
    cv::Mat frame;
    Controller controller = Controller();
    while (cap.read(frame)) {
        controller.Process(frame);
        cv::imshow("frame", frame);
        int key = cv::waitKey(1);
        if (key == 27)
            return 0;
    }

    return 0;
}

