#include <iostream>
#include "opencv2/opencv.hpp"
#include "src/blobgetter.h"
using namespace std;
/*Main steps:
 * 1)Segmentation
 * 2)Tracking (center of mass of hand - HOG calculation)
 */
int main()
{
    cv::VideoCapture cap;
    cap.open(0);
    //cap.open("C:\\Projects\\GestureTracking\\SampleVideos\\good_brightness_clips\\open_palm_shown_c.avi");
    cv::Mat frame;
    cv::Mat blobMat;
    BlobGetter blobGetter = BlobGetter(VIBE);
    while (cap.read(frame)){
        cv::imshow("Frame", frame);
        blobGetter.Process(frame, blobMat);
        cv::imshow("Blob", blobMat);
        int key = cv::waitKey(1);
        switch (key){
            case 27:
                return(0);
        }
    }
    return 0;
}

