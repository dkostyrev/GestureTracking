#include <iostream>
#include "opencv2/opencv.hpp"
#include "src/blobgetter.h"
#include "src/blobprocessor.h"
using namespace std;
/*Main steps:
 * 1) Segmentation
 * 2) Малое временное окно на кластере полученного из карты цвета, затем Собель, гистограмма, для определения смещения точек
 */
enum Mode {
    SKIN,
    FOREGROUND,
    COMBINATION
};

int main()
{
    cv::VideoCapture cap;
    //cap.open(0);
    cap.open("C:\\Projects\\GestureTracking\\SampleVideos\\good_brightness_clips\\open_palm_shown_c.avi");
    //cap.open("C:\\Projects\\GestureTracking\\SampleVideos\\medium_brightness_clips\\pan_c.avi");
    cv::Mat frame, skinMap, foregroundMap;
    BlobGetter blobGetter = BlobGetter(TIMEDISPERSION);
    BlobProcessor blobProcessor = BlobProcessor();
    Mode mode = SKIN;
    while (cap.read(frame)){
        blobGetter.Process(frame, skinMap, foregroundMap);
        blobProcessor.Process(skinMap, frame);
        //cv::imshow("Frame", frame);

        switch (mode){
            case SKIN:
                //cv::imshow("Segmentation", skinMap);
                break;
            case FOREGROUND:
                //cv::imshow("Segmentation", foregroundMap);
                break;
            case COMBINATION:
                throw("Not implemented yet");
        }

        int key = cv::waitKey(1);
        switch (key){
            case 27:
                return(0);
            case (int) 's':
                mode = SKIN;
                std::cout << "Selected mode = SKIN" << std::endl;
                break;
            case (int) 'f':
                mode = FOREGROUND;
                std::cout << "Selected mode = FOREGROUND" << std::endl;
                break;
        }
    }
    return 0;
}

