#include "controller.h"
Controller::Controller()
{
    blobgetter = BlobGetter(TIMEDISPERSION);
    int i = open("model.xml", 0);
    if (i == -1) {
        close(i);
        std::cout << "model.xml doesn't exists, Training mode" << std::endl;
    } else {
        close(i);
        classifier = Classifier("model.xml");
    }
}

bool isGestureActive = false;
int threshold = 0;
int topthreshold = 0;
std::vector<cv::Mat> gesture;
MotionEstimator motionEstimator;
std::vector<std::vector<std::vector<double> > > saved;
std::vector<std::vector<std::vector<double> > > wrong;

void Controller::Process(cv::Mat frame)
{

    cv::Mat foregroundMat;
    blobgetter.getForegroundMap(frame, foregroundMat);
    if (foregroundMat.empty())
        return;

    if (threshold == 0) {
        threshold = static_cast<int>(frame.rows * frame.cols * 0.1);
        std::cout << "Threshold = " << threshold << std::endl;
    }
    if (topthreshold == 0) {
        topthreshold = static_cast<int>(frame.rows * frame.cols * 0.9);
        std::cout << "Top threshold = " << topthreshold << std::endl;
    }
    int count = cv::countNonZero(foregroundMat);
    if (count < topthreshold) {
        if (count > threshold && isGestureActive) {
            motionEstimator.AddFrame(foregroundMat);
            std::stringstream ss;
            ss << count;
            cv::putText(foregroundMat, "GESTURE", cv::Point(100, 20), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 0, 0));
        }
        if (count > threshold && !isGestureActive) {
            isGestureActive = true;
            std::cout << "START" << std::endl;
            motionEstimator.clear();
        } else if (count < threshold && isGestureActive) {
            isGestureActive = false;
            std::cout << "STOP" << std::endl;
            if (motionEstimator.GetFrameCount() >= 5) {
                if (classifier.IsTrained())
                    classify();
                else
                    labelAndTrain();
            }
        }
    }
    cv::imshow("TimeDisp", foregroundMat);
    checkKeys();
}

void Controller::classify() {
    time_t start, stop;
    start = clock();
    std::vector<std::vector<double> > histograms;
    motionEstimator.calculateMotionHistograms(histograms, false, false);
    stop = clock();
    std::cout << "Motion histograms calculated. Took " <<  stop - start << "  milliseconds" << std::endl;
    int label = classifier.Recognize(histograms);
    std::cout << "Recognized as: = " << label << std::endl;
    std::stringstream str;
    str << label;
    sendSocket(str.str());
}

void Controller::labelAndTrain() {
    std::cout << "Label motion..." << std::endl;
    int key = cv::waitKey(0);
    int label = 0;
    if (key == 27) {
        std::cout << "skipping.." << std::endl;
        return;
    }
    switch ((char) key) {
        case '0':
            label = 0;
            break;
        case '1':
            label = 1;
            break;
        case '2':
            label = 2;
            break;
        case '3':
            label = 3;
            break;
        case '4':
            label = 4;
            break;
        case '5':
            label = 5;
            break;
    }
    std::vector<std::vector<double> > histograms;

    motionEstimator.calculateMotionHistograms(histograms, false, false);
    classifier.AddToTrainSet(label, histograms);

}

void Controller::sendSocket(std::string payload) {
    SOCKET SendSocket = INVALID_SOCKET;
    WSADATA data;
    int iResult;
    sockaddr_in recvAddr;
    iResult = WSAStartup(0x101, &data);
    if (iResult != NO_ERROR) {
        std::cout << "Wsa startup failed " << iResult << std::endl;
    }

    SendSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (SendSocket == INVALID_SOCKET) {
        std::cout << "Invalid socket " << WSAGetLastError() << std::endl;
        WSACleanup();
        return;
    }
    recvAddr.sin_family = AF_INET;
    recvAddr.sin_port = htons(10000);
    recvAddr.sin_addr.s_addr = inet_addr("127.0.0.1");
    std::cout << "connecting socket" << std::endl;
    iResult = connect(SendSocket, (struct sockaddr*) &recvAddr, sizeof(recvAddr));
    if (iResult == SOCKET_ERROR) {
        if (WSAGetLastError() == 10061)
            std::cout << "Server is not running!" << std::endl;
        else
            std::cout << "Error while connecting socket " << WSAGetLastError() << std::endl;
        closesocket(SendSocket);
        WSACleanup();
        return;
    }

    std::cout << "sending data to socket..." << std::endl;
    iResult = sendto(SendSocket, payload.c_str(), payload.length(), 0, (struct sockaddr*) & recvAddr, sizeof(recvAddr));
    if (iResult == SOCKET_ERROR) {
        std::cout << "Error while sending data socket " << WSAGetLastError() << std::endl;
        closesocket(SendSocket);
        WSACleanup();
        return;
    }
    std::cout << "finishing sending..." << std::endl;
    iResult = closesocket(SendSocket);
    if (iResult == SOCKET_ERROR) {
        std::cout << "failed to close socket " << WSAGetLastError()  << std::endl;
        WSACleanup();
    }
    WSACleanup();

}

void Controller::checkKeys() {
    int key = cv::waitKey(1);
    switch ((char) key) {
    case 's':
        std::cout << "serializing..." << std::endl;
        classifier.serializeTrainingVectors("vectors.csv");
        break;
    case 't':
        std::cout << "training..." << std::endl;
        classifier.Train(true);
    default:
        break;
    }
 }
