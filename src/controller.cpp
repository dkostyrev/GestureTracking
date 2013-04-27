#include "controller.h"
cv::Point center = cv::Point(100, 100);
cv::RotatedRect lRect = cv::RotatedRect(cv::Point(0,0), cv::Size2f(0, 0), 0.0);
cv::RotatedRect rRect = cv::RotatedRect(cv::Point(0,0), cv::Size2f(0, 0), 0.0);
cv::Rect adaptRect = cv::Rect(0, 0, 200, 200);
Controller::Controller()
{
    handRegion = cv::Rect();
    handRegionHistory = std::vector<cv::Rect>();
    currentCandidates = std::vector<cv::Rect>();
    currentClassifier = PALM;
    blobgetter = BlobGetter(TIMEDISPERSION);
    //classifier = Classifier();
    //classifier = Classifier("vectors.csv");
}

void Controller::Approach1(cv::Mat frame)
{
    //checkKeys(frame, std::vector<std::vector<cv::Point> >());
    //blobgetter.AdaptColourThresholds(frame, adaptRect);
    cv::Mat skinMap, foregroundMap, mix;
    //cv::rectangle(frame, adaptRect, cv::Scalar(0, 0, 255), 3);

    blobgetter.Process(frame, skinMap, foregroundMap);
    if (!foregroundMap.empty())
        cv::imshow("movement", foregroundMap);
    if (!skinMap.empty())
        cv::imshow("skin", skinMap);
    if (skinMap.empty() || foregroundMap.empty())
        return;
    cv::bitwise_and(foregroundMap, skinMap, mix);
    cv::imshow("mix", mix);
    //fillRect(lRect, cv::Rect_<int>(0, 0, 200, 200), mix);
    //fillRect(rRect, cv::Rect_<int>(440, 0, 200, 200), mix);

    //cv::rectangle(frame, lRect.boundingRect(), cv::Scalar(0, 255, 0));
    //cv::ellipse(frame, lRect, cv::Scalar(255, 0, 0), 3);
    //cv::ellipse(frame, rRect, cv::Scalar(0, 255, 0), 3);
    /*std::vector<std::vector<cv::Point> > contours;
    blobprocessor.getContours(skinMap, contours);


    for (size_t i = 0; i < contours.size(); ++i) {
        std::vector<cv::Point> hull, defects;
        cv::Mat hullm;
        cv::convexHull(contours.at(i), hull, true, true);
        //cv::convexHull(contours.at(i), hullm, true, false);
        //cv::convexityDefects(contours.at(i), hullm, defects);
        for (size_t h = 0; h < hull.size(); ++h) {
            cv::circle(frame, hull.at(h), 4, cv::Scalar(0, 0, 255), 4);
        }
        //for (size_t h = 0; h < defects.size(); ++h) {
//            cv::circle(frame, hull.at(h), 4, cv::Scalar(0, 0, 255), 4);
//        }
    }

    cv::drawContours(frame, contours, -1, cv::Scalar(255, 0, 0), 4);*/
    cv::imshow("frame", frame);
}

bool isGestureActive = false;
int threshold = 0;
int topthreshold = 0;
std::vector<cv::Mat> gesture;
MotionEstimator motionEstimator;

void Controller::Approach2(cv::Mat frame) {
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
            motionEstimator = MotionEstimator();
        } else if (count < threshold && isGestureActive) {
            isGestureActive = false;
            std::cout << "STOP" << std::endl;
            //motionEstimator.ShowAllFrames();
            motionEstimator.calculateMotionHistograms();
            /*
            cv::Mat motion;
            motionEstimator.GetMotionMat(motion);
            if (!motion.empty()) {
                //cv::imshow("Motion", motion);
                cv::waitKey(0);
            }
            */
        }
    }
    cv::imshow("TimeDisp", foregroundMat);
}

void Controller::fillRect(cv::RotatedRect &rrect, cv::Rect startRect, cv::Mat mat) {
    cv::Rect rect;
    if (rrect.boundingRect().area() == 0)
        rect = startRect;
    else
        rect = rrect.boundingRect();
    cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria().MAX_ITER, 5, 0.0);
    rrect = cv::CamShift(mat, rect, criteria);
}

void Controller::Process(cv::Mat frame)
{
    Approach2(frame);
    return;
    //
    std::vector<cv::Rect> regions;
    if (handRegion.area() == 0)
        acquireHandRegion(frame, regions);
    blobprocessor.resizeRegions(regions, frame.size(),20, 20, currentCandidates);
    filterRois(currentCandidates, currentCandidates);

    std::vector<int> labels;
    if (!classifier.IsLoadedModel()) {
        for (size_t i = 0; i < currentCandidates.size(); ++i) {
            cv::rectangle(frame, currentCandidates.at(i), cv::Scalar(255, 0, 0), 4);
        }
    }

    std::vector<cv::Point> contour;
    std::vector<std::vector<cv::Point> > contours;
    for (size_t i = 0; i < currentCandidates.size(); ++i){
        //blobprocessor.growRegions(frame, currentCandidates.at(i), contour);
        contours.push_back(contour);
    }
    if (classifier.IsLoadedModel()) {
        classifyRegions(frame.size(), contours, labels);
    }
    for (size_t i = 0; i < labels.size(); ++i) {
        if (labels.at(i) > 0) {
            cv::Rect r = cv::boundingRect(contours.at(i));
            cv::rectangle(frame, r, cv::Scalar(255, 0, 0), 2);
            std::stringstream ss;
            ss << labels.at(i);
            cv::putText(frame, ss.str(), r.tl() - cv::Point(0, 20), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 0, 0));
        }
    }
    checkKeys(frame, contours);

    cv::drawContours(frame, contours, -1, cv::Scalar(0, 255, 0));
    cv::imshow("labels", frame);

}

void Controller::filterRois(std::vector<cv::Rect> input, std::vector<cv::Rect>& output) {
    std::vector<cv::Rect> buf = std::vector<cv::Rect>(input);
    std::vector<bool> parentess = std::vector<bool>(input.size(), false);
    for (size_t i = 0; i < input.size(); ++i) {
        cv::Rect ri = input.at(i);
        for (size_t a = 0; a < input.size(); ++a) {
            if (a == i)
                continue;
            cv::Rect r = input.at(a);
            cv::Point ap = r.tl();
            cv::Point bp = cv::Point(ap.x + r.width, ap.y);
            cv::Point cp = r.br();
            cv::Point dp = cv::Point(ap.x, ap.y + r.height);
            if (input.at(i).contains(ap) ||
                input.at(i).contains(bp) ||
                input.at(i).contains(cp) ||
                input.at(i).contains(dp)) {
                if (ri.area() > r.area()) {
                    parentess.at(i) = true;
                    parentess.at(a) = false;
                } else {
                    parentess.at(a) = true;
                    parentess.at(i) = false;
                }
            }
            else {
                parentess.at(i) = true;
            }
        }
    }
    output.clear();
    for (size_t i = 0; i < parentess.size(); ++i) {
        if (parentess.at(i)) {
            output.push_back(buf.at(i));
        }
    }

}

void Controller::checkKeys(cv::Mat frame, std::vector<std::vector<cv::Point> > contours) {
    int key = cv::waitKey(1);
    switch (key) {
    case (int) 't':
        if (contours.size() >  0) {
            std::cout << "Train procedure" << std::endl;
            blobprocessor.trainProcedure(classifier, frame, contours);
        }
        break;

    case (int) 'f':
        std::cout << "gonna follow" << std::endl;
        //follow = true;
        break;
    case (int) 'a':
        std::cout << "adapt!" << std::endl;
        blobgetter.AdaptColourThresholds(frame, adaptRect);
        break;
    case (int) 'r':
        std::cout << "Resetting" << std::endl;
        blobgetter.ResetColourThresholds();
        break;
    }


 }

bool Controller::ifRegionInHistory(cv::Mat frame, std::vector<cv::Rect> regions) {
    return false;
}

void Controller::classifyRegions(cv::Size matSize, std::vector<std::vector<cv::Point> > contours, std::vector<int> &classes) {
    for (size_t i = 0; i < contours.size(); ++i) {
        classes.push_back(classifier.Recognize(contours.at(i), matSize));
    }
}


MatchedClassifier Controller::acquireHandRegion(cv::Mat frame, std::vector<cv::Rect> &candidates) {
    switch (currentClassifier) {
    case PALM:
        std::cout << "PALM" << std::endl;
        cascadeClassifier = cv::CascadeClassifier(CASCADE_PATH_PALM);
        cascadeClassifier.detectMultiScale(frame, candidates, 1.2, 2, 0, cv::Size(20, 20));
        //if (candidates.size() == 0)
        //    currentClassifier = FIST;
        break;
    case FIST:
        std::cout << "FIST" << std::endl;
        cascadeClassifier = cv::CascadeClassifier(CASCADE_PATH_FIST);
        cascadeClassifier.detectMultiScale(frame, candidates, 1.1, 3, 0, cv::Size(20, 20));
        //if (candidates.size() == 0)
        //    currentClassifier = PALM;
        break;
    case FINGER:
        break;
    }
	return PALM;
}
