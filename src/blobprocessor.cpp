#include "blobprocessor.h"

BlobProcessor::BlobProcessor()
{
    deserializeFileToContours("open_palm.txt", trainContours);
}
std::vector<cv::Point> trainContours;
void BlobProcessor::Process(cv::Mat skinMap,cv::Mat foregroundMap, cv::Mat input)
{
    //foreground map is null for the moment
    cv::Mat dispMap;
    cv::cvtColor(input, dispMap, CV_BGR2GRAY);
    DispMap(dispMap, dispMap, 3);
    cv::imshow("disp", dispMap);
    //std::vector<cv::Rect> rects;
    //getRects(skinMap, rects);
   /* std::vector<std::vector<cv::Point> > contours;
    cv::findContours(dispMap, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    size_t minContourIndex = -1;
    double min = 10e30;
    for (size_t i = 0; i < contours.size(); ++i) {
        if (cv::contourArea(contours.at(i)) > 100) {
            std::vector<cv::Point> approx;
            cv::approxPolyDP(contours.at(i), approx, 0.03, false);
            double sum = 0;
            for (size_t a = 0; a < trainContours.size(); ++a) {
                sum += cv::matchShapes(approx, trainContours.at(a), CV_CONTOURS_MATCH_I2, 0);
            }
            sum = sum / trainContours.size();
            if (sum < min) {
                min = sum;
                minContourIndex = i;
            }
        }
    } */

    std::vector<std::vector<cv::Point> > cont;
    getContours(skinMap, cont);
    std::vector<cv::Point> hull;
    for (size_t i = 0; i < cont.size(); ++i) {
        cv::convexHull(cont.at(i), hull, false, true);
        for (size_t a = 0; a < hull.size(); ++a) {
            cv::circle(input, hull.at(a), 3, cv::Scalar(255, 0, 0));
        }
    }

    //cv::rectangle(input, cv::boundingRect(contours.at(minContourIndex)), cv::Scalar(255, 0, 0), 10);
    //cv::imshow("output", input);
    cv::imshow("input", input);
    /*
    std::vector<cv::Rect> rects;
    std::vector<std::vector<cv::Point> > contours;
    getContours(skinMap, contours);
    cv::Mat buff;
    int key = 0;
    while(key != 27 || key == (int)'c') {
        for (size_t i = 0; i < contours.size(); ++i) {
            if (cv::contourArea(contours.at(i)) > 100) {
                skinMap.copyTo(buff);
                cv::rectangle(buff, cv::boundingRect(contours.at(i)), cv::Scalar(255));
                cv::imshow("pick rectangle", buff);
                key = cv::waitKey(0);
                if (key == (int)'c') {
                    serializeContour("hand.txt", contours.at(i));
                    break;
                }
                if (key == 27) {
                    break;
                }
            }
        }
    }
    */
}

void BlobProcessor::deserializeFileToContours(std::string filename, std::vector<std::vector<cv::Point> > &contours)
{
    std::ifstream file (filename.c_str());
    std::string line;
    if (file.is_open()) {
        while (file.good()) {
            std::getline(file, line);
            std::vector<cv::Point> contour;
            std::string buf = "";
            cv::Point point = cv::Point2i(0, 0);
            for (size_t i = 0; i < line.size(); ++i) {
                if (line[i] != ':' && line[i] != ';' && line[i] != ',')
                    buf += line[i];
                else if (line[i] == ';')
                    contours.push_back(contour);
                else if (line[i] == ':') {
                    point.x = atoi(buf.c_str());
                    buf = "";
                }
                else if (line[i] == ',') {
                    point.y = atoi(buf.c_str());
                    buf = "";
                    contour.push_back(point);
                }
            }
        }
    }
    file.close();
}

void BlobProcessor::serializeContour(std::string filename, std::vector<cv::Point> contour)
{
    std::ofstream file;
    file.open(filename.c_str(), std::ios_base::app);
    for (size_t i = 0; i < contour.size(); ++i) {
        file << contour.at(i).x << "," << contour.at(i).y << "\n";
        /*if (i == contour.size() - 1)
            file << ";\n";
        else
            file << ",";
            */
    }
    file.flush();
    file.close();
}

std::vector<cv::Point> BlobProcessor::getMaxContour(std::vector<std::vector<cv::Point> > contours) {
    int max = 0;
    size_t max_i = 0;
    for (size_t i = 0; i < contours.size(); ++i) {
        if (cv::contourArea(contours.at(i)) > max) {
            max = cv::contourArea(contours.at(i));
            max_i = i;
        }
    }
    return contours.at(max_i);
}

std::vector<cv::Point> BlobProcessor::getCommonContour(std::vector<std::vector<cv::Point> > bigger, std::vector<std::vector<cv::Point> > smaller)
{
    if (smaller.size() == 0)
        return getMaxContour(bigger);
    if (bigger.size() == 0)
        return getMaxContour(smaller);
    size_t smallestI = 0, smallestJ = 0;
    double min = 10e30;
    for (size_t i = 0; i < bigger.size(); ++i) {
        for (size_t j = 0; j < smaller.size(); ++j) {
            double similarity = cv::matchShapes(bigger.at(i), smaller.at(j), CV_CONTOURS_MATCH_I3, 0);
            if (similarity < min) {
                min = similarity;
                smallestI = i;
                smallestJ = j;
            }
        }
    }
    //std::cout << "smallest similarity = " << min << std::endl;
    //std::cout << "i = " << smallestI << " j = " << smallestJ << std::endl;
    if (cv::contourArea(bigger.at(smallestI)) >= cv::contourArea(smaller.at(smallestJ)))
        return bigger.at(smallestI);
    else
        return smaller.at(smallestJ);
}

void BlobProcessor::getContoursFromRoi(cv::Mat dispInput, cv::Mat skinMap, cv::Rect rect, std::vector<cv::Point>& contour) {
    contour.clear();
    cv::Mat dispRoi;
    dispInput.copyTo(dispRoi);
    dispRoi = dispRoi(rect);

    cv::Mat skinRoi;
    skinMap.copyTo(skinRoi);
    skinRoi = skinRoi(rect);

    std::vector<std::vector<cv::Point> > skinC, dispC;
    cv::findContours(dispRoi, dispC, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    cv::findContours(skinRoi, skinC, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    if (skinC.size() != 0 && dispC.size() != 0) {
        if (skinC.size() >= dispC.size())
            contour = getCommonContour(skinC, dispC);
        else
            contour = getCommonContour(dispC, skinC);
    }
}

void BlobProcessor::trainClassifier(cv::Mat dispInput, cv::Mat skinMap, cv::Rect rect, std::vector<cv::Point>& contour) {

}

void BlobProcessor::getContours(cv::Mat input, std::vector<std::vector<cv::Point> > &contours) {
    cv::Mat buf;
    input.copyTo(buf);
    cv::findContours(buf, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
}

void BlobProcessor::getRects(cv::Mat input, std::vector<cv::Rect>& rects) {
    std::vector<std::vector<cv::Point> > contours = std::vector<std::vector<cv::Point> >();
    cv::Mat buf;
    input.copyTo(buf);
    cv::findContours(buf, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    for (size_t i = 0; i < contours.size(); ++i) {
        rects.push_back(cv::boundingRect(contours.at(i)));
    }
}

void BlobProcessor::getMask(cv::Mat input, cv::Mat& mask) {
    std::vector<cv::Rect> rects = std::vector<cv::Rect>();
    getRects(input, rects);
    mask = cv::Mat(input.size(), CV_8UC1);
    for (size_t i = 0; i < rects.size(); ++i) {
        cv::rectangle(mask, rects.at(i), cv::Scalar(255), -1);
    }
}

bool compareFunction(std::pair<int, int> a, std::pair<int, int> b) { return a.second < b.second; }

void BlobProcessor::GetGoodClusters(cv::Mat input, std::vector<cv::Rect> clusters, std::vector<cv::Rect>& goodClusters) {
    cv::Mat hsv;
    cv::cvtColor(input, hsv, CV_BGR2HSV);
    float PERCENTAGE = 0.4;
    int AREA_THRESHOLD = 100;
    std::vector<std::pair<int, int> > clustersValues;
    ColorFinder finder = ColorFinder();
    for (size_t i = 0; i < clusters.size(); ++i) {
        if (clusters.at(i).area() > AREA_THRESHOLD) {
            cv::Mat roi = hsv(clusters.at(i));
            std::pair<int, int> pair;
            pair.first = i;
            pair.second = finder.getIntervalsCount(roi, 90);
            clustersValues.push_back(pair);
        }
    }
    std::sort(clustersValues.begin(), clustersValues.end(), compareFunction);
    int goodClustersCount = clustersValues.size() * PERCENTAGE;
    int currentGoodClustersValue = 0;
    for (size_t i = 0; i < goodClustersCount; ++i) {
        currentGoodClustersValue += clustersValues.at(clustersValues.size() - i - 1).second;
    }
    if (goodClustersCount > 0) {
        currentGoodClustersValue /= goodClustersCount;
        for (size_t i = clustersValues.size() - 1; i > 0; --i) {
            if (clustersValues.at(i).second >= currentGoodClustersValue) {
                goodClusters.push_back(clusters.at(clustersValues.at(i).first));
            }
        }
    }
}

void BlobProcessor::RecognizeClusters(cv::Mat input, std::vector<cv::Rect> clusters) {
    cv::cvtColor(input, input, CV_BGR2GRAY);
    cv::Mat output;
    DispMap(input, output, 5);
    cv::imshow("contour", output);

}


void BlobProcessor::DispMap(cv::Mat input, cv::Mat &output, int threshold)
{
    input.convertTo(input, CV_32F);
    cv::Mat md, sqmd, mdsq, sq;
    cv::Mat k = (cv::Mat_<float>(3,3) << 1./9, 1./9, 1./9, 1./9, 1./9, 1./9, 1./9, 1./9, 1./9);
    cv::filter2D(input, md, CV_32F, k);
    cv::pow(input, 2, sq);
    cv::pow(md, 2, sqmd);
    cv::filter2D(sq, mdsq, CV_32F, k);
    output= mdsq - sqmd;
    output = (output >= threshold * threshold);
}


