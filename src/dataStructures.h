#ifndef dataStructures_h
#define dataStructures_h

#include <vector>
#include <opencv2/core.hpp>


struct DataFrame { // represents the available sensor information at the same time instance
    
    cv::Mat cameraImg; // camera image
    
    std::vector<cv::KeyPoint> keypoints; // 2D keypoints within camera image
    cv::Mat descriptors; // keypoint descriptors
    std::vector<cv::DMatch> kptMatches; // keypoint matches between previous and current frame
};

struct TimeInformation {
    std::string detectorType;
    std::string descriptorType;
    std::string matcherType;
    std::string selectorType;

    std::array<int, 10> pointsPerFrame;
    std::array<int, 10> pointsLeftOnImage;
    std::array<int, 10> matchedPoints;
    std::array<double, 10> detectorElapsedTime;
    std::array<double, 10> descriptorElapsedTime;
    std::array<double, 10> matchElapsedTime;
};

struct CollectedData 
{
    int numKeyPoints;
    double elapsedTime;
};

#endif /* dataStructures_h */
