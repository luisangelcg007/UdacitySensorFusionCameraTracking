/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

bool isValidDescriptorDetectorCombo(const std::string descriptor, const std::string detector) 
{
    return !((descriptor.compare("AKAZE") == 0 && detector.compare("AKAZE") != 0) ||
            (descriptor.compare("ORB") == 0 && detector.compare("SIFT") == 0));
}

std::vector<TimingInfo> initializeTimingInfoVector(void) {
    const std::vector<std::string> detectorTypes{ "SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT" };
    const std::vector<std::string> descriptorTypes{ "BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT" };
    const std::vector<std::string> matcherTypes{ "MAT_BF" };
    const std::vector<std::string> selectorTypes{ "SEL_KNN" };

    std::vector<TimingInfo> info;

    for (auto detectorType : detectorTypes) 
    {
        for (auto descriptorType : descriptorTypes) 
        {
            for (auto matcherType : matcherTypes) 
            {
                for (auto selectorType : selectorTypes) 
                {

                    if (!isValidDescriptorDetectorCombo(descriptorType, detectorType)) { continue; }

                    info.push_back(TimingInfo(detectorType, descriptorType, matcherType, selectorType));
                }
            }
        }
    }

    return info;
}

void createCSVOutputFile(std::vector<TimingInfo> &timingInfo) 
{
    constexpr char COMMA[]{ ", " };
    constexpr char csvName[]{ "../report/LuisAngelCabralGuzmanProject.csv" };

    std::cout << "Writing output file: " << csvName << std::endl;
    std::ofstream csvStream{ csvName };

    csvStream << "Name: Luis Angel Cabral Guzman" << std::endl << "Date: 2021-06-01" << std::endl << std::endl;

    csvStream << "COMBINATION ID." << COMMA;
    csvStream << "IMAGE NO." << COMMA;
    csvStream << "DETECTOR TYPE" << COMMA;
    csvStream << "DESCRIPTOR TYPE" << COMMA;
    csvStream << "TOTAL KEYPOINTS" << COMMA;
    csvStream << "KEYPOINTS ON VEHICLE" << COMMA;
    csvStream << "DETECTOR ELAPSED TIME" << COMMA;
    csvStream << "DESCRIPTOR ELAPSED TIME" << COMMA;
    csvStream << "MATCHED KEYPOINTS" << COMMA;
    csvStream << "MATCHER ELAPSED TIME";
    csvStream << std::endl;

    int indexID = 1;

    for (auto &info : timingInfo) 
    {
        
        for (int index = 0; index < 10; index++) 
        {
            csvStream << indexID << COMMA;
            csvStream << index << COMMA;
            csvStream << info.detectorType << COMMA;
            csvStream << info.descriptorType << COMMA;
            csvStream << info.ptsPerFrame.at(index) << COMMA;
            csvStream << info.pointsLeftOnImg.at(index) << COMMA;
            csvStream << info.detElapsedTime.at(index) << COMMA;
            csvStream << info.descElapsedTime.at(index) << COMMA;
            csvStream << info.matchedPts.at(index) << COMMA;
            csvStream << info.matchElapsedTime.at(index) << std::endl;
        }
        indexID++;
        csvStream << std::endl;
    }

    csvStream.close();
}

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */

     std::vector<TimingInfo> timingInfo = initializeTimingInfoVector() ;

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    /* MAIN LOOP OVER ALL IMAGES */

    for (auto &info : timingInfo) 
    {
        dataBuffer.clear();

        for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
        {
            /* LOAD IMAGE INTO BUFFER */
            CollectedData collectedData;

            std::cout << "Detector type:= " << info.detectorType << std::endl;
            std::cout << "Descriptor type:= " << info.descriptorType << std::endl;
            std::cout << "Matcher type:= " << info.matcherType << std::endl;
            std::cout << "Selector type:= " << info.selectorType << std::endl;
            std::cout << std::endl;

            // assemble filenames for current index
            ostringstream imgNumber;
            imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
            string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

            // load image from file and convert to grayscale
            cv::Mat img, imgGray;
            img = cv::imread(imgFullFilename);
            cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

            //// STUDENT ASSIGNMENT
            //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

            // push image into data frame buffer
            DataFrame frame;
            frame.cameraImg = imgGray;

            if(dataBuffer.size() == dataBufferSize)
            {
                dataBuffer.erase(begin(dataBuffer));
            }

            dataBuffer.push_back(frame);

            //// EOF STUDENT ASSIGNMENT
            cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;
            std::cout << "imgIndex := " << imgIndex << std::endl;

            /* DETECT IMAGE KEYPOINTS */

            // extract 2D keypoints from current image
            vector<cv::KeyPoint> keypoints; // create empty feature list for current image

            //// STUDENT ASSIGNMENT
            //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
            //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

            if (info.detectorType.compare("SHITOMASI") == 0)
            {
                collectedData = detKeypointsShiTomasi(keypoints, imgGray, false);
            }
            else  if (info.detectorType.compare("HARRIS") == 0) 
            {
                collectedData = detKeypointsHarris(keypoints, imgGray, false);

            } else 
            {
                collectedData = detKeypointsModern(keypoints, imgGray, info.detectorType, false);
            }

            info.ptsPerFrame.at(imgIndex) = collectedData.numKeyPoints;
            info.detElapsedTime.at(imgIndex) = collectedData.elapsedTime;
            //// EOF STUDENT ASSIGNMENT

            //// STUDENT ASSIGNMENT
            //// TASK MP.3 -> only keep keypoints on the preceding vehicle

            // only keep keypoints on the preceding vehicle
            bool bFocusOnVehicle = true;
            cv::Rect vehicleRect(535, 180, 180, 150);
            if (bFocusOnVehicle)
            {
                std::vector<cv::KeyPoint> retainedPoints;

                for (auto point : keypoints) 
                {
                    if (vehicleRect.contains(cv::Point2f(point.pt))) 
                    {
                        retainedPoints.push_back(point); 
                    }
                }

                keypoints = retainedPoints;

                info.pointsLeftOnImg.at(imgIndex) = keypoints.size();
                std::cout << std::endl;
            }

            //// EOF STUDENT ASSIGNMENT

            // optional : limit number of keypoints (helpful for debugging and learning)
            bool bLimitKpts = false;
            if (bLimitKpts)
            {
                int maxKeypoints = 50;

                if (info.detectorType.compare("SHITOMASI") == 0)
                { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                    keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
                }
                cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
                cout << " NOTE: Keypoints have been limited!" << endl;
            }

            // push keypoints and descriptor for current frame to end of data buffer
            (dataBuffer.end() - 1)->keypoints = keypoints;
            cout << "#2 : DETECT KEYPOINTS done" << endl;

            /* EXTRACT KEYPOINT DESCRIPTORS */

            //// STUDENT ASSIGNMENT
            //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
            //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

            cv::Mat descriptors;
            // descriptorType = "BRISK"; // BRIEF, ORB, FREAK, AKAZE, SIFT
            collectedData = descKeypoints((dataBuffer.end() - 1)->keypoints, 
                                          (dataBuffer.end() - 1)->cameraImg, 
                                           descriptors, 
                                           info.descriptorType);

            info.descElapsedTime.at(imgIndex) = collectedData.elapsedTime;
            //// EOF STUDENT ASSIGNMENT

            // push descriptors for current frame to end of data buffer
            (dataBuffer.end() - 1)->descriptors = descriptors;

            cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

            if (dataBuffer.size() > 1) // wait until at least two images have been processed
            {
                /* MATCH KEYPOINT DESCRIPTORS */

                vector<cv::DMatch> matches;
                const std::string descriptorFamily{ (info.descriptorType.compare("SIFT") == 0) ? "DES_HOG" : "DES_BINARY" };
                std::cout << "descriptorFamily = " << descriptorFamily << std::endl;
                std::cout << "descriptorType = " << info.descriptorType << std::endl;
                // string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
                // string descriptorType = "DES_BINARY"; // DES_BINARY, DES_HOG
                // string selectorType = "SEL_NN";       // SEL_NN, SEL_KNN

                //// STUDENT ASSIGNMENT
                //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
                //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

                collectedData = matchDescriptors((dataBuffer.end() - 2)->keypoints, 
                                                (dataBuffer.end() - 1)->keypoints,
                                                (dataBuffer.end() - 2)->descriptors, 
                                                (dataBuffer.end() - 1)->descriptors,
                                                matches, 
                                                descriptorFamily, 
                                                info.matcherType, 
                                                info.selectorType);

                info.matchedPts.at(imgIndex) = collectedData.numKeyPoints;
                info.matchElapsedTime.at(imgIndex) = collectedData.elapsedTime;

                //// EOF STUDENT ASSIGNMENT

                // store matches in current data frame
                (dataBuffer.end() - 1)->kptMatches = matches;

                cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

                // visualize matches between current and previous image
                bVis = false;
                if (bVis)
                {
                    cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                    cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                    (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                    matches, matchImg,
                                    cv::Scalar::all(-1), cv::Scalar::all(-1),
                                    vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                    string windowName = "Matching keypoints between two camera images";
                    cv::namedWindow(windowName, 7);
                    cv::imshow(windowName, matchImg);
                    cout << "Press key to continue to next image" << endl;
                    cv::waitKey(0); // wait for key to be pressed
                }
                bVis = false;
            }
            else 
            {
                info.matchedPts.at(imgIndex) = info.matchElapsedTime.at(imgIndex) = 0;
            }

        } // eof loop over all images
    }
    createCSVOutputFile(timingInfo);

    return 0;
}
