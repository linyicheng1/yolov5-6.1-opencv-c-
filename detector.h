#pragma once
#include <opencv2/opencv.hpp>
#include <utility>
#include "utils.h"
#include "openvino/openvino.hpp"
using namespace ov;
using namespace std;
class Detector_OV
{
public:
    explicit Detector_OV(std::nullptr_t) {};
    Detector_OV(const std::string& modelPath,
                const bool& isGPU,
                const cv::Size& inputSize);

    std::vector<Detection> detect(cv::Mat& image, const float& confThreshold, const float& iouThreshold);


private:
    Core ie;
    std::shared_ptr<ov::Model> network;
    CompiledModel executable_network;
    cv::Size2f inputImageShape;

    void printInputAndOutputsInfoShort(const ov::Model& network);
    void preprocessing(cv::Mat& image, float*& blob, std::vector<int64_t>& inputTensorShape);

    std::vector<Detection> postprocessing(const cv::Size& resizedImageShape,
                                          const cv::Size& originalImageShape,
                                          const ov::Tensor& outputTensors,
                                          const float& confThreshold, const float& iouThreshold);

    static void getBestClassInfo(std::vector<float>::iterator it, const int& numClasses,
                                 float& bestConf, int& bestClassId);

};

