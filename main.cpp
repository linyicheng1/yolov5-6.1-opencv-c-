#include "detector.h"
#include <chrono>

std::vector<std::string> load_class_list()
{
    std::vector<std::string> class_list;

    std::ifstream ifs("../classes.txt");
    std::string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
    return class_list;
}


int main(int argc, char const *argv[])
{
    Detector_OV detector{nullptr};
    detector = Detector_OV("/home/hy/openvino/yolov5_cpp_openvino-master/demo/res/yolov5n.xml", true, cv::Size(640,640));
    const float confThreshold = 0.3f;
    const float iouThreshold = 0.4f;
    cv::Mat image = cv::imread("../person.jpg");
    auto classNames = load_class_list();
    auto start = std::chrono::system_clock::now();
    std::vector<Detection> result_OV;
    for (int i = 0;i < 100;i ++)
    {
        result_OV = detector.detect(image, confThreshold, iouThreshold);
    }
    auto end = std::chrono::system_clock::now();
    std::cout<<"cost: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.f <<" s"<<std::endl;
    utils::visualizeDetection(image, result_OV, classNames, cv::Scalar(229, 21, 229));
    cv::imwrite("result.jpg", image);
    cv::imshow("result", image);
    cv::waitKey(0);
}
