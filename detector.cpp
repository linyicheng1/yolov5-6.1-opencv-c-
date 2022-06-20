#include "detector.h"

Detector_OV::Detector_OV(const std::string& modelPath,
                         const bool& isGPU = true,
                         const cv::Size& inputSize = cv::Size(640, 640))
{
    vector<string> availableDevices = ie.get_available_devices();
    for (int i = 0; i < availableDevices.size(); i++) {
        printf("supported device name : %s \n", availableDevices[i].c_str());
    }

    network = ie.read_model(modelPath);

    this-> printInputAndOutputsInfoShort(*network);

    auto out_size = network->outputs().size();
    auto in_size = network->input().get_shape();


    ov::preprocess::PrePostProcessor ppp(network);

    ppp.input()
            .tensor()
            .set_element_type(ov::element::f32);

    ppp.input().model().set_layout("NCHW");

    network = ppp.build();


    // -------- Step 5. Loading a model to the device --------
    if (isGPU)
    {
        auto device_name = "GPU";
        executable_network = ie.compile_model(network, device_name);
    }
    else
    {
        executable_network = ie.compile_model(network, "CPU");
    }

    this->inputImageShape = cv::Size2f(inputSize);


}
void Detector_OV::getBestClassInfo(std::vector<float>::iterator it, const int& numClasses,
                                   float& bestConf, int& bestClassId)
{
    // first 5 element are box and obj confidence
    bestClassId = 5;
    bestConf = 0;

    for (int i = 5; i < numClasses + 5; i++)
    {
        if (it[i] > bestConf)
        {
            bestConf = it[i];
            bestClassId = i - 5;
        }
    }

}


void Detector_OV::printInputAndOutputsInfoShort(const ov::Model& network) {
    std::cout << "Network inputs:" << std::endl;
    for (auto&& input : network.inputs()) {
        std::cout << "    " << input.get_any_name() << " (node: " << input.get_node()->get_friendly_name()
                  << ") : " << input.get_element_type() << " / " << ov::layout::get_layout(input).to_string()
                  << std::endl;
    }

    std::cout << "Network outputs:" << std::endl;
    for (auto&& output : network.outputs()) {
        std::string out_name = "***NO_NAME***";
        std::string node_name = "***NO_NAME***";

        // Workaround for "tensor has no name" issue
        try {
            out_name = output.get_any_name();
        }
        catch (const ov::Exception&) {
        }
        try {
            node_name = output.get_node()->get_input_node_ptr(0)->get_friendly_name();
        }
        catch (const ov::Exception&) {
        }

        std::cout << "    " << out_name << " (node: " << node_name << ") : " << output.get_element_type() << " / "
                  << ov::layout::get_layout(output).to_string() << std::endl;
    }
}



void Detector_OV::preprocessing(cv::Mat& image, float*& blob, std::vector<int64_t>& inputTensorShape)
{
    cv::Mat resizedImage, floatImage;
    cv::cvtColor(image, resizedImage, cv::COLOR_BGR2RGB);
    utils:: letterbox(resizedImage, resizedImage, cv::Size(640, 640),
                      cv::Scalar(114, 114, 114), false,
                      false, true, 32);

    inputTensorShape[2] = resizedImage.rows;
    inputTensorShape[3] = resizedImage.cols;

    resizedImage.convertTo(floatImage, CV_32FC3, 1 / 255.0);
    blob = new float[floatImage.cols * floatImage.rows * floatImage.channels()];
    cv::Size floatImageSize{ floatImage.cols, floatImage.rows };

    // hwc -> chw
    std::vector<cv::Mat> chw(floatImage.channels());
    for (int i = 0; i < floatImage.channels(); ++i)
    {
        chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
    }
    cv::split(floatImage, chw);
}

std::vector<Detection> Detector_OV::postprocessing(const cv::Size& resizedImageShape,
                                                   const cv::Size& originalImageShape,
                                                   const ov::Tensor& outputTensors,
                                                   const float& confThreshold, const float& iouThreshold)
{
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;



    const ov::Shape outputShape = outputTensors.get_shape();
    auto* batchData = outputTensors.data<const float>();

    size_t count = outputTensors.get_size();
    std::vector<float> output(batchData, batchData + count);


    int numClasses = (int)outputShape[2] - 5;
    int elementsInBatch = (int)(outputShape[1] * outputShape[2]);


    for (auto it = output.begin(); it != output.begin() + elementsInBatch; it += outputShape[2])
    {
        float clsConf = it[4];

        if (clsConf > confThreshold)
        {
            int centerX = (int)(it[0]);
            int centerY = (int)(it[1]);
            int width = (int)(it[2]);
            int height = (int)(it[3]);
            int left = centerX - width / 2;
            int top = centerY - height / 2;

            float objConf;
            int classId;
            this->getBestClassInfo(it, numClasses, objConf, classId);

            float confidence = clsConf * objConf;

            boxes.emplace_back(left, top, width, height);
            confs.emplace_back(confidence);
            classIds.emplace_back(classId);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confs, confThreshold, iouThreshold, indices);

    std::vector<Detection> detections;

    for (int idx : indices)
    {
        Detection det;
        det.box = cv::Rect(boxes[idx]);
        utils::scaleCoords(resizedImageShape, det.box, originalImageShape);
        det.conf = confs[idx];
        det.classId = classIds[idx];
        detections.emplace_back(det);
    }

    return detections;
}




std::vector<Detection> Detector_OV::detect(cv::Mat& image, const float& confThreshold = 0.4,
                                           const float& iouThreshold = 0.45)
{

    ov::element::Type input_type = ov::element::f32;
    ov::Shape input_shape = { 1,3,640, 640 };

    float* blob = nullptr;

    std::vector<int64_t> inputTensorShape{ 1, 3, -1, -1 };
    this->preprocessing(image, blob, inputTensorShape);

    ov::Tensor input_tensor = ov::Tensor(input_type, input_shape, blob);// resize_img.ptr());

    const ov::Shape tensor_shape = input_tensor.get_shape();

    ov::InferRequest infer_request = executable_network.create_infer_request();

    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
    cv::Size resizedShape = cv::Size((int)inputTensorShape[3], (int)inputTensorShape[2]);
    std::vector<Detection> result = this->postprocessing(resizedShape,
                                                         image.size(),
                                                         infer_request.get_output_tensor(0),
                                                         confThreshold, iouThreshold);

    delete[] blob;

    return result;

}