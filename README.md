# yolov5-6.1-opencv-cpp

适配于当前2022-6-20 YoloV5最新版本Release6.1的OpenVINO加速的C++代码。

- 在YoloV5最新的更新版本中导出的IR模型版本为11，并且不能通过直接修改参数变成老版本IR10
- openVINO 推出了api2.0 版本，和原始版本有一定差异，其中api2.0支持IR11,而原始版本不支持

因此，部分较为老版本的推理代码均无法直接使用，本项目则基于openVINO api2.0 实现了YOLOv5的推理部署过程，支持cpu推理和集显GPU推理。

![](result.jpg)

## 依赖项
- openVINO 2022.1 (老版本不支持api2.0)
- opencl (可选，集显加速)
- opencv

## 环境配置

### openvion 安装

教程目录：https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_from_archive_linux.html

1. 下载[安装包](安装包 )
![20221017200145.png](http://www.static.linyicheng.com.cn/articles/43a160c21195aee05549aa8661e000c9.png)

2. 解压，添加环境变量
```
tar zxvf xxxxx.tgz 
./xxxxx/setupvars.sh
```

### opencl 安装 

教程目录：
https://support.zivid.com/en/v2.5/getting-started/software-installation/gpu/install-opencl-drivers-ubuntu.html

1. 下载安装包 
```
wget https://github.com/intel/compute-runtime/releases/download/19.07.12410/intel-gmmlib_18.4.1_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/19.07.12410/intel-igc-core_18.50.1270_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/19.07.12410/intel-igc-opencl_18.50.1270_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/19.07.12410/intel-opencl_19.07.12410_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/19.07.12410/intel-ocloc_19.07.12410_amd64.deb
```
2. 安装
```
sudo dpkg -i *.deb
sudo apt install -y clinfo
```
3. 查看版本
```
/usr/bin/clinfo -l
```

## 代码下载 

[github](https://github.com/linyicheng1/yolov5-6.1-opencv-c-)

### 修改代码中的路径 `main.cpp`
```c++
Detector_OV detector{nullptr};
detector = Detector_OV("/home/hy/openvino/yolov5_cpp_openvino-master/demo/res/yolov5n.xml", true, cv::Size(640,640));
const float confThreshold = 0.3f;
const float iouThreshold = 0.4f;
cv::Mat image = cv::imread("../person.jpg");
```

## 运行
![image.png](http://www.static.linyicheng.com.cn/articles/790b7c8daee30a02cd474d211f417b63.png)

