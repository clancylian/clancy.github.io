---
title: opencv GPU和CPUX实现直方图均衡差异
date: 2019-04-29 11:27:13
tags: opencv
categories: opencv
---

opencv gpu和cpu实现直方图均衡会出现不同，版本3.3.1，代码如下：

```c++
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

void testCPU()
{
    //cpu
    cv::Mat src = imread("/home/ubuntu/Pictures/2.png");
    //cv::resize(src, src, cv::Size(0, 0), 0.5, 0.5);
    vector<Mat> channels;
    split(src, channels);
    Mat B,G,R;
//#pragma omp parallel sections
    {
//#pragma omp section
        {
            equalizeHist( channels[0], B );
            //GaussianBlur(B,B,Size(3, 3), 2.0);
            //B = (B+ channels[0]) / 2;
            //addWeighted(B, 0.5, channels[0], 0.5, 0, B);
        }
//#pragma omp section
        {
            equalizeHist( channels[1], G );
            //GaussianBlur(G,G,Size(3, 3), 2.0);
            //G = (G+ channels[1]) / 2;
            //addWeighted(G, 0.5, channels[1], 0.5, 0, G);
        }
//#pragma omp section
        {
            equalizeHist( channels[2], R );
            //GaussianBlur(R,R,Size(3, 3), 2.0);
            //R = (R+ channels[2]) / 2;
            //addWeighted(R, 0.5, channels[2], 0.5, 0, R);
        }
    }

    vector<Mat> combined;
    combined.push_back(B);
    combined.push_back(G);
    combined.push_back(R);
    Mat sample_single;
    merge(combined, sample_single);

    imwrite("cpu.jpg", sample_single);
    imshow("cpu", sample_single);
    //waitKey(0);
}

void testGPU()
{
    //gpu
    cv::Mat src = imread("/home/ubuntu/Pictures/2.png");
    //cv::resize(src, src, cv::Size(0, 0), 0.5, 0.5);
    cv::cuda::GpuMat gpu_src;// = cv::cuda::GpuMat(src.height, src.width, CV_8UC3, src.data);
    gpu_src.upload(src);

    std::vector<cv::cuda::GpuMat> channels;
    cv::cuda::split(gpu_src, channels);

    cv::cuda::GpuMat B, G, R;
    cv::cuda::equalizeHist(channels[0], B);
    cv::cuda::equalizeHist(channels[1], G);
    cv::cuda::equalizeHist(channels[2], R);

    //创建高斯滤波器
    cv::Ptr<cv::cuda::Filter> gauss = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, Size(3, 3), 2.0);
    //高斯滤波
    //gauss->apply(B, B);
    //gauss->apply(G, G);
    //gauss->apply(R, R);

//    cv::cuda::addWeighted(B, 0.5, channels[0], 0.5, 0, B);
//    cv::cuda::addWeighted(G, 0.5, channels[1], 0.5, 0, G);
//    cv::cuda::addWeighted(R, 0.5, channels[2], 0.5, 0, R);
    vector<cv::cuda::GpuMat> combined;
    combined.push_back(B);
    combined.push_back(G);
    combined.push_back(R);
    cv::cuda::GpuMat gpu_dst;
    cv::cuda::merge(combined, gpu_dst);

    Mat img;
    gpu_dst.download(img);
    imwrite("gpu.jpg", img);
    imshow("gpu", img);
    waitKey(0);

}

int main()
{
    testCPU();
    testGPU();
    
    return 0;
}
```

## 原图
![原图](https://raw.githubusercontent.com/clancylian/blogpic/master/2.png)
## CPU
![cpu](https://raw.githubusercontent.com/clancylian/blogpic/master/cpu.jpg)
## GPU
![gpu](https://raw.githubusercontent.com/clancylian/blogpic/master/gpu.jpg)