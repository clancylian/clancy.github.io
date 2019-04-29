---
title: NVIDIA NPP LIBRARY SDK
date: 2019-04-29 10:44:47
tags:
- NPP
- CUDA
categories: NVIDIA
top: 13
---

# CUDA NPP库使用

NPP库是英伟达提供的可用在实现GPU加速图像处理，详细SDK文档可以[参考链接](https://docs.nvidia.com/cuda/npp/index.html)，主要包含的库如下：

```c++
//图像处理基础库，类似opencv core
nppc NPP core library which MUST be included when linking any application, functions are listed in nppCore.h
//算术逻辑操作
nppial  arithmetic and logical operation functions in nppi_arithmetic_and_logical_operations.h
//颜色转换操作
nppicc  color conversion and sampling functions in nppi_color_conversion.h
//图像压缩和解压
nppicom JPEG compression and decompression functions in nppi_compression_functions.h
//数据转换及初始化
nppidei data exchange and initialization functions in nppi_data_exchange_and_initialization.h
//滤波操作
nppif   filtering and computer vision functions in nppi_filter_functions.h
//几何变换
nppig   geometry transformation functions found in nppi_geometry_transforms.h
//形态学操作
nppim   morphological operation functions found in nppi_morphological_operations.h
//统计及线性变换
nppist  statistics and linear transform in nppi_statistics_functions.h and nppi_linear_transforms.h
//内存支持函数
nppisu  memory support functions in nppi_support_functions.h
//阈值及比较操作
nppitc  threshold and compare operation functions in nppi_threshold_and_compare_operations.h
```
由于项目需求，这里主要介绍一些常用的操作，主要是opencv中基本图像处理操作，比如颜色空间转换，图像伸缩变换等等。

## RESIZE

resize操作支持单通道、３通道、４通道。8u、16u、16s、32f，接口一般是*nppiResizeSqrPixel_　_　*，其中可以选择对感兴趣区域进行resize。这里需要注意的是resize的一些插值方式，和opencv不太一样，并且官方文档没有详细说明，导致有一些坑在里面。比如之前使用*NPPI_INTER_SUPER*插值方式的时候发现factor大于１的时候会出错。后面找到答案说*NPPI_INTER_SUPER*只支持降采样操作，[参考链接](https://devtalk.nvidia.com/default/topic/1043307/general/npp-library-functions-nppiresize_8u_c3r-and-nppibgrtolab_8u_c3r-differ-from-cv-resize-output/post/5302888/#5302888)。这里举个BGR进行通道转换的栗子：

```c++
bool imageResize_8u_C3R(void *src, int srcWidth, int srcHeight, void *dst, int dstWidth, int dstHeight)
{
    NppiSize oSrcSize;
    oSrcSize.width = srcWidth;
    oSrcSize.height = srcHeight;
    int nSrcStep = srcWidth * 3;

    NppiRect oSrcROI;
    oSrcROI.x = 0;
    oSrcROI.y = 0;
    oSrcROI.width = srcWidth;
    oSrcROI.height = srcHeight;

    int nDstStep = dstWidth * 3;
    NppiRect oDstROI;
    oDstROI.x = 0;
    oDstROI.y = 0;
    oDstROI.width = dstWidth;
    oDstROI.height = dstHeight;

    // Scale Factor
    double nXFactor = double(dstWidth) / (oSrcROI.width);
    double nYFactor = double(dstHeight) / (oSrcROI.height);

    // Scaled X/Y  Shift
    double nXShift = - oSrcROI.x * nXFactor ;
    double nYShift = - oSrcROI.y * nYFactor;
    int eInterpolation = NPPI_INTER_SUPER;
    if (nXFactor >= 1.f || nYFactor >= 1.f)
        eInterpolation = NPPI_INTER_LANCZOS;

    NppStatus ret = nppiResizeSqrPixel_8u_C3R((const Npp8u *)src, oSrcSize, nSrcStep, oSrcROI, (Npp8u *)dst,
                         nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation );
    if(ret != NPP_SUCCESS) {
        printf("imageResize_8u_C3R failed %d.\n", ret);
        return false;
    }

    return true;
}

```

resize库包含在**nppig**库里面，其中还有各种操作，包括mirror、remap、rotate、warp等等，这些在平常使用过程中比较少用到，需要用的时候再参考文档。

##　颜色转换

## 自己实现一些操作

### padding

```c++
__global__ void imagePaddingKernel(float3 *ptr, float3 *dst, int width, int height, int top,
                                   int bottom, int left, int right)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if(x < left || x >= (width - right) || y < top || y > (height - bottom)) {
        return;
    }

    float3 color = ptr[(y - top) * (width - top - right) + (x - left)];

    dst[y * width + x] = color;
}

void imagePadding(const void *src, void *dst, int width, int height, int top,
                  int bottom, int left, int right)
{
    int dstW = width + left + right;
    int dstH = height + top + bottom;

    cudaMemset(dst, 0, dstW * dstH * sizeof(float3));

    dim3 grids((dstW + 31) / 32, (dstH + 31) / 32);
    dim3 blocks(32, 32);
    imagePaddingKernel<<<grids, blocks>>>((float3 *)src, (float3 *)dst, dstW, dstH,
                                          top, bottom, left, right);
}
```

### split

```c++
__global__ void imageSplitKernel(float3 *ptr, float *dst, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) {
        return;
    }

    float3 color = ptr[y * width + x];

    dst[y * width + x] = color.x;
    dst[y * width + x + width * height] = color.y;
    dst[y * width + x + width * height * 2] = color.z;
}

void imageSplit(const void *src, float *dst, int width, int height, cudaStream_t stream)
{
    dim3 grids((width + 31) / 32, (height + 31) / 32);
    dim3 blocks(32, 32);
    imageSplitKernel<<<grids, blocks>>>((float3 *)src, (float *)dst, width, height);
}
```

### normalization

```c++
__global__ void imageNormalizationKernel(float3 *ptr, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) {
        return;
    }

    float3 color = ptr[y * width + x];
    color.x = (color.x - 127.5) * 0.0078125;
    color.y = (color.y - 127.5) * 0.0078125;
    color.z = (color.z - 127.5) * 0.0078125;

    ptr[y * width + x] = make_float3(color.x, color.y, color.z);
}

void imageNormalization(void *ptr, int width, int height, cudaStream_t stream)
{
    dim3 grids((width + 31) / 32, (height + 31) / 32);
    dim3 blocks(32, 32);
    imageNormalizationKernel<<<grids, blocks>>>((float3 *)ptr, width, height);
}
```

### BGR2RGBfloat

```c++
__global__ void convertBGR2RGBfloatKernel(uchar3 *src, float3 *dst, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) {
        return;
    }

    uchar3 color = src[y * width + x];
    dst[y * width + x] = make_float3(color.z, color.y, color.x);
}

void convertBGR2RGBfloat(void *src, void *dst, int width, int height, cudaStream_t stream)
{
    dim3 grids((width + 31) / 32, (height + 31) / 32);
    dim3 blocks(32, 32);
    convertBGR2RGBfloatKernel<<<grids, blocks>>>((uchar3 *)src, (float3 *)dst, width, height);
}
```

##　参考链接

[官网地址](https://developer.nvidia.com/npp)