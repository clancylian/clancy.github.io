---
title: opencv CUDA核函数学习
date: 2019-05-29 11:44:17
tags:
- CUDA
- OpenCV
categories: CUDA
top: 30
---

[https://github.com/opencv/opencv_contrib/blob/9735ec666c92f7853fd7f20468e7ac701d3a2df0/modules/cudev/include/opencv2/cudev/grid/detail/split_merge.hpp](https://github.com/opencv/opencv_contrib/blob/9735ec666c92f7853fd7f20468e7ac701d3a2df0/modules/cudev/include/opencv2/cudev/grid/detail/split_merge.hpp)

## 默认块大小设置

```c
//源码中块设置基本都是这个大小
struct DefaultSplitMergePolicy
{
    enum {
        block_size_x = 32,
        block_size_y = 8
    };
};
struct DefaultHistogramPolicy
{
    enum {
        block_size_x = 32,
        block_size_y = 8
    };
};
```



## split

```c++
template <class SrcPtr, typename DstType, class MaskPtr>
__global__ void split(const SrcPtr src, GlobPtr<DstType> dst1, GlobPtr<DstType> dst2, GlobPtr<DstType> dst3, const MaskPtr mask, const int rows, const int cols)
{
    typedef typename PtrTraits<SrcPtr>::value_type src_type;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows || !mask(y, x))
        return;
    
    const src_type src_value = src(y, x);
    dst1(y, x) = src_value.x;
    dst2(y, x) = src_value.y;
    dst3(y, x) = src_value.z;
}

template <class Policy, class SrcPtr, typename DstType, class MaskPtr>
__host__ void split(const SrcPtr& src, const GlobPtr<DstType>& dst1, const GlobPtr<DstType>& dst2, const GlobPtr<DstType>& dst3, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
{
    const dim3 block(Policy::block_size_x, Policy::block_size_y);
    const dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

    split<<<grid, block, 0, stream>>>(src, dst1, dst2, dst3, mask, rows, cols);
    CV_CUDEV_SAFE_CALL( cudaGetLastError() );

    //如果没有流同步，就做device同步
    if (stream == 0)
        CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
}

```



## merge

```c++
template <class Src1Ptr, class Src2Ptr, class Src3Ptr, typename DstType, class MaskPtr>
__global__ void mergeC3(const Src1Ptr src1, const Src2Ptr src2, const Src3Ptr src3, GlobPtr<DstType> dst, const MaskPtr mask, const int rows, const int cols)
{
        typedef typename VecTraits<DstType>::elem_type dst_elem_type;

        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= cols || y >= rows || !mask(y, x))
            return;

        dst(y, x) = VecTraits<DstType>::make(
                    saturate_cast<dst_elem_type>(src1(y, x)),
                    saturate_cast<dst_elem_type>(src2(y, x)),
                    saturate_cast<dst_elem_type>(src3(y, x))
                    );
}

template <class Policy, class Src1Ptr, class Src2Ptr, class Src3Ptr, typename DstType, class MaskPtr>
__host__ void mergeC3(const Src1Ptr& src1, const Src2Ptr& src2, const Src3Ptr& src3, const GlobPtr<DstType>& dst, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
{
        const dim3 block(Policy::block_size_x, Policy::block_size_y);
        const dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

        mergeC3<<<grid, block, 0, stream>>>(src1, src2, src3, dst, mask, rows, cols);
        CV_CUDEV_SAFE_CALL( cudaGetLastError() );

        if (stream == 0)
            CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
}
```



## copy

```c
template <class SrcPtr, typename DstType, class MaskPtr>
    __global__ void copy(const SrcPtr src, GlobPtr<DstType> dst, const MaskPtr mask, const int rows, const int cols)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= cols || y >= rows || !mask(y, x))
            return;

        dst(y, x) = saturate_cast<DstType>(src(y, x));
    }

    template <class Policy, class SrcPtr, typename DstType, class MaskPtr>
    __host__ void copy(const SrcPtr& src, const GlobPtr<DstType>& dst, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
    {
        const dim3 block(Policy::block_size_x, Policy::block_size_y);
        const dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

        copy<<<grid, block, 0, stream>>>(src, dst, mask, rows, cols);
        CV_CUDEV_SAFE_CALL( cudaGetLastError() );

        if (stream == 0)
            CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
}
```



## histogram －ＯpenCV代码有问题，可能是废弃的吧

```c
namespace cv { namespace cudev {

namespace grid_histogram_detail
{
	template <int BIN_COUNT, int BLOCK_SIZE, class SrcPtr, typename ResType, class MaskPtr>
    __global__ void histogram(const SrcPtr src, ResType* hist, const MaskPtr mask, const int rows, const int cols)
    {
    #if CV_CUDEV_ARCH >= 120
        //分配共享内存，大小为直方图bin个数，比如256
        __shared__ ResType smem[BIN_COUNT];
		 //to be fixed
        const int y = blockIdx.x * blockDim.y + threadIdx.y;
        //块内索引
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;

        //初始化为0
        for (int i = tid; i < BIN_COUNT; i += BLOCK_SIZE)
            smem[i] = 0;
        
        __syncthreads();

        if (y < rows)
        {
             //to be fixed
            for (int x = threadIdx.x; x < cols; x += blockDim.x)
            {
                if (mask(y, x))
                {
                    const uint data = src(y, x);
                    //to be fixed
                    atomicAdd(&smem[data % BIN_COUNT], 1);
                }
            }
        }

        __syncthreads();
		//所有块相加
        for (int i = tid; i < BIN_COUNT; i += BLOCK_SIZE)
        {
            const ResType histVal = smem[i];
            if (histVal > 0)
                atomicAdd(hist + i, histVal);
        }
    #endif
    }

    template <int BIN_COUNT, class Policy, class SrcPtr, typename ResType, class MaskPtr>
    __host__ void histogram(const SrcPtr& src, ResType* hist, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
    {
        const dim3 block(Policy::block_size_x, Policy::block_size_y);
        const dim3 grid(divUp(rows, block.y));

        const int BLOCK_SIZE = Policy::block_size_x * Policy::block_size_y;

        histogram<BIN_COUNT, BLOCK_SIZE><<<grid, block, 0, stream>>>(src, hist, mask, rows, cols);
        CV_CUDEV_SAFE_CALL( cudaGetLastError() );

        if (stream == 0)
            CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
    }
}

}}
```

### 自己实现

```c
__global__ void histogramKernel(int bin, int BLOCK_SIZE, uchar *src, int *dst, int width, int height)
{
    __shared__ int smem[bin];

    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    for (int i = tid; i < bin; i += BLOCK_SIZE){
        smem[i] = 0;
    }

    __syncthreads();

    if (y < height)
    {
        //此处要加限制，不然会重复计算
        if(blockIdx.x == 0)
        {
            //每个线程块处理多个直方图，减少写内存带宽
            for (int x = threadIdx.x; x < width; x += blockDim.x)
            {
                const uchar data = src[y * width + x];
                //这边没统计255的值，出来和OpenCV CPU版本结果一模一样
                //如果统计进去会最后一个bin变大，注意这里需要看内存有没有
                //越界，可以smem多开辟一个空间smem[bin+1]
                atomicAdd(&smem[data * bin / 0xff], 1);
            }
        }

    }

    __syncthreads();

    for (int i = tid; i < bin; i += BLOCK_SIZE)
    {
        const int histVal = smem[i];
        if (histVal > 0)
            atomicAdd(dst + i, histVal);
    }
}

void histogram(int bin, void *src, void *dst, int width, int height, cudaStream_t stream)
{
    dim3 grids((width + 15) / 16, (height + 15) / 16);
    dim3 blocks(16, 16);
    int blocksize = 16*16;
    histogramKernel<<<grids, blocks>>>(bin, blocksize, (uchar *)src, (int *)dst, width, height);
}

```

