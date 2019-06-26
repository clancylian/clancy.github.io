---
title: CUDA页锁定内存
date: 2019-06-06 11:07:43
tags:
- CUDA
categories: CUDA
top: 30
---

## 介绍

CUDA提供API可以来分配和使用页锁定主机内存(page-locked memory or pinned memory)，和传统使用malloc分配的可分页内存不同。

- 使用cudaHostAlloc()和cudaFreeHost()来分配和释放页锁定内存。
- 或者使用cudaHostRegister()将malloc()分配的内存注册为页锁定内存。

## 页锁定内存好处和缺点

### 好处

- 在页锁定内存和设备内存之前进行拷贝可以和内核函数并发执行，可参考异步拷贝。
- 在一些设备上，页锁定主机内存可以被映射到设备地址空间，不需要显式的执行拷贝操作。
- 在有前端总线的系统上，分配页锁定内存可以获得更高的带宽，如果加上write-combining标志的话，带宽会更高。

### 缺点

- 页锁定内存是稀缺资源，由于每个页锁定内存都需要分配物理内存，并且这些内存不能交换到磁盘上，分配过多将会减少操作系统可分页内存数量，降低系统性能。
- 在没有I/O一致性的Tegra设备上，页锁定内存是没有缓存的。比如之前在全景项目上使用页锁定内存，全景拼接速度反而更慢了。

## 零拷贝（zero-copy）

零拷贝其实是利用页锁定内存来实现的。

- 可分享存储器(Portable Memory)：默认情况下，分配页锁定内存只有在当前分配它的设备上所享有。为了让所有的设备享有这个好处，可以在使用cudaHostAlloc()时传入cudaHostAllocPortable标志，或者使用cudaHostRegister()时，传入cudaHostRegisterPortable标志。
- 写结合存储器(Write-Combining Memory)：默认情况下，页锁定主机内存是可缓存的。可以在使用cudaHostAlloc()分配页锁定内存时传入cudaHostAllocWriteCombined标志可以释放主机端的L1和L2缓存资源，使得其他应用程序可以使用更多的缓存。而且写结合存储器在通过PCI-e总线传输时不会被监视(snoop)，这能够获得高达40%的传输加速。从主机端读取写结合存储器极其慢，所以写结合存储器应当只用于那些主机只写的存储器。
- 映射内存(Mapped Memory)：通过使用cudaHostAlloc()或者cudaHostRegister()传入参数cudaHostAllocMapped标志，页锁定内存可以被映射到设备地址空间。此时，页锁定内存有两个地址：一个是主机内存上，主机指针从cudaHostAlloc()或malloc()返回，另一个在设备内存上，可以使用cudaHostGetDevicePointer()来获取，这个指针可以在核函数中使用来访问这块内存。唯一的例外就是主机和设备使用统一地址空间。

从内核中直接访问主机存储器有许多优点：

- 无须在设备上分配存储器，也不用在这块存储器和主机存储器间显式传输数据;数据传输是在内核需要的时候隐式进行的。
- 无须使用流重叠数据传输和内核执行;数据传输和内核执行自动重叠。

由于页锁定内存是在主机和设备间共享，应用必须使用流或者事件来同步内存访问，以避免潜在的读后写，写后读，或写后写危害。

为了能够检索到页锁定内存的设备指针，必须在调用任何CUDA运行时函数前调用cudaSetDeviceFlags()，并传入cudaDeviceMapHost标志。否则，cudaHostGetDevicePointer()将会返回错误。如果页锁定内存不支持映射的话，cudaHostGetDevicePointer()也会返回错误，使用之前，必须先确认硬件是否有这个特性。

**注意：从主机和其他设备的角度看，操作被映射分页锁定存储器的原子函数不是原子的。**



## 异步拷贝

CUDA将以下操作作为独立的任务，它们可以并发执行，至于并发执行的程度，需要看设备的硬件特性以及计算性能：

- 在主机端的计算
- 在设备端的计算
- 从主机拷贝内存到设备端
- 从设备拷贝内存到主机端
- 在设备内进行内存拷贝
- 在所有设备之间内存拷贝

### 主机和设备之间的并行执行

以下设备操作对于主机端来说是异步执行的：

- 启动内核函数
- 数据在设备内拷贝
- 内存从主机拷贝到设备的大小是64KB或者更小
- 调用后缀为Async的内存拷贝函数，如cudaMemcpyAsync()
- 调用内存设置函数，~~如cudaMemset()~~

注：在调试的时候，可以将CUDA_LAUNCH_BLOCKING设置为1把所有的异步关闭。当使用性能分析工具的时候(Nsight, Visual Profiler)一般是同步状态除非分析工具允许并发。当使用的主机内存不是页锁定的时候，异步拷贝将会被同步。

### 多个内核并行执行

一些计算性能为2.x或者更高的设备可以多个内核并发执行。具体可以使用如下方法查询concurrentKernels是否为1：

```c++
int deviceCount;
cudaGetDeviceCount(&deviceCount);
int device;
for (device = 0; device < deviceCount; ++device) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("Device %d has compute capability %d.%d.\n",
           device, deviceProp.major, deviceProp.minor);
}
```

两个不同CUDA context 的内核函数不能并发执行。如果内核函数使用大量的纹理内存或者使用大量局部内存，那么它和其他内核函数并行的可能性比较小。                       

### 数据传输和内核并行

一些设备可以执行异步内存拷贝和内核函数并行，具体需要查询asyncEngineCount看硬件是否有这个特性，像1080ti有2个拷贝引擎，如果拷贝涉及到主句内存，则必须是页锁定的内存。

在同一个设备之间内存拷贝可以和内核函数并行执行，如果concurrentKernels为1且asyncEngineCount>0的话。

###　数据传输之间的并行

如果拷贝引擎数量(大于)等于2的话，就可以并行多个拷贝。

### 流(Streams)

以上的所有异步并行都是通过流来管理的。同一个流是按顺序执行的，不同了流之间是乱序执行或者并行执行的。

```c++
//流的创建
cudaStream_t stream[2];
for (int i = 0; i < 2; ++i)
    cudaStreamCreate(&stream[i]);
float* hostPtr;
cudaMallocHost(&hostPtr, 2 * size);
//数据传输和内核并行执行
for (int i = 0; i < 2; ++i) {
    cudaMemcpyAsync(inputDevPtr + i * size, hostPtr + i * size,
                    size, cudaMemcpyHostToDevice, stream[i]);
    MyKernel <<<100, 512, 0, stream[i]>>>
          (outputDevPtr + i * size, inputDevPtr + i * size, size);
    cudaMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size,
                    size, cudaMemcpyDeviceToHost, stream[i]);
}
//流销毁
for (int i = 0; i < 2; ++i)
    cudaStreamDestroy(stream[i]);
```

为了防止调用cudaStreamDestroy()的时候设备还在工作，当调用销毁函数的时候，立即返回，并且和流相关的资源会在设备完成所有工作后自动释放。

当流设置成zero或者NULL的时候，或者不指定参数的时候，就会使用默认流，这时候就按顺序执行了。如果在编译的时候使用**--default-stream**标志或者在包含CUDA头文件之前定义宏**CUDA_API_PER_THREAD_DEFAULT_STREAM**，那么每个主机线程会有自己的默认流。           

显式同步流的方式有很多方法：cudaDeviceSynchronize() 会等待所有正在处理的stream完成；cudaStreamSynchronize()会等待指定的流完成，允许其他的流继续执行。cudaStreamWaitEvent()会等待指定的事件完成才会继续执行。cudaStreamQuery()可以查询流是否执行完。为了避免性能下降，同步函数最好在计时或者隔离失败的启动或内存拷贝。                                               

隐式同步：如果主机线程在来自不同流的两个命令之间发出以下任何一个操作，则它们不能并发运行：

- 页锁定主机内存的分配

- 设备内存的分配

- 设备内存设置（memset）

- 两个不同地址拷贝到相同设备地址

- 使用NULL流

流中的回调函数不能调用CUDA API否则会造成死锁；流可以设置优先级。

### 事件(Events)

```c++
//事件创建
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
//销毁
cudaEventDestroy(start);
cudaEventDestroy(stop);
//计时
cudaEventRecord(start, 0);
for (int i = 0; i < 2; ++i) {
    cudaMemcpyAsync(inputDev + i * size, inputHost + i * size,
                    size, cudaMemcpyHostToDevice, stream[i]);
    MyKernel<<<100, 512, 0, stream[i]>>>
               (outputDev + i * size, inputDev + i * size, size);
    cudaMemcpyAsync(outputHost + i * size, outputDev + i * size,
                    size, cudaMemcpyDeviceToHost, stream[i]);
}
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
float elapsedTime;
cudaEventElapsedTime(&elapsedTime, start, stop);
```

## 参考链接

### [Page-Locked Host Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#page-locked-host-memory)