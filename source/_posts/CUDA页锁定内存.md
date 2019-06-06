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
- 调用内存设置函数，如cudaMemset()

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

当流设置成zero或者NULL的时候，或者不指定参数的时候                                  

### [3.2.5.5.2. Default Stream](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#default-stream)

Kernel launches and host <-> device memory                                     copies that do not specify any stream parameter, or equivalently that set                                     the stream parameter to zero, are issued to the default stream. They are                                     therefore executed in order.                                  

​                                     For code that is compiled using the --default-stream per-thread compilation flag (or that defines the CUDA_API_PER_THREAD_DEFAULT_STREAM macro before including CUDA headers (cuda.h and cuda_runtime.h)), the default stream is a regular stream and                                     each host thread has its own default stream.                                                                       

For code that is compiled using the --default-stream legacy compilation flag, the default stream is a special stream called the NULL stream                                     and each device has a single NULL  stream used for all host threads. The NULL stream is special as it  causes implicit synchronization                                     as described in                                     [Implicit Synchronization](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#implicit-synchronization).                                                                       

For code that is compiled without specifying a --default-stream compilation flag, --default-stream legacy is assumed as the default.                                                                       

​                                  

### [3.2.5.5.3. Explicit Synchronization](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#explicit-synchronization)

There are various ways to explicitly synchronize streams with each other.

cudaDeviceSynchronize() waits until all preceding                                     commands in all streams of all host threads have completed.                                  

cudaStreamSynchronize()takes a stream as a parameter                                     and waits until all preceding commands in the given stream have                                     completed. It can be used to synchronize the host with a specific stream,                                     allowing other streams to continue executing on the device.                                  

cudaStreamWaitEvent()takes a stream and an event as                                     parameters (see [Events](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#events) for a description of events)and                                     makes all the commands added to the given stream after the call to                                     cudaStreamWaitEvent()delay their execution until the                                     given event has completed. The stream can be 0, in which case all the                                     commands added to any stream after the call to                                     cudaStreamWaitEvent()wait on the event.                                  

cudaStreamQuery()provides applications with a way to                                     know if all preceding commands in a stream have completed.                                  

To avoid unnecessary slowdowns, all these synchronization functions are                                     usually best used for timing purposes or to isolate a launch or memory                                     copy that is failing.                                  

​                                  

### [3.2.5.5.4. Implicit Synchronization](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#implicit-synchronization)

Two commands from different streams cannot run concurrently if any one of the following                                     operations is issued in-between them by the host thread:                                                                       

- a page-locked host memory allocation,
- a device memory allocation,
- a device memory set,
- a memory copy between two addresses to the same device memory,
- any CUDA command to the NULL stream,
- ​                                        a switch between the L1/shared memory configurations described in [Compute Capability 3.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-3-0) and [Compute Capability 7.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-7-x).                                                                             

For devices that support concurrent kernel execution and are of compute capability 3.0 or                                     lower, any operation that requires a dependency check to see if a streamed kernel launch                                     is complete:                                   

- Can start executing only when all thread blocks of all prior kernel launches from any stream                                        in the CUDA context have started executing;                                                                             
- Blocks all later kernel launches from any stream in the CUDA context until the kernel launch                                        being checked is complete.                                                                             

Operations that require a dependency check include any other commands within the same stream as                                     the launch being checked and any call to cudaStreamQuery()  on that                                     stream. Therefore, applications  should follow these guidelines to improve their potential for                                     concurrent kernel execution:                                                                       

- All independent operations should be issued before dependent operations,
- Synchronization of any kind should be delayed as long as possible.

​                                  

### [3.2.5.5.5. Overlapping Behavior](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#overlapping-behavior)

The amount of execution overlap between two streams depends on the order in which the commands are issued to each stream and                                     whether or not the device supports overlap of data transfer and kernel execution (see [Overlap of Data Transfer and Kernel Execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#overlap-of-data-transfer-and-kernel-execution)), concurrent kernel execution (see [Concurrent Kernel Execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#concurrent-kernel-execution)), and/or concurrent data transfers (see [Concurrent Data Transfers](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#concurrent-data-transfers)).                                                                       

For example, on devices that do not support concurrent data transfers, the two streams of the code sample of [Creation and Destruction](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#creation-and-destruction-streams)  do not overlap at all because the memory copy from host to device is  issued to stream[1] after the memory copy from device                                     to host is issued to stream[0], so  it can only start once the memory copy from device to host issued to  stream[0] has completed.                                     If the code is rewritten the  following way (and assuming the device supports overlap of data transfer  and kernel execution)                                                                       

```
for (int i = 0; i < 2; ++i)
    cudaMemcpyAsync(inputDevPtr + i * size, hostPtr + i * size,
                    size, cudaMemcpyHostToDevice, stream[i]);
for (int i = 0; i < 2; ++i)
    MyKernel<<<100, 512, 0, stream[i]>>>
          (outputDevPtr + i * size, inputDevPtr + i * size, size);
    for (int i = 0; i < 2; ++i)
    cudaMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size,
                    size, cudaMemcpyDeviceToHost, stream[i]);
```

then the memory copy from host to device issued to stream[1] overlaps with the kernel launch issued to stream[0].

On devices that do support concurrent data transfers, the two streams of the code sample                                     of [Creation and Destruction](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#creation-and-destruction-streams) do overlap: The memory copy from                                     host to device issued to stream[1] overlaps with the memory copy from device to host                                     issued to stream[0] and even with the kernel launch issued to stream[0] (assuming the                                     device supports overlap of data transfer and kernel execution). However, for devices of                                     compute capability 3.0 or lower, the kernel executions cannot possibly overlap because                                     the second kernel launch is issued to stream[1] after the memory copy from device to host                                     is issued to stream[0], so it is blocked until the first kernel launch issued to stream[0]                                     is complete as per [Implicit Synchronization](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#implicit-synchronization). If the code is rewritten                                     as above, the kernel executions overlap (assuming the device supports concurrent kernel                                     execution) since the second kernel launch is issued to stream[1] before the memory copy                                     from device to host is issued to stream[0]. In that case however, the memory copy from                                     device to host issued to stream[0] only overlaps with the last thread blocks of the                                     kernel launch issued to stream[1] as per [Implicit Synchronization](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#implicit-synchronization), which                                     can represent only a small portion of the total execution time of the kernel.                                   

​                                  

### [3.2.5.5.6. Callbacks](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-callbacks)

​                                     The runtime provides a way to insert a callback at any point into a stream via cudaStreamAddCallback().  A callback is a function that is executed on the host once all commands  issued to the stream before the callback have completed.                                     Callbacks in stream 0 are executed  once all preceding tasks and commands issued in all streams before the  callback have completed.                                                                       

​                                     The following code sample adds the callback function                                     MyCallback to each of two streams                                     after issuing a host-to-device memory copy, a kernel launch and a                                     device-to-host memory copy into each stream. The callback will                                     begin execution on the host after each of the device-to-host memory                                     copies completes.                                                                       

```
void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void *data){
    printf("Inside callback %d\n", (size_t)data);
}
...
for (size_t i = 0; i < 2; ++i) {
    cudaMemcpyAsync(devPtrIn[i], hostPtr[i], size, cudaMemcpyHostToDevice, stream[i]);
    MyKernel<<<100, 512, 0, stream[i]>>>(devPtrOut[i], devPtrIn[i], size);
    cudaMemcpyAsync(hostPtr[i], devPtrOut[i], size, cudaMemcpyDeviceToHost, stream[i]);
    cudaStreamAddCallback(stream[i], MyCallback, (void*)i, 0);
}
        
```

​                                     The commands that are issued in a  stream (or all commands issued to any stream if the callback is issued  to stream 0) after                                     a callback do not start executing  before the callback has completed.                                     The last parameter of cudaStreamAddCallback() is reserved for future use.                                                                       

​                                     A callback must not make CUDA API  calls (directly or indirectly), as it might end up waiting on itself if  it makes such a                                     call leading to a deadlock.                                                                        



​                                  

### [3.2.5.5.7. Stream Priorities](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-priorities)

 The relative priorities of streams can be specified at creation using cudaStreamCreateWithPriority(). The range of allowable priorities, ordered as [ highest priority, lowest priority ] can be obtained using the cudaDeviceGetStreamPriorityRange() function. At runtime, as blocks in low-priority schemes finish, waiting blocks in higher-priority streams are scheduled in                                     their place.                                     	                                  

 The following code sample obtains the allowable range of priorities for the current device, and creates streams with the                                     highest and lowest available priorities                                     	                                  

```
// get the range of stream priorities for this device
int priority_high, priority_low;
cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
// create streams with highest and lowest available priorities
cudaStream_t st_high, st_low;
cudaStreamCreateWithPriority(&st_high, cudaStreamNonBlocking, priority_high);
cudaStreamCreateWithPriority(&st_low, cudaStreamNonBlocking, priority_low);
```





​                                                                





## 统一虚拟地址(VUA)







## 参考链接

### [Page-Locked Host Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#page-locked-host-memory)