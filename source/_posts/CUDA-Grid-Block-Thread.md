---
title: CUDA Grid Block Thread
date: 2019-05-28 16:37:33
tags:
- CUDA
- GPU
categories: CUDA
top: 30
---

## GPU性能

如下所示，是调用NVIDIA_CUDA-10.1_Samples/1_Utilities/deviceQuery查询的GPU性能。

```bash
ubuntu@ubuntu-B250-HD3:~/NVIDIA_CUDA-10.1_Samples/1_Utilities/deviceQuery$ ./deviceQuery 
./deviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "GeForce GTX 1080 Ti"
  CUDA Driver Version / Runtime Version          10.2 / 10.1
  CUDA Capability Major/Minor version number:    6.1
  Total amount of global memory:                 11177 MBytes (11720130560 bytes)
  (28) Multiprocessors, (128) CUDA Cores/MP:     3584 CUDA Cores
  GPU Max Clock rate:                            1645 MHz (1.64 GHz)
  Memory Clock rate:                             5505 Mhz
  Memory Bus Width:                              352-bit
  L2 Cache Size:                                 2883584 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 10.2, CUDA Runtime Version = 10.1, NumDevs = 1
Result = PASS

```



首先要明确几个概念：

## 硬件

**SP**：最基本的处理单元，Streaming Processor，也称为CUDA Core。具体的指令和任务都是在SP上处理的。GPU进行并行计算，也就是很多个SP同时做处理。 从上表可以看出，1080Ti显卡一共有3584 CUDA Cores。

**SM**：多个SP加上其他的一些资源组成一个Streaming Multiprocessor，也叫流处理簇。其他资源如：warp scheduler，register，shared memory/L1Cache，Load/Store Units等。SM可以看做GPU的心脏（对比CPU核心），register和shared memory是SM的稀缺资源。CUDA将这些资源分配给所有驻留在SM中的threads。因此，这些有限的资源就使每个SM中active warps有非常严格的限制，也就限制了并行能力。从上表可以看出，1080Ti显卡一共有28个SM，每个SM有128个SP，所以共有3584个SP。

需要指出，每个SM包含的SP数量依据GPU架构而不同，Fermi架构GF100是32个，GF10X是48个，Kepler架构都是192个，Maxwell都是128个。相同架构的GPU包含的SM数量则根据GPU的中高低端来定。

## 软件

grid，block，thread，warp是CUDA编程上的概念，以方便程序员软件设计，组织线程，同样的我们给出一个示意图来表示。

- thread(线程)：一个CUDA的并行程序会被以许多个threads来执行。
- block(线程块)：数个threads会被群组成一个block，同一个block中的threads可以同步，也可以通过shared memory通信。
- grid(线程网格)：多个blocks则会再构成grid。
- warp(线程束)：GPU执行程序时的调度单位，目前CUDA的warp的大小为32，同在一个warp的线程，以不同数据资源执行相同的指令，这就是所谓 SIMT。

## 执行状态

CUDA 的 device 实际在执行的时候，会以 Block 为单位，把一个个的 block 分配给 SM 进行运算；而 block 中的thread，又会以warp为单位，把 thread 来做分组计算。目前 CUDA 的 warp 大小都是 32，也就是 32 个 thread 会被群组成一个 warp 来一起执行;同一个 warp 里的 thread，会以不同的数据，执行同样的指令。

基本上 warp 分组的动作是由 SM 自动进行的，会以连续的方式来做分组。比如说如果有一个 block 里有 128 个 thread 的话，就会被分成四组 warp，第 0-31 个 thread 会是 warp 1、32-63 是 warp 2、64-95 是 warp 3、96-127 是 warp 4。而如果 block 里面的 thread 数量不是 32 的倍数，那他会把剩下的 thread 独立成一个 warp;比如说 thread 数目是 66 的话，就会有三个 warp：0-31、32-63、64-65。由于最后一个 warp 里只剩下两个 thread，所以其实在计算时，就相当于浪费了 30 个 thread 的计算能力。这点是在设定 block 中 thread 数量一定要注意的事!

~~一个 SM 一次只会执行一个 block 里的一个 warp，但是 SM 不见得会一次就把这个 warp 的所有指令都执行完；当遇到正在执行的 warp 需要等待的时候(例如存取 global memory 就会要等好一段时间)，就切换到别的 warp 来继续做运算，藉此避免为了等待而浪费时间。所以理论上效率最好的状况，就是在 SM 中有够多的 warp 可以切换，让在执行的时候，不会有「所有 warp 都要等待」的情形发生;因为当所有的 warp 都要等待时，就会变成 SM 无事可做的状况了。~~

~~实际上，warp 也是 CUDA 中，每一个 SM 执行的最小单位；如果 GPU 有 16 个 SM 的话，也就代表他真正在执行的thread数目会是 32*16 个(resident thread)。~~不过由于 CUDA 是要透过 warp 的切换来隐藏 thread 的延迟、等待，来达到大量平行化的目的，所以会用所谓的 active thread 这个名词来代表一个 SM 里同时可以处理的 thread 数目。 active warp是指已经分配给SM的warp，并且该warp需要的资源（寄存器）也已经分配。

而在 block 的方面，一个 SM 可以同时处理多个 thread block，当其中有 block 的所有 thread 都处理完后，他就会再去找其他还没处理的 block 来处理。假设有 16 个 SM、64 个 block、每个 SM 可以同时处理三个 block 的话，那一开始执时，device 就会同时处理 48 个 block，而剩下的 16 个 block 则会等 SM 有处理完 block 后，再进到 SM 中处理，直到所有 block 都处理结束。

为一个SM指定了一个或多个要执行的线程块时，它会将其分成warp块，并由SIMT单元进行调度。将块分割为warp的方法总是相同的，每个warp都包含连续的线程，递增线程索引，第一个warp中包含全局线程过索引0-31。每发出一条指令时，SIMT单元都会选择一个已准备好执行的warp块，并将指令发送到该warp块的活动线程。Warp块每次执行一条通用指令，因此在warp块的全部32个线程执行同一条路径时，可达到最高效率。如果一个warp块的线程通过独立于数据的条件分支而分散，warp块将连续执行所使用的各分支路径，而禁用未在此路径上的线程，完成所有路径时，线程重新汇聚到同一执行路径下，其执行时间为各时间总和。分支仅在warp块内出现，不同的warp块总是独立执行的--无论它们执行的是通用的代码路径还是彼此无关的代码路径。



## 总结

- 一个SM可以同时处理多个线程块，warp是SM最小执行单元。比如一个SM可以同时处理3个线程块，每个线程块有256个线程，那么就有3*256/32=24warp，同一时刻SM只能执行一个warp。注意只有一个block全部warp执行完才会换其它block来执行，同一个block的所有线程必定在同一个SM执行。
- 注意区分active warp和resident thread概念，active warp不一定在SM执行，而是分配好资源，等待SM调度。一个SM不一定要全部执行完，比如访存的时候可以换入其它warp来计算。这样就可以隐藏thread延迟、等待。
- 线程块数量一般分配为SM个数的8倍。
- 一个线程块分配的线程数是32的倍数。
- 线程块和线程网格都是三维索引。
- 尽量减少线程束分歧产生。
- 注意上表每个SM最大支持的线程数，以及每个线程块最大线程数。

## 问题

每个SM可以同时执行多少个block?

充分利用资源的话每个SM可以同时有多少个warp ?

block合理设计？



##　参考链接

[https://blog.csdn.net/junparadox/article/details/50540602](https://blog.csdn.net/junparadox/article/details/50540602)

[https://blog.csdn.net/yu132563/article/details/52548913](https://blog.csdn.net/yu132563/article/details/52548913)

