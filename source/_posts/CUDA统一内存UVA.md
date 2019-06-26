---
title: CUDA统一内存UVA
date: 2019-06-11 17:42:23
tags: CUDA
categories: CUDA
top: 30
---

## 介绍

设备是否支持统一内存可以通过一下代码查询：

```c++
//如果支持，unifiedAddressing字段为1
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

当应用程序是64位进程，并且主机和所有具有[计算能力2.0](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#unified-virtual-address-space)([附录说的是3.0以上](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd))及更高版本的设备都使用一个地址空间，此时通过CUDA API分配的所有主机内存和所有受支持设备上开辟的设备内存都在此虚拟地址范围内。

- 通过CUDA接口分配的主机内存或者任何使用统一地址空间的设备分配的设备内存都可以使用**cudaPointerGetAttributes()**的指针来确定地址。
- 使用统一内存进行数据拷贝不需要执行拷贝类型，只需使用**cudaMemcpyDefault**，这同样适用于不使用CUDA API分配的主机内存，只要设备使用的是统一地址。
- 如果使用统一地址空间，通过cudaHostAlloc()分配的页锁定主机内存会默认cudaHostAllocPortable，此时指针可以直接在内核函数使用，而无需像页锁定内存那样先通过cudaHostGetDevicePointer()函数获取设备指针。

```c++
    void *ptr;
    //分配页锁定内存默认cudaHostAllocPortable，指针可以直接在核函数使用
	cudaHostAlloc(&ptr, 1000, cudaHostAllocDefault);
    //分配统一内存
    cudaMallocManaged(&ptr, 1000);
    //host_ptr为malloc开辟的内存
    memcpy(ptr, host_ptr, 1000);
	//也可直接使用cudaMemcpy
    cudaMemcpy(ptr, host_ptr, 1000, cudaMemcpyDefault);
```



统一内存是CUDA编程模型的一个组成部分，CUDA 6.0首次介绍了该模型，它定义了一个托管(managed)内存空间，其中所有处理器(包括CPU和GPU)都可以看到共同的地址空间。 

通过让底层系统自己管理CUDA数据访问和位置，避免显式数据拷贝，这主要带来两方面好处：

- 简化编程
- 通过透明地将数据迁移到使用它的处理器，可以最大限度地提高数据访问速度。

简单来说就是统一内存消除了显式数据拷贝并且不会像zero-copy那样带来性能下降(页锁定内存分配过多性能会下降)，当然数据迁移仍然会发生，所以程序速度不会明显加快，不过可以简化代码编写和维护。

统一内存提供“单一指针”模型，这有点像zero-copy。主要不同点是零拷贝分配的内存是固定主机内存，程序的性能可能快也可能慢，这取决于从哪里访问。而统一内存分离内存和执行空间，所以数据访问很快。                      

统一内存是一套内存管理服务的系统，该系统的一部分定义了加入统一内存服务的托管内存(managed memory)空间，换句话说，managed memory只是统一内存的一部分。

统一内存可以像其他设备内存一样使用CUDA的任何操作，最主要的区别就是主机可以直接引用和访问统一内存。

统一内存不支持附加在Tegra上的离散GPU。

### 系统要求

- SM架构是3.0(Kepler)或者更高
- 64位程序并且是非嵌入式系统

如果SM的架构是6.x(Pascal)或者更高，统一内存有新功能，如按需页面迁移和GPU内存超量分配(oversubscription)，请注意，目前只有Linux操作系统支持这些功能。在Windows（无论是在TCC或WDDM模式下）或MacOS上运行的应用程序将使用基本的统一内存模型，就像在6.x之前的体系结构上一样，即使它们运行在具有6.x或更高计算能力的硬件上。

### 简化编程

```c++
__global__ void AplusB(int *ret, int a, int b) {
    ret[threadIdx.x] = a + b + threadIdx.x;
}
int main() {
    int *ret;
    //第一种分配方式
    cudaMallocManaged(&ret, 1000 * sizeof(int));
    AplusB<<< 1, 1000 >>>(ret, 10, 100);
    cudaDeviceSynchronize();
    for(int i = 0; i < 1000; i++)
        printf("%d: A+B = %d\n", i, ret[i]);
    cudaFree(ret); 
    return 0;
}
//第二种分配方式
__device__ __managed__ int ret[1000];
__global__ void AplusB(int a, int b) {
    ret[threadIdx.x] = a + b + threadIdx.x;
}
int main() {
    AplusB<<< 1, 1000 >>>(10, 100);
    cudaDeviceSynchronize();
    for(int i = 0; i < 1000; i++)
        printf("%d: A+B = %d\n", i, ret[i]);
    return 0;
}
```

以上代码未使用cudaMemcpy()，所以未使用到隐式同步，所以需要做显式同步cudaDeviceSynchronize()。

### 数据迁移和一致性

统一内存试图通过将数据迁移到正在访问它的设备来优化内存性能（即，如果CPU正在访问数据，则将数据移动到主机内存；如果GPU将访问数据，则将数据移动到设备内存）。**数据迁移是统一内存的基础**，但对程序是透明的。系统将尝试将数据放置在可以最有效地访问数据的位置，而不会违反一致性。

对于程序来说，数据的物理地址是不可见的，随时都可能变化，但是访问数据的虚拟地址是保持有效和一致。注意，在性能之前，**保持一致性是主要要求**；在主机操作系统的限制范围内，允许系统失败访问或移动数据，以保持处理器之间的全局一致性。

计算能力低于6.x的GPU体系结构不支持按需将托管数据细粒度移动到GPU。每当启动GPU内核时，通常必须将所有托管内存(managed memory)传输到GPU内存，以避免内存访问出错。随着计算能力6.x到来，引入了一种新的GPU页面错误机制，提供更无缝的统一内存功能。结合系统范围的虚拟地址空间，页面错误提供了几个好处。首先，页面错误意味着CUDA系统软件不需要在每个内核启动之前将所有托管内存分配同步到GPU。如果在GPU上运行的内核访问一个不在其内存中的页面，那么它会出错，允许该页面按需自动迁移到GPU内存。或者，可以将页面映射到GPU地址空间，以便通过PCIe或NVLink互连进行访问（访问时的映射有时比迁移更快）。注意，统一内存是系统范围的：GPU（和CPU）可以在内存页上发生故障并从CPU内存或系统中其他GPU的内存迁移内存页。

### 显存超量分配

计算能力低于6.x的设备无法分配比显存的物理大小更多的托管内存。具有计算能力6.x的设备扩展了寻址模式以支持49位虚拟寻址。它的大小足以覆盖现代CPU的48位虚拟地址空间，以及GPU显存。巨大的虚拟地址空间和页面错误功能使应用程序能够访问整个系统虚拟内存，而不受任何一个处理器的物理内存大小的限制。这意味着应用程序可以超额订阅内存系统：换句话说，它们可以分配、访问和共享大于系统总物理容量的数组，从而实现超大型数据集的核心外处理。只要有足够的系统内存可供分配，cudaMallocManaged就不会耗尽内存。 

### 多GPU设备

对于计算能力低于6.x的设备来说，managed memory分配的行为和其他非managed内存一样，内存分配实际物理地址是当前活动设备，其他GPU设备都接收相同的内存映射。这意味着其他GPU将通过PCIe总线以较低的带宽访问内存。如果系统中的GPU之间不支持对等映射，那么托管内存页将放置在CPU系统内存（“零拷贝”内存）中，并且所有GPU都将遇到PCIe带宽限制。

具有计算能力6.x设备的系统上的托管分配对所有GPU都可见，可以按需迁移到任何处理器。

### ~~系统分配器~~<!--此部分特性基本用不到，暂不需要了解-->

~~计算能力7.0的设备支持NVLink上的地址转换服务（ATS）。ATS允许GPU直接访问CPU的页表。GPU MMU中的丢失将导致对CPU的地址转换请求（ATR）。CPU在其页表中查找该地址的虚拟到物理映射，并将转换返回到GPU。ATS提供对系统内存的GPU完全访问，例如分配malloc的内存、分配在堆栈上的内存、全局变量和文件备份内存。应用程序可以通过检查pageablememoryacessuseshostpagetables属性来查询设备是否支持通过ATS一致地访问可分页内存。~~

```c++
//前面介绍两种分配Managed内存方式，
int *data = (int*)malloc(sizeof(int) * n);
kernel<<<grid, block>>>(data);

int data[1024];
kernel<<<grid, block>>>(data);

extern int *data;
kernel<<<grid, block>>>(data); 
```

~~注：NVLink上的ATS目前仅在IBM Power9系统上受支持。~~

### 硬件一致性

第二代NVLink允许CPU直接加载、存储、原子访问每个GPU的内存。结合新的CPU控制功能，NVLink支持一致性操作，允许从GPU内存读取的数据存储在CPU的缓存层次结构中。从CPU缓存访问的较低延迟是CPU性能的关键。计算能力6.x的设备只支持对等的GPU原子。具有计算能力7.x的设备可以通过NVLink发送GPU原子并在目标CPU上完成它们，因此第二代NVLink增加了对由GPU或CPU启动的原子的支持。

注意，cudaMalloc分配内存不能从CPU访问。因此，为了利用硬件一致性，用户必须使用统一的内存分配器，如cudaMallocManaged或具有ATS支持的系统分配器（请参阅系统分配器）。新的属性directmanagedmeaccessfromhost指示主机是否可以在不迁移的情况下直接访问设备上的托管内存。默认情况下，CPU访问的cudaMallocManaged分配的驻留在GPU的内存都将触发页面错误和数据迁移。应用程序可以使用cudamemadvisesetacaccessedby性能提示和cudapudeviceid，以便在支持的系统上直接访问GPU内存。

```c++
__global__ void write(int *ret, int a, int b) {
    ret[threadIdx.x] = a + b + threadIdx.x;
}
__global__ void append(int *ret, int a, int b) {
    ret[threadIdx.x] += a + b + threadIdx.x;
}
int main() {
    int *ret;
    cudaMallocManaged(&ret, 1000 * sizeof(int));
    cudaMemAdvise(ret, 1000 * sizeof(int), cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);       // set direct access hint

    write<<< 1, 1000 >>>(ret, 10, 100);            // pages populated in GPU memory
    cudaDeviceSynchronize();
    //如果directManagedMemAccessFromHost=1，不会发生数据迁移
    //如果directManagedMemAccessFromHost=0，发生错误并触发device-to-host数据迁移
    for(int i = 0; i < 1000; i++)
        printf("%d: A+B = %d\n", i, ret[i]);      
    
    //如果directManagedMemAccessFromHost=1，不会发生数据迁移
    //如果directManagedMemAccessFromHost=0，发生错误并触发host-to-device数据迁移        
    append<<< 1, 1000 >>>(ret, 10, 100);   
    cudaDeviceSynchronize();                     
    cudaFree(ret); 
    return 0;
}
```

### 访问计数器

具有计算能力7.0的设备引入了一种新的访问计数器功能，可以跟踪GPU对位于其他处理器上的内存的访问频率。访问计数器有助于确保将内存页移到访问页最频繁的处理器的物理内存中。访问计数器功能可以指导CPU和GPU之间以及对等GPU之间的迁移。 

对于cudaMallocManaged，可以通过使用cudamemadvisesetacessedby和相应的设备ID来选择使用访问计数器迁移。驱动程序还可以使用访问计数器来更有效地缓解震荡或内存超额订阅情况。 

注意：访问计数器当前仅在IBM POWER9系统上启用，并且仅对cudaMallocManaged分配器启用。 



## 编程模型

### Managed memory

大多数平台需要使用__ device __ 和 __ managed __关键字或者使用cudaMallocManaged()函数开辟统一内存来自动管理数据。计算能力低于6.x的设备必须始终使用分配器或通过声明全局存储在堆上分配托管内存。不能将以前分配的内存与统一内存相关联，也不能让统一内存系统管理CPU或GPU堆栈指针。从CUDA 8.0开始，在具有计算能力6.x设备的**支持系统**上，可以使用同一指针从GPU代码和CPU代码访问分配给默认OS分配器（例如malloc或new）的内存。在这些系统上，统一内存是默认的：不需要使用特殊的分配器或创建特殊管理的内存池。 

### 一致性与并发

1. 对于计算能力低于6.x的设备来说，不能同时访问managed memory，因为不能保证数据的一致性，可能GPU正在操作的时候刚好CPU访问，会造成错误数据。对于计算能力6.x并且支持的设备由于引入了页面错误机制所以可以同时访问统一内存，可以查询concurrentManagedAccess是否支持并发访问。

```c++
__device__ __managed__ int x, y=2;
__global__  void  kernel() {
    x = 10;
}
int main() {
    kernel<<< 1, 1 >>>();
    //如果同步函数放在这里就不会出错
    y = 20;            // Error on GPUs not supporting concurrent access
                       
    cudaDeviceSynchronize();
    return  0;
}
```

对于计算能力6.x之前的架构来说，当GPU正在执行的时候，使用CPU访问会发生段错误，如上代码所示。实际上，当任何内核操作正在执行时，GPU都可以**独占访问所有托管数据**，而不管特定的内核是否在积极地使用这些数据。从上面可以看到即使GPU使用的是x变量，CPU访问的是y变量，访问不同数据也会出错。 

注意，在上面的例子中，即使内核运行得很快并且在CPU接触Y之前完成，也需要显式同步。统一内存使用逻辑活动来确定GPU是否空闲。这与CUDA编程模型一致，CUDA编程模型指定内核可以在启动后的任何时间运行，并且在主机发出同步调用之前，不保证已经完成。

2.逻辑上保证GPU完成其工作的任何函数调用都是有效的。这包括cudaDeviceSynchronize()；cudaStreamSynchronize() and cudaStreamQuery()（前提是它返回cudaSuccess而不是cudaErrorNotReady），其中指定的流是在GPU上仍在执行的**唯一流**；cudaEventSynchronize()和cudaEventQuery()在指定事件后面没有任何设备工作的情况下；以及使用记录为与主机完全同步的cudaMemcpy() 和cudaMemset()。

CPU从流回调中访问托管数据是合法的，前提是GPU上没有其他可能正在访问托管数据的流处于活动状态。此外，没有任何设备工作的回调可用于同步：例如，通过从回调内部发出条件变量的信号；否则，CPU访问仅在回调期间有效。

注意以下几点： 

- 总是允许CPU在GPU处于活动状态时访问非托管零拷贝数据。 
- GPU在运行任何内核时都被认为是活动的，即使该内核不使用托管数据。如果内核可能使用数据，则禁止访问，除非设备属性ConcurrentManagedAccess为1。 
- 除了应用于非托管内存的多GPU访问之外，对托管内存的并发GPU访问没有限制。 
- 对访问托管数据的并发GPU内核没有约束。 

具体如下代码所示

```c++
int main() {
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    int *non_managed, *managed, *also_managed;
    cudaMallocHost(&non_managed, 4);    // Non-managed, CPU-accessible memory
    cudaMallocManaged(&managed, 4);
    cudaMallocManaged(&also_managed, 4);
    // Point 1: CPU can access non-managed data.
    kernel<<< 1, 1, 0, stream1 >>>(managed);
    *non_managed = 1;
    // Point 2: CPU cannot access any managed data while GPU is busy,
    //          unless concurrentManagedAccess = 1
    // Note we have not yet synchronized, so "kernel" is still active.
    *also_managed = 2;      // Will issue segmentation fault
    // Point 3: Concurrent GPU kernels can access the same data.
    kernel<<< 1, 1, 0, stream2 >>>(managed);
    // Point 4: Multi-GPU concurrent access is also permitted.
    cudaSetDevice(1);
    kernel<<< 1, 1 >>>(managed);
    return  0;
}
```



3.之前介绍的都是GPU会占用整个托管内存，为了更细粒度的访问托管内存，CUDA提供函数可以将托管内存和特定的流绑定在一起，这样，只要这个流执行完，CPU就可以访问，而不用管其他流是否以及完成。如果没绑定的话，那么整个托管内存对GPU都是可见的。

```c++
__device__ __managed__ int x, y=2;
__global__  void  kernel() {
    x = 10;
    //在内核访问y会产生未定义
}
int main() {
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    //将y和主机可访问关联在一起，　这样做有什么用？？如果内核不能访问了还不如开辟主机内存？？
    cudaStreamAttachMemAsync(stream1, &y, 0, cudaMemAttachHost);
    cudaDeviceSynchronize();          // Wait for Host attachment to occur.
    kernel<<< 1, 1, 0, stream1 >>>(); // Note: Launches into stream1.
    y = 20;                           // Success – a kernel is running but “y” 
                                      // has been associated with no stream.
    return  0;
}

//=============================分割线==================================

__device__ __managed__ int x, y=2;
__global__  void  kernel() {
    x = 10;
}
int main() {
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudaStreamAttachMemAsync(stream1, &x);// Associate “x” with stream1.
    cudaDeviceSynchronize();              // Wait for “x” attachment to occur.
    kernel<<< 1, 1, 0, stream1 >>>();     // Note: Launches into stream1.
    y = 20;                               // ERROR: “y” is still associated globally 
                                          // with all streams by default
    return  0;
}
```

**使用cudaStreamAttachMemAsync()的主要用途是可以让CPU线程并行执行独立的任务。每个CPU线程创建自己的流，这样不会造成使用默认流带来的依赖性问题，比如如果托管内存没有绑定特定的流，托管数据的默认全局可见性都会使多线程程序中的CPU线程之间的交互难以避免，那么每个CPU线程就会产生依赖，这会使程序的性能下降。**

```c++
// This function performs some task, in its own private stream.
void run_task(int *in, int *out, int length) {
    // Create a stream for us to use.
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // Allocate some managed data and associate with our stream.
    // Note the use of the host-attach flag to cudaMallocManaged();
    // we then associate the allocation with our stream so that
    // our GPU kernel launches can access it.
    int *data;
    //开辟的统一内存和特定流关联在一起，不会产生依赖
    cudaMallocManaged((void **)&data, length, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream, data);
    cudaStreamSynchronize(stream);
    // Iterate on the data in some way, using both Host & Device.
    for(int i=0; i<N; i++) {
        transform<<< 100, 256, 0, stream >>>(in, data, length);
        cudaStreamSynchronize(stream);
        host_process(data, length);    // CPU uses managed data.
        convert<<< 100, 256, 0, stream >>>(out, data, length);
    }
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(data);
}
```

在上面代码中，cudaMallocManaged()函数指定了cudaMemAttachHost标志，该标志创建了一个最初对设备端执行不可见的分配（默认分配对所有流上的所有GPU内核都可见）。这确保在数据分配和绑定特定流获取数据之间的时间间隔内不会与另一个线程的执行发生意外交互。 

如果没有这个标志，如果由另一个线程启动的内核恰好正在运行，则会考虑在GPU上使用新的分配。这可能会影响线程在能够显式地将其附加到私有流之前从CPU（例如，在基类构造函数内）访问新分配的数据的能力。因此，为了在线程之间实现安全的独立性，应该进行分配来指定这个标志。 

注意：另一种方法是在分配附加到流之后，在所有线程上设置一个进程范围的屏障。这将确保所有线程在启动任何内核之前完成其数据/流关联，从而避免危险。在销毁流之前需要第二个屏障，因为流销毁会导致分配恢复到其默认可见性。cudaMemAttachHost标志的存在不仅是为了简化这个过程，而且因为在需要的地方不可能总是插入全局屏障。 

4.由于托管内存可以从主机或设备访问，因此cudaMemcpy*()依赖于使用cudaMemcpyKind指定的传输类型来确定将数据作为主机指针或设备指针访问。


如果使用cudaMemcpyHostTo函数，并且源数据是managed memory(源数据可以一致访问)，那么它将从主机访问；否则，它将从设备访问。这同样适用于cudaMemcpyToHost函数并且目标内存是managed memory。同理如果指定了cudaMemcpyDeviceTo并且源数据是managed memory(目标数据可以一致访问)，则将从设备访问它。这同样适用于cudaMemcpyToDevice()函数并且目标内存是managed memory。


如果指定了cudaMemcpyDefault，则如果无法从设备一致访问托管数据，或者如果数据的首选位置是cudapudeviceid，并且可以从主机一致访问托管数据，则将从主机访问托管数据；否则，将从设备访问它。


当对托管内存使用cudaMemset时，始终可以从设备访问数据。数据必须可以从设备一致的访问；否则，将返回错误。


当通过cudamemcpy*或cudamemset*从设备访问数据时，操作流在GPU上被认为是活动的。在此期间，如果GPU的设备属性ConcurrentManagedAccess的值为零，则与该流或具有全局可见性的数据关联的任何CPU访问都将导致段错误。程序必须进行适当的同步，以确保在从CPU访问任何相关数据之前操作已经完成。

（1）为了在给定流中从主机一致地访问托管内存，必须至少满足以下条件之一：

- 与给定流关联的设备的设备属性ConcurrentManagedAccess具有非零值。
- 内存既没有全局可见性，也没有与给定流关联。(开辟的时候使用cudaMemAttachHost标志)

（2）对于在给定流中从设备一致地访问的托管内存，必须至少满足以下条件之一：

- 设备的设备属性ConcurrentManagedAccess具有非零值。
- 内存要么具有全局可见性，要么与给定的流相关联。



# 参考链接

[https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd)