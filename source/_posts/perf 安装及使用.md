---
title: perf 安装及使用
date: 2019-05-23 14:27:19
tags: 
- perf
categories: perf
top: 25
---

# perf 安装及使用



## perf 安装

```bash
$ sudo apt install linux-tools-common
## 注意版本号
$ sudo apt-get install linux-tools-4.4.0-24-generic linux-cloud-tools-4.4.0-24-generic linux-tools-generic linux-cloud-tools-generic
```



## perf 命令

```bash
ubuntu@ubuntu-B250-HD3:~$ perf

 usage: perf [--version] [--help] [OPTIONS] COMMAND [ARGS]

 The most commonly used perf commands are:
   annotate        Read perf.data (created by perf record) and display annotated code
   archive         Create archive with object files with build-ids found in perf.data file
   bench           General framework for benchmark suites
   buildid-cache   Manage build-id cache.
   buildid-list    List the buildids in a perf.data file
   c2c             Shared Data C2C/HITM Analyzer.
   config          Get and set variables in a configuration file.
   data            Data file related processing
   diff            Read perf.data files and display the differential profile
   evlist          List the event names in a perf.data file
   ftrace          simple wrapper for kernel's ftrace functionality
   inject          Filter to augment the events stream with additional information
   kallsyms        Searches running kernel for symbols
   kmem            Tool to trace/measure kernel memory properties
   kvm             Tool to trace/measure kvm guest os
   list            List all symbolic event types
   lock            Analyze lock events
   mem             Profile memory accesses
   record          Run a command and record its profile into perf.data
   report          Read perf.data (created by perf record) and display the profile
   sched           Tool to trace/measure scheduler properties (latencies)
   script          Read perf.data (created by perf record) and display trace output
   stat            Run a command and gather performance counter statistics
   test            Runs sanity tests.
   timechart       Tool to visualize total system behavior during a workload
   top             System profiling tool.
   probe           Define new dynamic tracepoints
   trace           strace inspired tool

 See 'perf help COMMAND' for more information on a specific command.
```



| 序号 | 命令          | 作用                                                         |
| ---- | ------------- | ------------------------------------------------------------ |
| 1    | annotate      | 解析perf record生成的perf.data文件，显示被注释的代码。       |
| 2    | archive       | 根据数据文件记录的build-id，将所有被采样到的elf文件打包。利用此压缩包，可以再任何机器上分析数据文件中记录的采样数据。 |
| 3    | bench         | perf中内置的benchmark，目前包括两套针对调度器和内存管理子系统的benchmark。 |
| 4    | buildid-cache | 管理perf的buildid缓存，每个elf文件都有一个独一无二的buildid。buildid被perf用来关联性能数据与elf文件。 |
| 5    | buildid-list  | 列出数据文件中记录的所有buildid。                            |
| 6    | diff          | 对比两个数据文件的差异。能够给出每个符号（函数）在热点分析上的具体差异。 |
| 7    | evlist        | 列出数据文件perf.data中所有性能事件。                        |
| 8    | inject        | 该工具读取perf record工具记录的事件流，并将其定向到标准输出。在被分析代码中的任何一点，都可以向事件流中注入其它事件。 |
| 9    | kmem          | 针对内核内存（slab）子系统进行追踪测量的工具                 |
| 10   | kvm           | 用来追踪测试运行在KVM虚拟机上的Guest OS。                    |
| 11   | list          | 列出当前系统支持的所有性能事件。包括硬件性能事件、软件性能事件以及检查点。 |
| 12   | lock          | 分析内核中的锁信息，包括锁的争用情况，等待延迟等。           |
| 13   | mem           | 内存存取情况                                                 |
| 14   | record        | 收集采样信息，并将其记录在数据文件中。随后可通过其它工具对数据文件进行分析。 |
| 15   | report        | 读取perf record创建的数据文件，并给出热点分析结果。          |
| 16   | sched         | 针对调度器子系统的分析工具。                                 |
| 17   | script        | 执行perl或python写的功能扩展脚本、生成脚本框架、读取数据文件中的数据信息等。 |
| 18   | stat          | 执行某个命令，收集特定进程的性能概况，包括CPI、Cache丢失率等。 |
| 19   | test          | perf对当前软硬件平台进行健全性测试，可用此工具测试当前的软硬件平台是否能支持perf的所有功能。 |
| 20   | timechart     | 针对测试期间系统行为进行可视化的工具                         |
| 21   | top           | 类似于linux的top命令，对系统性能进行实时分析。               |
| 22   | trace         | 关于syscall的工具。                                          |
| 23   | probe         | 用于定义动态检查点。                                         |



## perf 使用

系统级性能优化通常包括两个阶段：性能剖析（performance profiling）和代码优化。

性能剖析的目标是寻找性能瓶颈，查找引发性能问题的原因及热点代码。

代码优化的目标是针对具体性能问题而优化代码或编译选项，以改善软件性能。

### perf list

perf list 可以显示所有支持的事件类型，可以显示特定模块支持的perf事件：hw/cache/pmu都是硬件相关的；tracepoint基于内核的ftrace；sw实际上是内核计数器

```bash
ubuntu@ubuntu-B250-HD3:~$ sudo perf list

List of pre-defined events (to be used in -e):

  branch-instructions OR branches                    [Hardware event]
  branch-misses                                      [Hardware event]
  bus-cycles                                         [Hardware event]
  cache-misses                                       [Hardware event]
  cache-references                                   [Hardware event]
  cpu-cycles OR cycles                               [Hardware event]
  instructions                                       [Hardware event]
  ref-cycles                                         [Hardware event]

  alignment-faults                                   [Software event]
  bpf-output                                         [Software event]
  context-switches OR cs                             [Software event]
  ...
  minor-faults                                       [Software event]
  page-faults OR faults                              [Software event]
  task-clock                                         [Software event]

  L1-dcache-load-misses                              [Hardware cache event]
  L1-dcache-loads                                    [Hardware cache event]
  L1-dcache-stores                                   [Hardware cache event]
　...
  iTLB-loads                                         [Hardware cache event]
  node-load-misses                                   [Hardware cache event]
  node-loads                                         [Hardware cache event]
  node-store-misses                                  [Hardware cache event]
  node-stores                                        [Hardware cache event]

  branch-instructions OR cpu/branch-instructions/    [Kernel PMU event]
  branch-misses OR cpu/branch-misses/                [Kernel PMU event]
  bus-cycles OR cpu/bus-cycles/                      [Kernel PMU event]
  ...
  cycles-ct OR cpu/cycles-ct/                        [Kernel PMU event]
  cycles-t OR cpu/cycles-t/                          [Kernel PMU event]
  el-abort OR cpu/el-abort/                          [Kernel PMU event]
```



### perf stat

```
ubuntu@ubuntu-B250-HD3:~$ sudo perf stat

 Performance counter stats for 'system wide':

      78762.210390      cpu-clock (msec)          #    8.000 CPUs utilized          
           102,321      context-switches          #    0.001 M/sec                  
               888      cpu-migrations            #    0.011 K/sec                  
           102,842      page-faults               #    0.001 M/sec                  
    34,026,798,403      cycles                    #    0.432 GHz                    
    34,823,881,180      instructions              #    1.02  insn per cycle         
     7,258,813,402      branches                  #   92.161 M/sec                  
        59,271,180      branch-misses             #    0.82% of all branches        

       9.845475012 seconds time elapsed
```

- cpu-clock：任务真正占用的处理器时间，单位为ms。CPUs utilized = task-clock / time elapsed，CPU的占用率。
- context-switches：程序在运行过程中上下文的切换次数。
- CPU-migrations：程序在运行过程中发生的处理器迁移次数。Linux为了维持多个处理器的负载均衡，在特定条件下会将某个任务从一个CPU迁移到另一个CPU。
- CPU迁移和上下文切换：发生上下文切换不一定会发生CPU迁移，而发生CPU迁移时肯定会发生上下文切换。发生上下文切换有可能只是把上下文从当前CPU中换出，下一次调度器还是将进程安排在这个CPU上执行。
- page-faults：缺页异常的次数。当应用程序请求的页面尚未建立、请求的页面不在内存中，或者请求的页面虽然在内存中，但物理地址和虚拟地址的映射关系尚未建立时，都会触发一次缺页异常。另外TLB不命中，页面访问权限不匹配等情况也会触发缺页异常。
- cycles：消耗的处理器周期数。如果把被ls使用的cpu cycles看成是一个处理器的，那么它的主频为2.486GHz。可以用cycles / task-clock算出。
- stalled-cycles-frontend：指令读取或解码的质量步骤，未能按理想状态发挥并行左右，发生停滞的时钟周期。
- stalled-cycles-backend：指令执行步骤，发生停滞的时钟周期。
- instructions：执行了多少条指令。IPC为平均每个cpu cycle执行了多少条指令。
- branches：遇到的分支指令数。
- branch-misses：是预测错误的分支指令数。

perf stat 常用参数：

    -a, --all-cpus        显示所有CPU上的统计信息
    -C, --cpu <cpu>       显示指定CPU的统计信息
    -c, --scale           scale/normalize counters
    -D, --delay <n>       ms to wait before starting measurement after program start
    -d, --detailed        detailed run - start a lot of events
    -e, --event <event>   event selector. use 'perf list' to list available events
    -G, --cgroup <name>   monitor event in cgroup name only
    -g, --group           put the counters into a counter group
    -I, --interval-print <n>
                          print counts at regular interval in ms (>= 10)
    -i, --no-inherit      child tasks do not inherit counters
    -n, --null            null run - dont start any counters
    -o, --output <file>   输出统计信息到文件
    -p, --pid <pid>       stat events on existing process id
    -r, --repeat <n>      repeat command and print average + stddev (max: 100, forever: 0)
    -S, --sync            call sync() before starting a run
    -t, --tid <tid>       stat events on existing thread id
### perf top

```bash
$ sudo perf top

Samples: 147K of event 'cycles:ppp', Event count (approx.): 59239863864
Overhead  Shared Object                                 Symbol
   9.86%  [kernel]                                      [k] do_syscall_64
   5.12%  [kernel]                                      [k] syscall_return_via_sysret
   3.12%  libcuda.so.418.56                             [.] 0x00000000002fb584
   2.99%  libgomp.so.1.0.0                              [.] 0x0000000000011b27
   1.82%  [kernel]                                      [k] __schedule
   1.21%  [kernel]                                      [k] pick_next_task_fair
   1.07%  [kernel]                                      [k] _raw_spin_lock
   1.00%  [unknown]                                     [k] 0xfffffe000013a01b
   0.98%  libpthread-2.23.so                            [.] pthread_mutex_lock
   0.81%  [kernel]                                      [k] prepare_exit_to_usermode
   0.81%  libc-2.23.so                                  [.] __sched_yield
   0.80%  [kernel]                                      [k] cpuacct_charge
   0.79%  libpthread-2.23.so                            [.] pthread_mutex_unlock
   0.76%  [kernel]                                      [k] clear_page_erms
   0.69%  [kernel]                                      [k] native_sched_clock
   0.69%  [kernel]                                      [k] update_curr
   0.63%  [kernel]                                      [k] yield_task_fair
```

- 第一列：符号引发的性能事件的比例，指占用的cpu周期比例。
- 第二列：符号所在的DSO(Dynamic Shared Object)，可以是应用程序、内核、动态链接库、模块。
- 第三列：DSO的类型。[.]表示此符号属于用户态的ELF文件，包括可执行文件与动态链接库；[k]表述此符号属于内核或模块。
- 第四列：符号名。有些符号不能解析为函数名，只能用地址表示。

perf top 常用的选项：

- -e <event>：指明要分析的性能事件。
- -p <pid>：Profile events on existing Process ID (comma sperated list). 仅分析目标进程及其创建的线程。
- -k <path>：Path to vmlinux. Required for annotation functionality. 带符号表的内核映像所在的路径。
- -K：不显示属于内核或模块的符号。
- -U：不显示属于用户态程序的符号。
- -d <n>：界面的刷新周期，默认为2s，因为perf top默认每2s从mmap的内存区域读取一次性能数据。
- -g：得到函数的调用关系图。

### perf reocrd

使用 top 和 stat之后，已经大致有数了。要进一步分析，便需要一些粒度更细的信息。比如说已经断定目标程序计算量较大，也许是因为有些代码写的不够精简。那么面对长长的代码文件，究竟哪几行代码需要进一步修改呢？这便需要使用perf record 记录单个函数级别的统计信息，并使用 perf report 来显示统计结果。

```bash
$ sudo perf record -g -e cpu-clock ./test
$ sudo perf report
## 查看百分比比较高的为耗时比较严重的函数
```

perf record 常用参数：

- -e record指定PMU事件
- --filter  event事件过滤器
- -a  录取所有CPU的事件
- -p  录取指定pid进程的事件
- -o  指定录取保存数据的文件名
- -g  使能函数调用图功能
- -C 录取指定CPU的事件

perf report 常用参数：

- -i  导入的数据文件名称，如果没有则默认为perf.data
- -g  生成函数调用关系图，**此时内核要打开CONFIG_KALLSYMS；用户空间库或者执行文件需要带符号信息(not stripped)，编译选项需要加上-g。**
- --sort  从更高层面显示分类统计信息，比如： pid, comm, dso, symbol, parent, cpu,socket, srcline, weight, local_weight.

## TODO

还有很多功能，带后续慢慢挖掘。

## 参考链接

[https://www.cnblogs.com/arnoldlu/p/6241297.html](https://www.cnblogs.com/arnoldlu/p/6241297.html)

[https://www.ibm.com/developerworks/cn/linux/l-cn-perf1/index.html](https://www.ibm.com/developerworks/cn/linux/l-cn-perf1/index.html)