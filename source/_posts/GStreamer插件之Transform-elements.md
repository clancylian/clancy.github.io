---
title: GStreamer插件之Transform elements
date: 2019-03-29 11:41:44
tags:
 - GStreamer
 - 插件
 - 视频流
categories: GStreamer
top: 10
---

# 前言
写了个gstreamer插件继承于基类Transform element. 其中有些概念需要理解一下，特此做下笔记。参考[官网链接][1]

Transform elements基于sink和src pad的caps将输入的buffer转换为输出buffer。而输出的caps完全由输入的caps所定义，这表明像解码器这种组件不能够由Transform elements来实现，因为其输出的视频帧的宽高在输入时是被压缩在流当中的，所以输入是没有宽高这种属性的。如下所示的avdec_h264解码组件所示：
```c++
Pad Templates:
  SRC template: 'src'
    Availability: Always
    Capabilities:
      video/x-raw
                 format: { (string)I420, (string)YUY2, (string)RGB, (string)BGR,
				 (string)Y42B, (string)Y444, (string)YUV9, (string)Y41B, (string)GRAY8,
				 (string)RGB8P, (string)I420, (string)Y42B, (string)Y444, (string)UYVY,
				 (string)NV12, (string)NV21, (string)ARGB, (string)RGBA, (string)ABGR, 
				 (string)BGRA, (string)GRAY16_BE, (string)GRAY16_LE, (string)A420, 
				 (string)RGB16, (string)RGB15, (string)I420_10BE, (string)I420_10LE, 
				 (string)I422_10BE, (string)I422_10LE, (string)Y444_10BE, (string)Y444_10LE, 
				 (string)GBR, (string)GBR_10BE, (string)GBR_10LE, (string)A420_10BE,
				 (string)A420_10LE, (string)A422_10BE, (string)A422_10LE, (string)A444_10BE,
				 (string)A444_10LE, (string)GBRA, (string)xRGB, (string)RGBx, (string)xBGR, 
				 (string)BGRx, (string)I420_12BE, (string)I420_12LE, (string)I422_12BE, 
				 (string)I422_12LE, (string)Y444_12BE, (string)Y444_12LE, (string)GBR_12BE,
				 (string)GBR_12LE, (string)GBRA_12BE, (string)GBRA_12LE }

  SINK template: 'sink'
    Availability: Always
    Capabilities:
      video/x-h264
              alignment: au
          stream-format: { (string)avc, (string)byte-stream }
```

典型transform elements包含：

 - audio convertors (audioconvert, audioresample,…)
 - video convertors (colorspace, videoscale, …)
 - filters (capsfilter, volume, colorbalance, …)

要实现transform elements必须关心的是：

 - efficient negotiation both up and downstream
 - efficient buffer alloc and other buffer management

transform elements可以使用不同模式：

 - passthrough (no changes are done on the input buffers)
 - in-place (changes made directly to the incoming buffers without requiring a copy or new buffer allocation)
 - metadata changes only

transform元素通常也会处理以下事项：

 - flushing, seeking
 - state changes
 - timestamping, this is typically done by copying the input timestamps to the output buffers but subclasses should be able to override this.
 - QoS, avoiding calls to the subclass transform function
 - handle scheduling issues such as push and pull based operation.

transform element应在任何时候可以重新协商sink和src caps，并改变操作模式。根据不同的模式，buffer的分配可能采用不同策略。

# transform element behaviour
## Processing
transform主要由两种处理函数：

 - transform(): Transform the input buffer to the output buffer. The output buffer is guaranteed to be writable and different from the input buffer.
 - transform_ip(): Transform the input buffer in-place. The input buffer is writable and of bigger or equal size than the output buffer.

转换操作有以下模式：

 - passthrough: The element will not make changes to the buffers, buffers are pushed straight through, caps on both sides need to be the same. The element can optionally implement a transform_ip() function to take a look at the data, the buffer does not have to be writable.
 - in-place: Changes can be made to the input buffer directly to obtain the output buffer. The transform must implement a transform_ip() function.
 - copy-transform: The transform is performed by copying and transforming the input buffer to a new output buffer. The transform must implement a transform() function.

当没有使用*transform()*函数的时候，只有in-place 和 passthrough模式可以使用，这意味着sinkpad和srcpad要一样或者src buffer大于等于sink buffer。

当没有使用*transform_ip()*函数的时候，只允许passthrough和copy-transforms两种模式，提供这个函数可以避免内存的拷贝。

当没有使用以上两种函数时，只使用passthrough模式。

## Negotiation
在push mode下transform element的协商总是从sink到src：

 - sinkpad接收到新的caps事件
 - transform函数算出它可以将此caps转化成什么
 - 尝试不做任何修改，因为我们倾向于不做任何事情
 - transform配置自身使得可以将sink caps转换到模板src caps
 - transform在srcpad上设置处理输出caps
```c++
          sinkpad              transform               srcpad
CAPS event   |                    |                      |
------------>|  find_transform()  |                      |
             |------------------->|                      |
             |                    |       CAPS event     |
             |                    |--------------------->|
             | <configure caps> <-|                      |
```
transform 有三个函数执行协商：

 - transform_caps(): Transform the caps on a certain pad to all the possible supported caps on the other pad. The input caps are guaranteed to be a simple caps with just one structure. The caps do not have to be fixed.
 - fixate_caps(): Given a caps on one pad, fixate the caps on the other pad. The target caps are writable.
 - set_caps(): Configure the transform for a transformation between src caps and dest caps. Both caps are guaranteed to be fixed caps.

如果*transform_caps()*未定义，默认只执行同样的转换。
如果*set_caps()*未定义，我们不关心caps，在这种情况下我们假设没有任何内容写到缓冲区，我们不会为该*transform_ip()*函数强制执行可写缓冲区（如果存在）。
 
我们对transform元素需要的一个常见函数是找到从一种格式（src）到另一种格式（dest）的最佳转换。该函数的一些要求是：
 
 - 有一个固定的src caps
 - 找到一个固定的transform element可以转换成的dest caps 
 - dest caps是兼容的并且可被peer elements接受
 - transform函数倾向于使src caps == dest caps
 - transform函数可以选择性固定dest caps

*find_transform()*函数执行如下:

 - 从一个固定的src caps开始；
 - 检测这些caps是否可以被用作src caps，这通常由元素的padtemplate强制执行；
 - 使用*transform_caps()*计算所有的可以转换生成的caps
 - 如果原始的caps是transforms的一个子集，尝试caps是否能被peer接受。如果可行，我们可以执行passthrough然后设置src == dest。这只要简单调用*gst_pad_peer_query_accept_caps()*即可。
 - 如果caps不是固定的，我们需要固定它们。
 - *transform_caps()*检索每个转换的caps
 - 使用*fixate_caps()*固定caps
 - 如果caps是固定的，使用*_peer_query_accept_caps()*检测peer是否接受他们，如果接受，我们就找到了dest caps。
 - 如果找遍caps还没发现可转换的caps就表明失败了。
 - 如果找到dest caps，使用*set_caps()*进行配置。

在协商过程之后，transform元素通常是一个稳定的状态。我们可以确定这个状态：

### src和sink pads有同样的caps

 - passthrough: buffers are inspected but no metadata or buffer data is changed. The input buffers don’t need to be writable. The input buffer is simply pushed out again without modifications. (SCP)
```c++
          sinkpad              transform               srcpad
  chain()    |                    |                      |
------------>|   handle_buffer()  |                      |
             |------------------->|      pad_push()      |
             |                    |--------------------->|
             |                    |                      |
```
 - in-place: buffers are modified in-place, this means that the input buffer is modified to produce a new output buffer. This requires the input buffer to be writable. If the input buffer is not writable, a new buffer has to be allocated from the bufferpool. (SCI)
```c++
          sinkpad              transform               srcpad
  chain()    |                    |                      |
------------>|   handle_buffer()  |                      |
             |------------------->|                      |
             |                    |   [!writable]        |
             |                    |   alloc buffer       |
             |                  .-|                      |
             |  <transform_ip>  | |                      |
             |                  '>|                      |
             |                    |      pad_push()      |
             |                    |--------------------->|
             |                    |                      |
```
 - copy transform: a new output buffer is allocate from the bufferpool and data from the input buffer is transformed into the output buffer. (SCC)
```c++
         sinkpad              transform               srcpad
  chain()    |                    |                      |
------------>|   handle_buffer()  |                      |
             |------------------->|                      |
             |                    |     alloc buffer     |
             |                  .-|                      |
             |     <transform>  | |                      |
             |                  '>|                      |
             |                    |      pad_push()      |
             |                    |--------------------->|
             |                    |                      |
```

    
### src和sink pads有不一样的样的caps

 - in-place: input buffers are modified in-place. This means that the input buffer has a size that is larger or equal to the output size. The input buffer will be resized to the size of the output buffer. If the input buffer is not writable or the output size is bigger than the input size, we need to pad-alloc a new buffer. (DCI)
```c++
          sinkpad              transform               srcpad
  chain()    |                    |                      |
------------>|   handle_buffer()  |                      |
             |------------------->|                      |
             |                    | [!writable || !size] |
             |                    |     alloc buffer     |
             |                  .-|                      |
             |  <transform_ip>  | |                      |
             |                  '>|                      |
             |                    |      pad_push()      |
             |                    |--------------------->|
             |                    |                      |
```
 - copy transform: a new output buffer is allocated and the data from the input buffer is transformed into the output buffer. The flow is exactly the same as the case with the same-caps negotiation. (DCC)

## Allocation
当transform element配置完成之后，缓冲池需要根据caps开辟内存，主要有两种情况：

 - 当使用passthrough模式的时候不需要在transform element中开辟内存。
 - 当不使用passthrough，并且需要开辟输出buffer。

对于第一种情况，我们不需要查询和配置pool。我们让upstream自动决定是否需要bufferpool，然后我们将从下游到上游进行代理。

对于第二种情况，我们在srcpad设置分配内存池。

为了分配内存，我们还需要知道输出空间的大小，这里有两个函数获取大小：

 - transform_size(): Given a caps and a size on one pad, and a caps on the other pad, calculate the size of the other buffer. This function is able to perform all size transforms and is the preferred method of transforming a size.

 - get_unit_size(): When the input size and output size are always a multiple of each other (audio conversion, ..) we can define a more simple get_unit_size() function. The transform will use this function to get the same amount of units in the source and destination buffers. For performance reasons, the mapping between caps and size is kept in a cache.


  [1]: https://gstreamer.freedesktop.org/documentation/design/element-transform.html