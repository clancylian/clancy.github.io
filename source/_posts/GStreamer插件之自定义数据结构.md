---
title: GStreamer插件之自定义数据结构
date: 2019-04-01 18:18:58
tags:
 - GStreamer
 - 插件
 - 视频流
categories: GStreamer
top: 10
---

# GstMeta

### 当我们需要添加自定义的数据结构到GstBuffer中时，需要使用GstMeta自定义数据结构。

### GstMeta数据结构为：
```c++
struct _GstMeta {
  GstMetaFlags       flags;
  const GstMetaInfo *info;    /* tag and info for the meta item */
};
```
### 其中 info 结构体为：
```c++
struct _GstMetaInfo {
  GType                      api;       //api 类型
  GType                      type;      //具体实现类型
  gsize                      size;      //自定义数据结构体大小

  GstMetaInitFunction        init_func; //初始化函数
  GstMetaFreeFunction        free_func; //释放函数
  GstMetaTransformFunction   transform_func; //转换函数
};
```
其中api成员需要由*gst_meta_api_type_register()*函数注册生成，具体看下文实施例。

### GstMeta是我们自定义数据结构体的公共头，比如gstreamer自带时间信息结构体：
```c++
struct _GstMetaTiming {
  GstMeta        meta;        /* common meta header */

  GstClockTime   dts;         /* decoding timestamp */
  GstClockTime   pts;         /* presentation timestamp */
  GstClockTime   duration;    /* duration of the data */
  GstClockTime   clock_rate;  /* clock rate for the above values */
};
```
### 自定义的数据结构体由字段或者方法组成。一个典型的buffer可能是如以下结构：
```c++
                         +----------------------------------+
GstMiniObject            |  GType (GstBuffer)               |
                         |  refcount, flags, copy/disp/free |
                         +----------------------------------+
GstBuffer                |  pool,pts,dts,duration,offsets   |
                         |  <private data>                  |
                         +..................................+
                         |  next                           ---+
                      +- |  info                           ------> GstMetaInfo
GstMetaTiming         |  |                                  | |
                      |  |  dts                             | |
                      |  |  pts                             | |
                      |  |  duration                        | |
                      +- |  clock_rate                      | |
                         +  . . . . . . . . . . . . . . . . + |
                         |  next                           <--+
GstVideoMeta       +- +- |  info                           ------> GstMetaInfo
                   |  |  |                                  | |
                   |  |  |  flags                           | |
                   |  |  |  n_planes                        | |
                   |  |  |  planes[]                        | |
                   |  |  |  map                             | |
                   |  |  |  unmap                           | |
                   +- |  |                                  | |
                      |  |  private fields                  | |
GstVideoMetaImpl      |  |  ...                             | |
                      |  |  ...                             | |
                      +- |                                  | |
                         +  . . . . . . . . . . . . . . . . + .
```

# 自定义meta实现
## 头文件
```c++
#ifndef GSTFACEPARAMMETA_H
#define GSTFACEPARAMMETA_H

#include "gst/gst.h"

/** Defines GStreamer metadata types. */
//定义枚举类型，用来指定meta_data指针所指向的数据类型
typedef enum
{
    META_INIT = 0x0,
    FACE_PARAM,
} Meta_type;

typedef struct _FaceParamMeta FaceParamMeta;

//该结构体为要附加的元数据结构体
struct _FaceParamMeta {
    //公共头
    GstMeta meta;
    //存储各种需要的自定义结构体，方便扩展
    void *meta_data;
    //meta_data指针指向的结构体类型
    int meta_type;
    //类似回调函数指针，在buffer销毁的时候会调用
    GDestroyNotify destroy;
};

//注册api type的接口，返回api用于注册meta
GType face_param_meta_api_get_type(void);
#define FACE_PARAM_META_API_TYPE (face_param_meta_api_get_type())

//注册meta的时候返回GstMetaInfo
const GstMetaInfo *face_param_meta_get_info(void);
#define FACE_PARAM_META_INFO (face_param_meta_get_info())

//往buffer中添加数据，返回FaceParamMeta*指针指向添加位置
FaceParamMeta *gst_buffer_add_face_param_meta(GstBuffer *buffer, void *metadata, int metatype, GDestroyNotify destroy);

//从buffer中获取自定义meta数据
FaceParamMeta *gst_buffer_get_face_param_meta(GstBuffer *buffer);

#endif // GSTFACEPARAMMETA_H
```
## 实现文件
```c++
#include "gstfaceparammeta.h"

GType face_param_meta_api_get_type(void)
{
    static volatile GType type;
    //可以往api type加入标签，使用gst_meta_api_type_has_tag查找是否由我们自定义结构体
    static const gchar *tags[] = {"face", "detect", NULL};

    if(g_once_init_enter(&type)) {
        //注册api type
        GType _type = gst_meta_api_type_register("FaceParamMetaAPI", tags);
        g_once_init_leave(&type, _type);
    }

    return type;
}

//初始化函数实现
static gboolean face_param_meta_init(GstMeta *meta, gpointer params, GstBuffer *buffer)
{
    FaceParamMeta *emeta = (FaceParamMeta*) meta;

    emeta->meta_type = META_INIT;
    emeta->meta_data = NULL;
    emeta->destroy = NULL;
    return TRUE;
}

//转换函数实现，具体复杂情况下怎么用还需琢磨
static gboolean face_param_meta_transform(GstBuffer *transbuf, GstMeta *meta,
                            GstBuffer *buffer, GQuark type, gpointer data)
{
    FaceParamMeta *emeta = (FaceParamMeta*) meta;
    gst_buffer_add_face_param_meta(transbuf, emeta->meta_data, emeta->meta_type, emeta->destroy);

    return TRUE;
}

//释放函数
static void face_param_meta_free(GstMeta *meta, GstBuffer *buffer)
{
    FaceParamMeta *emeta = (FaceParamMeta*) meta;

    emeta->meta_type = META_INIT;
    //回调函数
    emeta->destroy(emeta->meta_data);
}

//注册我们的meta函数
const GstMetaInfo *face_param_meta_get_info(void)
{
    static const GstMetaInfo *meta_info = NULL;

    if(g_once_init_enter(&meta_info)) {
        //注册返回GstMetaInfo
        //第二个参数可以在gst_meta_get_info（）函数使用
        const GstMetaInfo *mi = gst_meta_register(FACE_PARAM_META_API_TYPE,
                                                  "FaceParamMeta",
                                                  sizeof(FaceParamMeta),
                                                  face_param_meta_init,
                                                  face_param_meta_free,
                                                  face_param_meta_transform);
        g_once_init_leave(&meta_info, mi);
    }

    return meta_info;
}

//增加数据
FaceParamMeta *gst_buffer_add_face_param_meta(GstBuffer *buffer, void *metadata, int metatype, GDestroyNotify destroy)
{
    FaceParamMeta *meta;
    g_return_val_if_fail(GST_IS_BUFFER(buffer), NULL);

    meta = (FaceParamMeta *)gst_buffer_add_meta(buffer, FACE_PARAM_META_INFO, NULL);
    //指针简单赋值，需要注意外面传进来的指针必须不是局部变量，否则会自动释放
    meta->meta_data = metadata;
    //meta_data指向类型
    meta->meta_type = metatype;
    //回调函数，在free的时候使用
    meta->destroy = destroy;

    return meta;
}

//获取数据
FaceParamMeta *gst_buffer_get_face_param_meta(GstBuffer *buffer)
{
    FaceParamMeta *meta;

    meta = (FaceParamMeta *)gst_buffer_get_meta(buffer, FACE_PARAM_META_API_TYPE);

    return meta;
}
```
# 具体使用

## 调用添加数据函数
```c++
faceParamMeta = gst_buffer_add_face_param_meta(outbuf, metadata, FACE_PARAM, free_face_param_meta);
```
## 释放函数为：
```c++
/**
 * Free the metadata allocated in attach_metadata_full_frame
 */
static void
free_face_param_meta (gpointer meta_data)
{
     g_print ("free output buffer, free output buffer.\n");
  MtcnnPluginOutput *output = (MtcnnPluginOutput *)meta_data;
  g_free(output);
}
```
## 获取数据可以使用如下方法：
### 方法1：
直接调用我们自己定义的函数
```c++
metadata = gst_buffer_get_face_param_meta();
```
### 方法2：
因为我们可能在不同的element中添加很多个自定义数据结构，使用方法1只能取到最后一个添加的，需要使用以下方法进行遍历：
```c++
static GQuark _ivameta_quark = 0;
if (!_ivameta_quark) {
    //注意参数时我们注册api type时添加的tag
    //类似键值对
    _ivameta_quark = g_quark_from_static_string ("face");
}
```
```c++
GstMeta *gst_meta;

// Standard way of iterating through buffer metadata
while ((gst_meta = gst_buffer_iterate_meta (outbuf, &state)) != NULL) {
    
    //可以获取GstMetaInfo，注意参数需要和注册时一致
    const GstMetaInfo *info = gst_meta_get_info("FaceParamMeta");

    //查询是否有我们想要的api type
    if (!gst_meta_api_type_has_tag (gst_meta->info->api, _ivameta_quark)) {
          continue;
    }

    //如果是，转成我们的结构体
    ivameta = (FaceParamMeta *) gst_meta;

    // Check if the metadata of IvaMeta contains object bounding boxes
    if (ivameta->meta_type != FACE_PARAM)
        continue;

    //根据meta_type获取具体数据
    meta_data = (MtcnnPluginOutput *) ivameta->meta_data;
}
```

# 参考官网链接:
[GstMeta][1]
[Memory allocation][2]

[1]: https://gstreamer.freedesktop.org/documentation/design/meta.html
[2]: https://gstreamer.freedesktop.org/documentation/plugin-development/advanced/allocation.html