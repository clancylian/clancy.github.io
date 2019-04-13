---
title: GStreamer插件编写学习
date: 2019-03-29 12:31:35
tags:
 - GStreamer
 - 插件
 - 视频流
categories: GStreamer
top: 10
---

# 下载插件模板
```
shell $ git clone https://gitlab.freedesktop.org/gstreamer/gst-template.git
Initialized empty Git repository in /some/path/gst-template/.git/
remote: Counting objects: 373, done.
remote: Compressing objects: 100% (114/114), done.
remote: Total 373 (delta 240), reused 373 (delta 240)
Receiving objects: 100% (373/373), 75.16 KiB | 78 KiB/s, done.
Resolving deltas: 100% (240/240), done.
```
**以上方案有点过时，可到github下载gst-plugins-bad模块，使用里面的工具生成模板。**

# 使用make_element生成模板
```
cd gst-template/gst-plugin/src
../tools/make_element MyFilter <基类文件如:gsttransform>
```

# 修改makefile.am文件
```make
    # Note: plugindir is set in configure
    ##############################################################################
    # TODO: change libgstplugin.la to something else, e.g. libmysomething.la     #
    ##############################################################################
    plugin_LTLIBRARIES = libgstplugin.la libgstaudiofilterexample.la
    ##############################################################################
    # TODO: for the next set of variables, name the prefix if you named the .la, #
    #  e.g. libmysomething.la => libmysomething_la_SOURCES                       #
    #                            libmysomething_la_CFLAGS                        #
    #                            libmysomething_la_LIBADD                        #
    #                            libmysomething_la_LDFLAGS                       #
    ##############################################################################
    ## Plugin 1
    # sources used to compile this plug-in
    libgstplugin_la_SOURCES = gstplugin.c gstplugin.h
    # compiler and linker flags used to compile this plugin, set in configure.ac
    libgstplugin_la_CFLAGS = $(GST_CFLAGS)
    libgstplugin_la_LIBADD = $(GST_LIBS)
    libgstplugin_la_LDFLAGS = $(GST_PLUGIN_LDFLAGS)
    libgstplugin_la_LIBTOOLFLAGS = --tag=disable-static
    # headers we need but don't want installed
    noinst_HEADERS = gstplugin.h
```

# 生成makefile文件
```make
./autogen.sh
make
sudo make install
```
**注意：需要修改configure.ac文件里面安装的路径，不然插件会被安装到/usr/local/lib/gstreamer-1.0中** 

# Example Demo
```c++
#include <gst/gst.h>
/* Definition of structure storing data for this element. */
typedef struct _GstMyFilter {
  GstElement element;
  GstPad *sinkpad, *srcpad;
  gboolean silent;
} GstMyFilter;
    
/* Standard definition defining a class for this element. */
typedef struct _GstMyFilterClass {
  GstElementClass parent_class;
} GstMyFilterClass;

/* Standard macros for defining types for this element.  */
#define GST_TYPE_MY_FILTER (gst_my_filter_get_type())
#define GST_MY_FILTER(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_MY_FILTER,GstMyFilter))
#define GST_MY_FILTER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_MY_FILTER,GstMyFilterClass))
#define GST_IS_MY_FILTER(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_MY_FILTER))
#define GST_IS_MY_FILTER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_MY_FILTER))
  
/* Standard function returning type information. */
GType gst_my_filter_get_type (void);
```

# Element metadata
元素元数据提供额外的元素信息，使用*gst_element_class_set_metadata*或者*gst_element_class_set_static_metadata*函数来设置，其参数包含：

 - A long, English, name for the element.
 - The type of the element, see the docs/design/draft-klass.txt document
   in the GStreamer core source tree for details and examples.
 - A brief description of the purpose of the element.
 - The name of the author of the element, optionally followed by a
   contact email address in angle brackets.
#### 例如：
```c++
gst_element_class_set_static_metadata (klass,
"An example plugin",
"Example/FirstExample",
"Shows the basic structure of a plugin",
"your name <your.name@your.isp>");
```
#### 以上函数在初始化*_class_init ()*插件的时候调用
```c++
static void
gst_my_filter_class_init (GstMyFilterClass * klass)
{
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);

[..]
  gst_element_class_set_static_metadata (element_klass,
    "An example plugin",
    "Example/FirstExample",
    "Shows the basic structure of a plugin",
    "your name <your.name@your.isp>");
}
```

# GstStaticPadTemplate
GstStaticPadTemplate是用来描述将要创建的pad的信息，包含:

 - A short name for the pad.
 - Pad direction.
 - Existence property. This indicates whether the pad exists always (an “always” pad), only in some cases (a “sometimes” pad) or only if the application requested such a pad (a “request” pad).
 - Supported types by this element (capabilities).
#### 例如：
```c++
static GstStaticPadTemplate sink_factory =
GST_STATIC_PAD_TEMPLATE (
  "sink",         //名称
  GST_PAD_SINK,   //方向sink or src
  GST_PAD_ALWAYS,   //availability
  GST_STATIC_CAPS ("ANY")  //capability
);
```
同样该*sink_factory*也是在初始化时候使用，通过*gst_element_class_add_pad_template ()*和*gst_static_pad_template_get ()*来调用
```c++
static GstStaticPadTemplate sink_factory = [..],
    src_factory = [..];

static void
gst_my_filter_class_init (GstMyFilterClass * klass)
{
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);
[..]

  gst_element_class_add_pad_template (element_class,
    gst_static_pad_template_get (&src_factory));
  gst_element_class_add_pad_template (element_class,
    gst_static_pad_template_get (&sink_factory));
}
```
该pad将在构造函数*_init ()*函数中使用*gst_pad_new_from_static_template ()*来创建。

**注：对于每个element来说，有两个构造函数，其中*_class_init()*函数只调用一次，用来说明类所拥有的信号、参数、虚函数以及设置全局状态；*_init()*函数则用在初始化实例特定的实例。**

# 插件初始化函数
当我们写完插件的所有部件之后，需要编写插件的初始化函数，这个函数在插件加载的时候调用，需要返回TRUE or FALSE来觉得是否正确加载。在这个函数中，任何支持的element插件应该被注册。
```c++
static gboolean
plugin_init (GstPlugin *plugin)
{
  return gst_element_register (plugin, "my_filter",
                   GST_RANK_NONE,
                   GST_TYPE_MY_FILTER);
}

GST_PLUGIN_DEFINE (
  GST_VERSION_MAJOR,
  GST_VERSION_MINOR,
  my_filter,
  "My filter plugin",
  plugin_init,
  VERSION,
  "LGPL",
  "GStreamer",
  "http://gstreamer.net/"
)
```
    
# pads具体说明
PADS是数据流进出每个element的端口，在初始化*_init ()*函数中，你从pad template中创建了一个pad，这个pad template是在*_class_init ()*函数中注册的，创建了pad之后，你必须在sinkpad设置*_chain ()*函数指针用来接收和处理数据。可选的，你也可以设置*_event ()*和*_query ()*指针；pad亦可以使用循环模式，这意味着它们可以自己拉取数据。
```c++
static void
gst_my_filter_init (GstMyFilter *filter)
{
  /* pad through which data comes in to the element */
  filter->sinkpad = gst_pad_new_from_static_template (
    &sink_template, "sink");
  /* pads are configured here with gst_pad_set_*_function () */

  gst_element_add_pad (GST_ELEMENT (filter), filter->sinkpad);

  /* pad through which data goes out of the element */
  filter->srcpad = gst_pad_new_from_static_template (
    &src_template, "src");
  /* pads are configured here with gst_pad_set_*_function () */
    
  gst_element_add_pad (GST_ELEMENT (filter), filter->srcpad);
    
  /* properties initial value */
      filter->silent = FALSE;
    }
```
## Sometimes Pad
Sometimes pad 只有在一些特定情况下才创建，这取决于流内容。比如demuxers解析流头。每个element可以创建多个sometimes pad，唯一限制就是都要有唯一的名字。当流数据被销毁时（比如从PAUSED到READY状态），pad也应该被销毁，但是在EOS状态不应该被销毁。
```c++
typedef struct _GstMyFilter {
[..]
  gboolean firstrun;
  GList *srcpadlist;
} GstMyFilter;

//静态目标类型sometimes
static GstStaticPadTemplate src_factory =
GST_STATIC_PAD_TEMPLATE (
  "src_%u",
  GST_PAD_SRC,
  GST_PAD_SOMETIMES,
  GST_STATIC_CAPS ("ANY")
);

static void
gst_my_filter_class_init (GstMyFilterClass *klass)
{
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);
[..]
//注册pad
  gst_element_class_add_pad_template (element_class,
    gst_static_pad_template_get (&src_factory));
[..]
}

static void
gst_my_filter_init (GstMyFilter *filter)
{
[..]
  filter->firstrun = TRUE;
  filter->srcpadlist = NULL;
}

/*
 * Get one line of data - without newline.
 */

static GstBuffer *
gst_my_filter_getline (GstMyFilter *filter)
{
  guint8 *data;
  gint n, num;

  /* max. line length is 512 characters - for safety */
  for (n = 0; n < 512; n++) {
    num = gst_bytestream_peek_bytes (filter->bs, &data, n + 1);
    if (num != n + 1)
      return NULL;

    /* newline? */
    if (data[n] == '\n') {
      GstBuffer *buf = gst_buffer_new_allocate (NULL, n + 1, NULL);

      gst_bytestream_peek_bytes (filter->bs, &data, n);
      gst_buffer_fill (buf, 0, data, n);
      gst_buffer_memset (buf, n, '\0', 1);
      gst_bytestream_flush_fast (filter->bs, n + 1);

      return buf;
    }
  }
}

static void
gst_my_filter_loopfunc (GstElement *element)
{
  GstMyFilter *filter = GST_MY_FILTER (element);
  GstBuffer *buf;
  GstPad *pad;
  GstMapInfo map;
  gint num, n;

  /* parse header */
  if (filter->firstrun) {
    gchar *padname;
    guint8 id;

    if (!(buf = gst_my_filter_getline (filter))) {
      gst_element_error (element, STREAM, READ, (NULL),
             ("Stream contains no header"));
      return;
    }
    gst_buffer_extract (buf, 0, &id, 1);
    num = atoi (id);
    gst_buffer_unref (buf);

    /* for each of the streams, create a pad */
    //根据头的流个数，创建num个pad
    for (n = 0; n < num; n++) {
      padname = g_strdup_printf ("src_%u", n);
      pad = gst_pad_new_from_static_template (src_factory, padname);
      g_free (padname);

      /* here, you would set _event () and _query () functions */

      /* need to activate the pad before adding */
      gst_pad_set_active (pad, TRUE);

      gst_element_add_pad (element, pad);
      filter->srcpadlist = g_list_append (filter->srcpadlist, pad);
    }
  }

  /* and now, simply parse each line and push over */
  if (!(buf = gst_my_filter_getline (filter))) {
    GstEvent *event = gst_event_new (GST_EVENT_EOS);
    GList *padlist;

    for (padlist = srcpadlist;
         padlist != NULL; padlist = g_list_next (padlist)) {
      pad = GST_PAD (padlist->data);
      gst_pad_push_event (pad, gst_event_ref (event));
    }
    gst_event_unref (event);
    /* pause the task here */
    return;
  }

  /* parse stream number and go beyond the ':' in the data */
  gst_buffer_map (buf, &map, GST_MAP_READ);
  num = atoi (map.data[0]);
  if (num >= 0 && num < g_list_length (filter->srcpadlist)) {
  //取第N个pad塞数据
    pad = GST_PAD (g_list_nth_data (filter->srcpadlist, num);

    /* magic buffer parsing foo */
    for (n = 0; map.data[n] != ':' &&
                map.data[n] != '\0'; n++) ;
    if (map.data[n] != '\0') {
      GstBuffer *sub;

      /* create region copy that starts right past the space. The reason
       * that we don't just forward the data pointer is because the
       * pointer is no longer the start of an allocated block of memory,
       * but just a pointer to a position somewhere in the middle of it.
       * That cannot be freed upon disposal, so we'd either crash or have
       * a memleak. Creating a region copy is a simple way to solve that. */
      sub = gst_buffer_copy_region (buf, GST_BUFFER_COPY_ALL,
          n + 1, map.size - n - 1);
      gst_pad_push (pad, sub);
    }
  }
  gst_buffer_unmap (buf, &map);
  gst_buffer_unref (buf);
}
```
## Request pad
Request pad 只有在外部需要的时候才创建而不是element内部。比如tee element可以根据需要拷贝多份数据到不同的分支。需要实现*request_new_pad*和*release_pad*两个虚函数
```c++
static GstPad * gst_my_filter_request_new_pad   (GstElement     *element,
                         GstPadTemplate *templ,
                                                 const gchar    *name,
                                                 const GstCaps  *caps);

static void gst_my_filter_release_pad (GstElement *element,
                                       GstPad *pad);

static GstStaticPadTemplate sink_factory =
GST_STATIC_PAD_TEMPLATE (
  "sink_%u",
  GST_PAD_SINK,
  GST_PAD_REQUEST,
  GST_STATIC_CAPS ("ANY")
);

static void
gst_my_filter_class_init (GstMyFilterClass *klass)
{
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);
[..]
  gst_element_class_add_pad_template (klass,
    gst_static_pad_template_get (&sink_factory));
[..]
  element_class->request_new_pad = gst_my_filter_request_new_pad;
  element_class->release_pad = gst_my_filter_release_pad;
}

static GstPad *
gst_my_filter_request_new_pad (GstElement     *element,
                   GstPadTemplate *templ,
                   const gchar    *name,
                               const GstCaps  *caps)
{
  GstPad *pad;
  GstMyFilterInputContext *context;

  context = g_new0 (GstMyFilterInputContext, 1);
  pad = gst_pad_new_from_template (templ, name);
  gst_pad_set_element_private (pad, context);

  /* normally, you would set _chain () and _event () functions here */

  gst_element_add_pad (element, pad);

  return pad;
}

static void
gst_my_filter_release_pad (GstElement *element,
                           GstPad *pad)
{
  GstMyFilterInputContext *context;

  context = gst_pad_get_element_private (pad);
  g_free (context);

  gst_element_remove_pad (element, pad);
}

```


# The chain function
chain函数是处理数据的地方，记住buffers并不总是可写的。
```c++
static GstFlowReturn gst_my_filter_chain (GstPad    *pad,
                                          GstObject *parent,
                                          GstBuffer *buf);

[..]

static void
gst_my_filter_init (GstMyFilter * filter)
{
[..]
  /* configure chain function on the pad before adding
   * the pad to the element */
  gst_pad_set_chain_function (filter->sinkpad,
      gst_my_filter_chain);
[..]
}

static GstFlowReturn
gst_my_filter_chain (GstPad    *pad,
                     GstObject *parent,
             GstBuffer *buf)
{
  GstMyFilter *filter = GST_MY_FILTER (parent);

  if (!filter->silent)
    g_print ("Have data of size %" G_GSIZE_FORMAT" bytes!\n",
        gst_buffer_get_size (buf));

  return gst_pad_push (filter->srcpad, buf);
}
```
# The event function
在一些高级的element中，需要设置event处理函数。它通知一些发生在数据流中的特定事件，比如（caps, end-of-stream, newsegment, tags, etc.），事件可以传播到上游和下游，所以你可以在sink pads和source pads接收到。
```c++
static void
gst_my_filter_init (GstMyFilter * filter)
{
[..]
  gst_pad_set_event_function (filter->sinkpad,
      gst_my_filter_sink_event);
[..]
}



static gboolean
gst_my_filter_sink_event (GstPad    *pad,
                  GstObject *parent,
                  GstEvent  *event)
{
  GstMyFilter *filter = GST_MY_FILTER (parent);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_CAPS:
      /* we should handle the format here */
      break;
    case GST_EVENT_EOS:
      /* end-of-stream, we should close down all stream leftovers here */
      gst_my_filter_stop_processing (filter);
      break;
    default:
      break;
  }

  return gst_pad_event_default (pad, parent, event);
}

static GstFlowReturn
gst_my_filter_chain (GstPad    *pad,
             GstObject *parent,
             GstBuffer *buf)
{
  GstMyFilter *filter = GST_MY_FILTER (parent);
  GstBuffer *outbuf;

  outbuf = gst_my_filter_process_data (filter, buf);
  gst_buffer_unref (buf);
  if (!outbuf) {
    /* something went wrong - signal an error */
    GST_ELEMENT_ERROR (GST_ELEMENT (filter), STREAM, FAILED, (NULL), (NULL));
    return GST_FLOW_ERROR;
  }

  return gst_pad_push (filter->srcpad, outbuf);
}
```
# The query function
通过query函数，element可以接收查询事件并作出回复，比如（position, duration，supported formats，scheduling modes）。查询同样可以传播到上下游，你可以在sink pads和source pads接收到。
```c++
static gboolean gst_my_filter_src_query (GstPad    *pad,
                                         GstObject *parent,
                                         GstQuery  *query);

[..]

static void
gst_my_filter_init (GstMyFilter * filter)
{
[..]
  /* configure event function on the pad before adding
   * the pad to the element */
  gst_pad_set_query_function (filter->srcpad,
      gst_my_filter_src_query);
[..]
}

static gboolean
gst_my_filter_src_query (GstPad    *pad,
                 GstObject *parent,
                 GstQuery  *query)
{
  gboolean ret;
  GstMyFilter *filter = GST_MY_FILTER (parent);

  switch (GST_QUERY_TYPE (query)) {
    case GST_QUERY_POSITION:
      /* we should report the current position */
      [...]
      break;
    case GST_QUERY_DURATION:
      /* we should report the duration here */
      [...]
      break;
    case GST_QUERY_CAPS:
      /* we should report the supported caps here */
      [...]
      break;
    default:
      /* just call the default handler */
      ret = gst_pad_query_default (pad, parent, query);
      break;
  }
  return ret;
}
```
# element状态
状态可以用来描述element是否初始化，是否准备传送数据，是否正在处理数据。

 - GST_STATE_NULL element默认状态，不分配任何资源，不加载任何库。
 - GST_STATE_READY 分配默认资源(runtime-libraries, runtime-memory)，不分配或者定义stream-specific相关的资源，当状态从NULL到READY时，分配任何non-stream-specific资源，加载运行时库。当状态从READY转到NULL时，卸载相关资源，比如硬件设备。注意文件也是流，所以在READY状态下不会分配。
 - GST_STATE_PAUSED element准备接收和处理数据，对于大部分elements来说PAUSED和PLAYING是一样的，唯一的区别是sink elements，它只接收一次数据然后阻塞住，这时候pipeline处于'prerolled'状态，准备渲染数据。
 - GST_STATE_PLAYING 在播放状态下sink elements渲染接收的数据，其他elements和PAUSED状态一样。

### 管理filter状态
一般来说，elements继承一些基类，比如sources, sinks and filter/transformation elements。如果是继承这些基类，你就不需要亲自处理状态的改变。你只需继承基类的虚函数*start()*和*stop()*。如果不是继承基类，而是继承GstElement，你就必须自己处理状态的改变，比如像demuxer和muxer这种插件。通过虚函数，element可以被通知状态的改变然后初始化必要数据，也可以选择状态改变失败。

**注意，向上（NULL => READY，READY => PAUSED，PAUSED => PLAYING）和向下（PLAYING => PAUSED，PAUSED => READY，READY => NULL）状态更改在两个单独的块中处理，向下状态发生变化只有在我们链接到父类的状态更改函数之后才会处理。这是为了安全地处理多个线程的并发访问所必需的**。
```c++
static GstStateChangeReturn
gst_my_filter_change_state (GstElement *element, GstStateChange transition);

static void
gst_my_filter_class_init (GstMyFilterClass *klass)
{
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);

  element_class->change_state = gst_my_filter_change_state;
}



static GstStateChangeReturn
gst_my_filter_change_state (GstElement *element, GstStateChange transition)
{
  GstStateChangeReturn ret = GST_STATE_CHANGE_SUCCESS;
  GstMyFilter *filter = GST_MY_FILTER (element);

  switch (transition) {
	case GST_STATE_CHANGE_NULL_TO_READY:
	  if (!gst_my_filter_allocate_memory (filter))
		return GST_STATE_CHANGE_FAILURE;
	  break;
	default:
	  break;
  }

  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);
  if (ret == GST_STATE_CHANGE_FAILURE)
	return ret;

  switch (transition) {
	case GST_STATE_CHANGE_READY_TO_NULL:
	  gst_my_filter_free_memory (filter);
	  break;
	default:
	  break;
  }

  return ret;
}
```
# 增加属性

通过属性可以控制element的行为。属性在*_class_init ()*函数中定义，在*_get_property ()*和*a _set_property ()*函数中设置或者获取。可以在*_init ()*构造函数中初始化属性值。
```c++
/* properties */
enum {
  PROP_0,
  PROP_SILENT
  /* FILL ME */
};

static void gst_my_filter_set_property  (GObject      *object,
                         guint         prop_id,
                         const GValue *value,
                         GParamSpec   *pspec);
static void gst_my_filter_get_property  (GObject      *object,
                         guint         prop_id,
                         GValue       *value,
                         GParamSpec   *pspec);

static void
gst_my_filter_class_init (GstMyFilterClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  /* define virtual function pointers */
  object_class->set_property = gst_my_filter_set_property;
  object_class->get_property = gst_my_filter_get_property;

  /* define properties */
  g_object_class_install_property (object_class, PROP_SILENT,
    g_param_spec_boolean ("silent", "Silent",
              "Whether to be very verbose or not",
              FALSE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
}

static void
gst_my_filter_set_property (GObject      *object,
                guint         prop_id,
                const GValue *value,
                GParamSpec   *pspec)
{
  GstMyFilter *filter = GST_MY_FILTER (object);

  switch (prop_id) {
    case PROP_SILENT:
      filter->silent = g_value_get_boolean (value);
      g_print ("Silent argument was changed to %s\n",
           filter->silent ? "true" : "false");
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_my_filter_get_property (GObject    *object,
                guint       prop_id,
                GValue     *value,
                GParamSpec *pspec)
{
  GstMyFilter *filter = GST_MY_FILTER (object);

  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, filter->silent);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}
```
# 两种调度模式
Gstreamer 有两种调度模式

 - push mode
 - pull mode

我们之前讨论的*_chain ()*函数属于push mode，通过调用*gst_pad_push ()*，使得下游的element的*_chain ()*被调用。



后续补充