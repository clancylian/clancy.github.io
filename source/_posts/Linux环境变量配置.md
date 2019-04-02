---
title: Linux环境变量配置
date: 2019-04-02 11:15:24
tags:
 - Linux
categories: Linux
top: 1
---

# 设置文件为:
```
//全局设置文件
/etc/profile

//针对某个用户设置
~/.bashrc
````

# 设置PATH:
```
export PATH=/your/bin/path:$PATH
```
**注意：** PATH=中间不能有空格。

# 设置LD_LIBRARY_PATH:
```
export LD_LIBRARY_PATH=/your/lib/path:$LD_LIBRARY_PATH
```
另：也可以在/etc/ld.so.conf.d/ 下设置目录，然后调用ldconfig生效。

# 生效：
```
source ~/.bashrc
or
source /etc/profile
```