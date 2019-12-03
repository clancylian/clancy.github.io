---
title: Ubuntu18.04本地源制作方法--转
date: 2019-12-03 12:19:11
tags:
- ubuntu
categories: 环境搭建
top: 50
---

# Ubuntu18.04本地源制作方法--转

## 环境

一台能上网的Ubuntu电脑，一台不能上网的Ubuntu电脑。Ubuntu版本都是18.04LTS。

## 目标

将能上网的Ubuntu电脑安装的软件制作成源，通过U盘拷贝给内网电脑，内网电脑根据此离线源通过apt安装软件。

为什么不直接拷贝deb安装呢？因为有些软件安装依赖的包比较多。

备注：从Ubuntu 16.04 (xenial)起， 在将本地deb软件包创建repo时候，跟14.04以前的版本相比，强制要求gpg对Release文件签名，否则无法使用。

## 步骤

#### 1.安装gpg软件和相关软件：

```bash
apt-get install gnupg
apt-get install rng-tools
```

密钥创建过程中，需要使用到足够的随机数(random)，可先行安装rng-tools, 该工具可以常驻后台的方式, 生成随机数，避免gpg密钥创建过程中的长时间等待问题 

```
rngd -r /dev/urandom
```

生成公钥和私钥：

```
gpg --gen-key
```

执行gpg会进入一些对话，其中要新建一个用户名username和相应的密码。

结束之后，输入命令，可以查看key：

```
gpg --list-key
```

公钥，需在不能上网的Ubuntu电脑导入，供apt-get使用

```
gpg -a --export username> username.pub
```



#### 2.在外网电脑上准备安装包源

以下是安装包目录

```
sudo rm -rf /var/cache/apt/archives/*  # 清空缓存目录，这一步也可以不做
```

-d只是下载安装包，并不安装。

```
sudo apt-get -d install <包名>
```

在本地建一个目录，将下载下来的安装包拷贝到此目录：

```
$ mkdir /var/debs
$ cp -r /var/cache/apt/archives/*.deb /var/debs/
```

在debs这个目录创建Packages.gz，注意生成的路径带debs，否则内网安装时会说找不到文件

```
$ cd /var
$ apt-ftparchive packages debs > debs/Packages
$ cd debs
$ gzip -c Packages > Packages.gz
```

在debs这个目录下创建release file

```
apt-ftparchive release ./ > Release
```

ubuntu apt-get 对软件包索引，首先要求InRelease文件，其次才去找Release、Release.gpg文件； 这情况下， 其实只需要创建InRelease文件(包含Release文件和明文签名)即可:

```
gpg --clearsign -o InRelease Release 
gpg -abs -o Release.gpg Release 
```

#### 3.将生成的debs目录和公钥文件username.pub拷贝到U盘

#### 4.在内网的电脑上将debs目录拷贝到/vars/下面，注意和外网的目录一样。

如下并导入公钥。

```
apt-key add username.pub
```

#### 5.在内网电脑上备份apt源文件/etc/apt/source.list,并修改源。

```
sudo gedit /etc/apt/sources.list
```

将sources.list 原来的内容都注释掉。在最后添加

```
deb file:/var debs/
```

注意上面的 /var 和 debs/ 之间的空格，以及 “/”。不要写错/var/debs/路径了。

#### 6.更新索引

```
sudo apt-get update
```



## 附

依赖包查找

```shell
#!/bin/bash

logfile=/home/yzy/Desktop/log
ret=""
function getDepends
{
   echo "fileName is" $1>>$logfile
   # use tr to del < >
   ret=`apt-cache depends $1|grep Depends |cut -d: -f2 |tr -d "<>"`
   echo $ret|tee  -a $logfile
}
# 需要获取其所依赖包的包
libs="build-essential"                  # 或者用$1，从命令行输入库名字

# download libs dependen. deep in 3
i=0
while [ $i -lt 5 ] ;
do
    let i++
    echo $i
    # download libs
    newlist=" "
    for j in $libs
    do
        added="$(getDepends $j)"
        newlist="$newlist $added"
        apt install $added --reinstall -d -y
    done

    libs=$newlist
done
```

注意：用bash执行不是sh，因为sh指向不是bash。



## 参考链接

[https://blog.csdn.net/yruilin/article/details/85124870](https://blog.csdn.net/yruilin/article/details/85124870)

[https://blog.csdn.net/junbujianwpl/article/details/52811153](https://blog.csdn.net/junbujianwpl/article/details/52811153)