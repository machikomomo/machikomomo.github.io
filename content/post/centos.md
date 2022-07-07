---
author: "momo"
date: 2022-07-04
title: "数据开发"
categories: [
    "数据开发",
]
---

## CentOS

安装：https://zhuanlan.zhihu.com/p/136547600

1.下载CentOS 7.7阿里云镜像，得到BT文件，迅雷下载镜像

2.VMware官网下载VMware Fusion并激活

3.用下载好的.ios镜像文件安装虚拟机

4.安装完成。用户momo酱（momochan），密码6位数。原文章后半段的问题没遇到。

## linux命令（内存）

`free` 查询可用内存

`free -h` 以MB、GB的形式展示可用内存

mem和sawp的区别：

原文链接：https://blog.csdn.net/JineD/article/details/107675345

memory就是机器的物理内存，读写速度低于cpu一个量级，但是高于磁盘不止一个量级。所以，程序和数据如果在内存的话，会有非常快的读写速度。但是，内存的造价是要高于磁盘的，虽然相对来说价格一直在降低。除此之外，内存的断电丢失数据也是一个原因。所以不能把所有数据和程序都保存在内存中。既然不能全部使用内存，那数据还有程序肯定不可能一直霸占在内存中。

当内存没有可用的，就必须要把内存中不经常运行的程序给踢出去。但是：踢到哪里去？这时候swap就出现了。swap全称为swap place，即交换区。当内存不够的时候，被踢出的进程被暂时存储到交换区（swap位于硬盘）。当需要这条被踢出的进程的时候，就从交换区重新加载到内存，否则它不会主动交换到真实内存中。内存与swap之间是按照内存页为单位来交换数据的，一般Linux中页的大小设置为4kb。而内存与磁盘则是按照块来交换数据的。

当物理内存使用完或者达到一定比例之后，可以使用swap做临时的内存使用。当物理内存和swap都被使用完那么就会出错：“out of memory（内存不足）”。

![](https://halfbit.oss-cn-hangzhou.aliyuncs.com/13141656992660_.pic.jpg)

## linux命令（磁盘）

`df -h `df（带-h 以MB、GB的形式展示）命令可以获取硬盘被占用了多少空间，目前还剩下多少空间等信息。第一列是硬盘分区。Mounted on列表示文件系统的安装点。注意看根目录。	

![](https://halfbit.oss-cn-hangzhou.aliyuncs.com/13151656992910_.pic.jpg)

https://www.cnblogs.com/zhoug2020/p/5609220.html

磁盘是Linux系统中一项非常重要的资源，如何对其进行有效的管理直接关系到整个系统的性能问题。对Linux磁盘管理稍微有一些学习和经验的朋 友们应该都知道df、du和fdisk这三个常用命令：df用于检查文件系统磁盘占用情况，du检查磁盘空间占用情况，而fdisk用于磁盘分区。

## 网络配置

https://www.csdn.net/tags/MtTaEg5sMzE1ODE0LWJsb2cO0O0O.html

照着做就可以联网。

无权限vim报错：https://blog.csdn.net/weixin_40853073/article/details/81707177

## 修改主机名称

修改为hadoop1：https://cloud.tencent.com/developer/article/1721856

## 修改虚拟机配置

在vm里面调整关机，关机不等于挂起。修改内核数量2个、内存大小2G、磁盘20G。

系统操作尽量用root：su root

把进程kill掉：kill -9 xxxx（进程号）

克隆虚拟机，完整克隆。

## 资源共享

简单手册：https://linuxtools-rst.readthedocs.io/zh_CN/latest/index.html

书籍：《鸟哥的私房菜》

## 安装jdk

jar包下载，oracle官网（注册了，gmail是账号，密码是带大写和特殊字符）：https://blog.csdn.net/lyhkmm/article/details/79524712

`cp /home/momochan/tmp/jdk-8u333-linux-x64.tar.gz /opt/software/`

解压缩到：

`tar -zxvf jdk-8u333-linux-x64.tar.gz -C /opt/module`

jdk要用需要配置环境变量

`cd /etc/profile.d/`

`sudo vim my_env.sh` 写my_env.sh文件

![](https://halfbit.oss-cn-hangzhou.aliyuncs.com/13171657003956_.pic.jpg)

写完保存 再 `source /etc/profile` 即可

输入java检测环境是否配置完成

JDK：Java Development Kit 是程序员使用java语言编写java程序所需的开发工具包，是提供给程序员使用的。JDK包含了JRE，同时还包含了编译java源码的编译器javac，还包含了很多java程序调试和分析的工具：jconsole，jvisualvm等工具软件，还包含了java程序编写所需的文档和demo例子程序。

## 安装hadoop

hadoop 3.3.1 ： https://mirrors.cnnic.cn/apache/hadoop/common/hadoop-3.3.1/

`cp /home/momochan/tmp/hadoop-3.3.1.tar.gz /opt/software/`

`tar -zxvf hadoop-3.3.1.tar.gz -C /opt/module`

`sudo vim /etc/profile.d/my_env.sh`

![](https://halfbit.oss-cn-hangzhou.aliyuncs.com/13181657004878_.pic.jpg)

`source /etc/profile`

输入hadoop检查是否配置完成

## 修改系统文件的两种方法

1.用root用户

2.命令前加sudo

## 将102的jdk和hadoop拷贝

服务器之间的拷贝：scp

`scp -r jdk1.8.0_333/ momochan@hadoop103:/opt/module/`

-r 表示递归 对方表示：用户名@主机名称:路径

报错，ssh: Could not resolve hostname master: Name or service not known 

需要修改hosts文件，建立ip和主机名联系：https://blog.csdn.net/weixin_45648723/article/details/101013877

或者也可以：

`scp -r momochan@hadoop102:/opt/module/hadoop-3.3.1 ./`

在103上把102的module下的jdk和hadoop同时拷贝到104的module下：

`scp -r momochan@hadoop102:/opt/module/* momochan@hadoop104:/opt/module/`

## 同步比拷贝更高效

`[momochan@hadoop103 hadoop-3.3.1]$ rm -rf wcinput/ wcoutput/`

`[momochan@hadoop102 module]$ rsync -av hadoop-3.3.1/ momochan@hadoop103:/opt/module/hadoop-3.3.1/`

第一次用scp，后续更改可以用rsync

## 写一个集群分发脚本

`[momochan@hadoop102 ~]$ echo $PATH
/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/home/momochan/.local/bin:/home/momochan/bin:/opt/module/jdk1.8.0_333/bin:/opt/module/jdk1.8.0_333/bin:/opt/module/hadoop-3.3.1/bin:/opt/module/hadoop-3.3.1/sbin`

在/home/momochan/bin写xsync，就可以在全局使用xsync

在home下mkdir bin

然后在bin下创建脚本即可。

我决定在本地写然后放上去。这样写起来简单。

rm xsync 删除刚创建的

## 创建txt文件

touch 1.txt

获取某个文件所在目录

## 软连接

`ln -s aaa bbb`

`[momochan@hadoop102 ~]$ cd -P bbb`

加上-P可以进到根目录即aaa目录

## -p

`mkdir -p $pdir`

不管有没有这个路径文件夹，都创建一个

## 访问和退出

ssh hadoop103

exit

## 修改脚本执行权限

`[momochan@hadoop102 bin]$ chmod 777 xsync`

## 测试脚本

`[momochan@hadoop102 ~]$ xsync bin`

报错：line 19: [: missing `]'

原因，shell脚本里面if语句要有空格：https://blog.csdn.net/u012478275/article/details/121403329

## 用脚本分发环境变量

`sudo ./bin/xsync /etc/profile.d/my_env.sh`

分发完毕以后分别在103和104上：

`source /etc/profile`

## ssh免密登陆配置

目标：在102ssh103，不需要密码

`ls -al` 可以看到所有的隐藏文件

`[momochan@hadoop102 .ssh]$ cd .ssh`

`[momochan@hadoop102 .ssh]$ ll`

`[momochan@hadoop102 .ssh]$ ssh-keygen -t rsa` 生成密钥对

`[momochan@hadoop102 .ssh]$ cat id_rsa` 查看私钥

`[momochan@hadoop102 .ssh]$ cat id_rsa.pub` 查看公钥

`[momochan@hadoop102 .ssh]$ ssh-copy-id hadoop103` 即可

以上。记得对自己102本身也要进行一次授权。

`[momochan@hadoop102 .ssh]$ cat authorized_keys ` 看对哪些用户授权了

以上是在momochan用户下，配置。如果切换用户eg：root的话需要再配置。并且步骤要从cd开始。回到家目录。

在root下exit即可退回到momochan用户。

## hadoop配置

在sublime text里面使用command+f可以搜索查询

把笔记word里面，四个xml文件的configuration中间的部分修改到对应文件

`cd /opt/module/hadoop-3.3.1/`

`cd etc/hadoop/`

`ll`

`vim core-site.xml` 其他三个同理

3.3.1有个小bug，在之后的版本在配置yarn-site.xml的时候不需要配置“环境变量的继承“这一部分。话虽然这么说，我还是原样粘贴吧。

以上。在102上把hdfs和yarn配置完毕。把hadoop分发到103、104即可。

`[momochan@hadoop102 etc]$ xsync hadoop/`

## workers配置

略

## 启动集群

第一次需要初始化

启动

## 网页端

报错：本机上不去网页hadoop102:9870

解决，在本地hosts里面添加映射：https://blog.csdn.net/ChangXinZaiCi/article/details/108012168

## HDFS数据的位置

`/opt/module/hadoop-3.1.3/data/dfs/data/current/BP-1436128598-192.168.10.102-1610603650062/current/finalized/subdir0/subdir0`

hadoop的数据会在三台（恰好为三台）服务器上备份。即使一个服务器宕机，另外两台也还是保留数据。

Replication:表示备份数量（一共）

![](https://halfbit.oss-cn-hangzhou.aliyuncs.com/13201657031782_.pic.jpg)

## yarn运行情况查看

略

## 配置历史服务器

目的：查看程序的历史运行情况

`[momochan@hadoop102 hadoop-3.3.1]$ hadoop fs -mkdir /input`

`hadoop fs -put wcinput/word.txt /input`

`hadoop jar share/hadoop/mapreduce/hadoop-mapreduce-examples-3.3.1.jar wordcount /input /output`

## 内部接口 外部接口（端口）

略

## 配置日志的聚集功能

略

## API-客户端hadoop

1.mac上进行hadoop安装环境配置：97

https://blog.csdn.net/pmdream/article/details/113180085（到hadoop version完成即可。3.3.1。）

2.org.apache.hadoop.conf.configuration 不存在 说明maven依赖没有下载好，解决：

https://blog.csdn.net/qq_46092061/article/details/120127385 （貌似不是这个原因）

https://blog.csdn.net/qq_38509173/article/details/110946260 （pom.xml更新以后没有自动下载）

=========其实到这里就可以完成功能了（可以在hdfs上创建文件夹），但是日志无法显示，修改bug3

3.SLF4J: Failed to load class “org.slf4j.impl.StaticLoggerBinder” 解决方法

修改pom.xml里，增加一段：https://blog.51cto.com/YangRoc/5045868 版本对应即可

4.可以成功运行，但是有warning，NativeCodeLoader: Unable to load native-hadoop library for your platform

https://blog.csdn.net/sun7_9/article/details/121381611 （不介意，留着了，warning）

## 大数据

1.开源系统的系统架构和实现

2.系统实现的要求和关键点，从系统的需求出发

3.流式计算，分布式计算

4.开源大数据系统：分布式计算 分布式存储等

三大马车：

计算需求：计算量 TB PB 数据实时性 毫秒级

主流系统面向不同场景的实现

存储需求：数据量越来越大 数据库是结构化的表设计 现在的数据（视频 日志 行为等非结构化）写入的性能 吞吐 要求高

存储系统的可扩展性和可挪用性

存储场景的细分 数据流 

数据的一致性要求

目前主流的存储系统 数据湖 数据流

数据库 行存 大数据 列存

资源调度：1.资源隔离 2.资源利用率 3.调度性能

数据应用：怎么在业务上用起来，结合业务场景，选择大数据系统



1.选方向，但底层技术互通

2.大数据系统都是开源系统，各个系统apache都有搭建的模型，从零开始，启动，提交作业

3.主动参与社区（开源？）

4.数据密集型应用系统设计 大数据系统背后的设计思想和能力

5.计算系统以java为主 存储系统以c和c++为主 python系统也是有的 总的来讲对

6.进阶4-6人 基础4-8人





