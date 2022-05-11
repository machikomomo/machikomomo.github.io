---
author: "momo"
date: 2022-05-11
title: "编写bcc程序提取内核进程级TCP网络流量"
categories: [
    "linux",
    "os",
    "性能优化",
]
---

了解Linux内核，才能知道eBPF需要去追踪/监控哪个内核函数。

## TCP协议收发数据关键函数执行流

左边接收数据，右边发送数据。以当前任务为例，关键函数是tcp_recvmsg和tcp_tcp_sendmsg（监控函数）。

![](https://halfbit.oss-cn-hangzhou.aliyuncs.com/tcp1.png)

## 流程

需要获取的是第三个参数。

进程级流量原始数据获取以后，可以拿来做监控/优化，看具体任务。

![](https://halfbit.oss-cn-hangzhou.aliyuncs.com/liucheng.png)

## eBPF Map

![](https://halfbit.oss-cn-hangzhou.aliyuncs.com/map2.png)



## 开销

和上图的对比。接收数据的函数变成了tcp_cleanup_rbuf这个函数。**ebpf程序开销最大的应该就是函数的触发次数。**如果一个函数，一分钟只触发几次，那开销就很小；如果一分钟触发成百上千上万次，开销就很大。这里需要对linux内核的熟悉，下面这个函数的第二次字段，等价于被成功接收的数据的大小，等价于释放数据的大小。下面这个函数触发频率低，所以监控下面的。

![](https://halfbit.oss-cn-hangzhou.aliyuncs.com/kaixiao.png)

bcc编写ebpf开销还是比较大的，开销小的话最好用libbpf（重c）来编写ebpf程序。两个，kernel.c(会被编译成bpf 程序，也就是kernel.o作为目标文件，通过user.c注入到内核中，然后user.c拿数据。

tracepoint开销小于kprobe。

## bcc程序

存放在tcp_list/recv_send.py

## 可视化展示eBPF提取的数据

eBPF exporter基于bcc。通过配置文件。

![](https://halfbit.oss-cn-hangzhou.aliyuncs.com/keshihua.png)

## 参考

http://www.vinin.me/2022/04/10/Hello-eBPF/#more