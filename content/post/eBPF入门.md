---
author: "momo"
date: 2022-05-11
title: "eBPF入门"
categories: [
    "linux",
    "eBPF",
    "os",
    "性能优化",
]
---

不区分BPF/eBPF。

## eBPF功能

1、跟踪，通过编写kpobes、tracepoints等bpf程序跟踪内核函数；

2、观测，提取内核数据；

3、安全。

4、网络。如XDP。

5、性能调优。

## eBPF整体架构（跟踪内核的原理）

![](https://halfbit.oss-cn-hangzhou.aliyuncs.com/ebpf_framework.png)

## eBPF program

eBPF需要编写三个部分的⼯作，⽤户逻辑，加载逻辑，内核逻辑。

用户逻辑（frontend logic）：建立用户空间和内核的数据交互对象Map，声明eBPF需要执行的位置，获取内核输出（TCP包数、进程调用数）。

内核逻辑（backend logic，图中绿色的模块）：两种方式。1、用户自定义的受限的逻辑（条件判断、运行），2.eBPF提供的helper函数，使用内核的一些系统函数。

加载逻辑（loader logic）：确定用户声明的eBPF内核逻辑加载到内核的位置，bytecode加载到内核的eBPF VM中。

eBPF program 示例（C）：

```c
frontend_logic // prepare shared map 
int backend_logic(void *ctx){
	backend_logic 
} 
load(backend_logic, type, location) // load eBPF into kernel 
bind_hook // bind some events using provided hook 
frontend_logic // deal output
```

bpf program可以用c、go编写，也可以用bcc（是一个工具集，bpf的前端，可以用python程序写；也可以自己编写bcc程序），bpftrace（命令行）编写。

eBPF program怎样加载到内核？通过系统调用bpf（）。可以使用bpf_prog_load函数。

## Clang+LLVM

编译器的前端和后端，是apple公司开发⽤来替换gcc的编译器。

Clang：词法分析，语法分析，类型检查，中间代码生成。LLVM：中间代码优化，目标代码生成，对象是CPU（机器码）还是VM（VM支持的码）

流程：Clang把前端语⾔（c、python等）转化为中间语⾔IR，经过优化之后由LLVM转化为⽬标代码（bytecode）。这⾥因为要做到跨平台，所以⽬标代码是Bytecode。

## eBPF Bytecode

eBPF Bytecode是⼀类遵循RISC指令集的⾃定义指令集。

## Map共享数据

BPF Maps。这个空间可以同时被内核和用户访问。内核和⽤户态需要共享数据通过Map的⽅式将数据回传给⽤户空间，Map即key-value存储⽅式， 内核空间创造Map并返回描述符，⽤户空间可以（编写python等程序）直接获取值。也可以把用户态的数据（配置）放到内核。eBPF程序不但可以把数据放到map，也可以把map里的数据读取出来。

## Verifier

验证器，测试可能执⾏的路径，确保不会出现内核死锁，同时监测寄存器状态、程序⼤⼩和地址边界等问题。如果不安全，会执行出错。如果报错，会继续执行。防止写的bpf程序破坏内核。

## JIT

即时编译器。把bytecode翻译成机器码（native code）。

到这里就是注入内核。

## 跟踪函数

当内核函数kernel functions触发（开始执行）时，就会执行ebf程序。将这个函数的执行入口做一个跳转/中断/异常，转去执行bpf程序，bpf程序执行完以后，再跳转到内核函数。

## 编程实例

根据之前所示，编程分为frontend logic，loader logic，backend logic。

| 名称           | 工作状态 | 作用                                                         | 编写方式                                     |
| -------------- | -------- | ------------------------------------------------------------ | -------------------------------------------- |
| frontend logic | 用户态   | 组织Map结构，处理数据结果                                    | python（借助bcc框架），lua（借助bcc框架），c |
| loader logic   | 系统调用 | 把backend logic编译成 bytecode传入eBPF VM中，同时确定附着事件的位置。 | python（借助bcc框架），lua（借助bcc框架），c |
| backend logic  | 内核态   | 在内核态进行数据处理                                         | 受限c，assembler                             |

## 借助bcc编程框架

```python
#!/usr/bin/python
from bcc import BPF

# backend c语言编写，就是bpf程序，最终会被编译成bytecode，注入内核中执行
prog = """
    #include <asm/ptrace.h>
    #include <linux/fs.h>

    int trace_vfs_write(struct pt_regs *ctx, struct file *file, const char __user *buf, size_t count, loff_t *pos)
    {
        u32 pid = bpf_get_current_pid_tgid() >> 32;
        u32 uid = bpf_get_current_uid_gid();

        bpf_trace_printk("%d %d %s\\n", pid, file->f_inode->i_ino, file->f_path.dentry->d_iname);
        return 0;
    }
"""

# loader bpf程序注入内核
b = BPF(text = prog)
b.attach_kprobe(event = "vfs_write", fn_name = "trace_vfs_write")

print("%-18s %-16s %-6s %s"%("TIME(s)", "COMM", "PID", "MESSAGE"))

# frontend 把提取的数据拿出来
while 1:
    try:
        (task, pid, cpu, flags, ts, msg) = b.trace_fields()
    except ValueError:
        continue
    print("%-18.9f %-16s %-6d %s"%(ts, task, pid, msg))

```

## 钩⼦函数整体分类

• kernel functions (kprobes)

• userspace functions (uprobes)

• system calls

• fentry/fexit

• Tracepoints

• network devices (tc/xdp)

• network routes

• TCP congestion algorithms

## 参考资料

bcc开发实例：https://blog.cyru1s.com/posts/ebpf-bcc.html

bcc&bpftrace：https://mp.weixin.qq.com/s?__biz=MzA3NjY2NzY1MA==&mid=2649740037&idx=1&sn=48333af185d61a7bdefc8f55943929b7&chksm=8746bd68b031347ede1db4e23a526bceab7a216db11d56a899e133f616e8728dcac243125d0d&scene=21&token=1525812246&lang=zh_CN#wechat_redirect