# 🚀快速开始

# 使用AutoDl服务器进行训练

## 1.选择合适服务器并且选择自己镜像

## （如果自己没有镜像，推荐使用社区镜像--Deformable_Detr）

## 2.创建自己的虚拟环境并安装对应 pytorch （没有使用镜像）

### 创建虚拟环境

```bash

# 创建名为 deformable-detr 的新环境，指定Python版本（3.8或3.9较稳定）

conda create -n deformable-detr python=3.8 -y

# 激活环境

conda activate deformable-detr

```
### 安装对应 pytorch（略）

### 选择合适 torchvision（避免报错）

```bash

# 先卸载现有版本
pip uninstall -y torch torchvision

# 安装兼容版本（CUDA 11.8 可用此命令，自动匹配兼容的 CUDA 版本）
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

```

#### 该版本可以兼容向上的cuda

#### 安装完成后，执行以下命令验证 _NewEmptyTensorOp 是否可导入：
```bash

python -c "from torchvision.ops.misc import _NewEmptyTensorOp; print('导入成功')"

```
## 3.使用指令克隆 github 上 deformable detr 的文件

```bash

git clone https://github.com/fundamentalvision/Deformable-DETR.git

```

### 安装官方需要依赖 

```bash

pip install -r requirements.txt

```
### 编译 cuda 环境

```bash

cd ./models/ops

sh ./make.sh

python test.py

# 用于验证

```

## 开始训练

回到 Deformable detr 所在根目录

训练

```bash

python main.py \
    --coco_path autodl-tmp/coco \
    --epochs 500 \
    --batch_size 2 \
    --num_workers 2 \
    --output_dir ./outputs/single_gpu \
    --with_box_refine \
    --two_stage \
    --num_queries 300

```

这里 coco_path 需要改到自己的目录

# 可能的错误

## 1.数据增强验证

使用 diagnose_structure(使用后判断诊断路径).py 寻找验证目录

再使用 label_checker(诊断数据增强是否成功).py 验证

## 2.数据增强后类别显示0.0浮点数导致无法验证

使用 class_float_fix.py 修复

## 3.yolo 转 coco 数据集后在服务器端显示路径错误

在服务器端复制使用 AutoDL_coco_fix.py 修复

## 4.训练时显示 _NewEmptyTensorOp 报错

说明之前修改的 torchvision 失败建议修改 misc.py 文件

可以复制粘贴 misc_fix.py 代替即可使用（修改后需要重新编译cuda算子）

1. 清除旧的编译结果
```bash

进入算子目录，删除之前编译生成的文件（避免残留文件干扰）：

cd /root/Deformable-DETR/models/ops  # 切换到算子目录
rm -rf build/  # 删除编译中间文件
rm -rf *.egg-info  # 删除之前安装的egg信息
rm -f MultiScaleDeformableAttention.cpython-*.so  # 删除生成的动态库

```
2. 重新编译算子

运行仓库提供的编译脚本（确保当前环境已激活，且 PyTorch 版本正确）：
```bash
sh make.sh  # 重新编译
注意编译过程中的输出：
如果出现 error: command 'gcc' failed 等错误，说明缺少编译工具，需安装
```
```bash
apt update && apt install -y gcc g++  # Ubuntu/Debian 系统
如果提示 PyTorch not found，需确认当前环境中 PyTorch 已正确安装（import torch 无报错）。
```

3. 验证编译结果
编译成功后，目录下会生成新的 .so 动态库（如 MultiScaleDeformableAttention.cpython-38-x86_64-linux-gnu.so）。此时重新运行训练命令：

# 脱离本地终端让服务器自动训练

nohup 是 Linux 系统中一个简单实用的命令，核心作用是让进程脱离终端独立运行，即使本地电脑关机、SSH 连接断开，服务器上的训练进程也能继续执行。下面详细讲解其使用步骤和原理：

## 一、nohup 核心原理

nohup 全称 "no hang up"（不挂断），它会忽略终端发送的 "挂断信号"（当本地断开 SSH 连接时，终端会向进程发送此信号，导致进程终止）。

通过 nohup 启动的进程会在后台运行，其输出日志会被重定向到文件（默认 nohup.out，也可自定义），方便后续查看训练状态。

## 二、详细使用步骤

### 1. 准备工作

确保已通过 SSH 登录服务器，并进入训练代码所在目录（例如 cd ~/Deformable-DETR）。

确认训练命令可正常运行（先在终端直接执行一次简短的训练，比如 python main.py --epochs 1，验证代码无语法错误、数据集路径正确等，避免后台启动后才发现问题）。

### 2. 用 nohup 启动训练

假设你的训练命令是：

```bash
python main.py \
    --coco_path /root/autodl-tmp/coco \
    --epochs 500 \
    --batch_size 2 \
    --num_workers 2 \
    --output_dir ./outputs/single_gpu \
    --with_box_refine \
    --two_stage \
    --num_queries 300

```
用 nohup 启动的完整命令为：

```bash
nohup python main.py \
    --coco_path /root/autodl-tmp/coco \
    --epochs 50 \
    --batch_size 2 \
    --num_workers 2 \
    --output_dir ./outputs/single_gpu \
    --with_box_refine \
    --two_stage \
    --num_queries 300 > train_log.txt 2>&1 &
```
命令参数解释：

nohup：核心命令，让后续的训练进程忽略终端挂断信号。

> train_log.txt：将训练过程中的标准输出（如 loss 打印、进度提示等）重定向到 train_log.txt 文件（避免日志丢失）。

2>&1：将错误输出（如报错信息、警告等）也重定向到 train_log.txt（方便统一查看问题）。

&：将进程放到后台运行（终端会立即返回，不阻塞操作）。

### 3. 验证进程是否启动成功

命令执行后，终端会显示一个进程 ID（PID），例如：

```plaintext
[1] 1293  # 5435 就是该进程的 PID
```

可通过以下命令验证进程是否在运行：

```bash
ps -ef | grep python  # 查看所有 python 进程
```

在输出中找到包含你的训练命令（如 main.py）的行，确认其状态为 R（运行中）。

### 4. 关闭本地电脑，服务器继续训练

此时无需保持 SSH 连接，直接关闭本地电脑即可。服务器上的训练进程会在后台持续运行，直到训练结束或被手动终止。

### 5. 后续查看训练状态（需重新登录服务器）

当你再次通过 SSH 登录服务器后，可通过以下方式查看训练进度：

实时查看日志（推荐）：

```bash
tail -f train_log.txt  # 实时显示日志末尾内容（按 Ctrl+C 退出查看）
```

适合监控训练进度（如当前 epoch、loss 变化等）。

查看完整日志：

```bash
cat train_log.txt  # 一次性显示所有日志（适合日志较短时）
## 或用分页查看（日志较长时）
less train_log.txt  # 按空格键翻页，按 q 退出
```

确认进程是否仍在运行：

```bash
ps -ef | grep python | grep main.py  # 精准查找训练进程
```

若输出为空，说明进程已终止（可能是训练完成或出错，需查看 train_log.txt 排查）。

6. 手动终止训练（如需提前停止）

如果需要中途停止训练，步骤如下：

查找进程 PID：

```bash
ps -ef | grep python | grep main.py
```

输出类似：

```plaintext
root  5435  1  3 10:00 ?  00:05:20 python main.py ...
```

其中 5435 就是 PID。

终止进程：

```bash
kill -9 5435  # 替换为实际的 PID
```

执行后，训练进程会立即停止。
