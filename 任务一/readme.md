@[toc](【任务1 - 前期预备工作】时长：3天)
# Anaconda 安装
# Conda 学习
# Python编辑器安装与学习： jupyter notebook  或者 pycharm 
# Tensorflow 库安装与学习
* （1）打开Anaconda Prompt，输入清华仓库镜像，这样更新会快一些：
这里写图片描述
```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes
```
* （2）同样在Anaconda Prompt中利用Anaconda创建一个python3.5的环境，环境名称为tensorflow ，输入下面命令：
```
    conda create -n tensorflow python=3.6
```
* （3）在Anaconda Prompt中启动tensorflow环境：
```
activate tensorflow
```
* （4）安装cpu版本的TensorFlow
```
pip install --upgrade --ignore-installed tensorflow
pip install tensorflow==1.9.0
```
* (5) 验证
```angular2html
    打印框架版本号
```
# GPU 安装
参考：https://blog.csdn.net/lwj_12345678/article/details/79419981
* 根据版本选择
* cuda https://developer.nvidia.com/cuda-80-download-archive
* Cudnn下载地址：https://developer.nvidia.com/rdp/cudnn-download
* pip install tensorflow-gpu  # stable
* 下载安装
pip install --upgrade https://link.zhihu.com/?target=https%3A//storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-0.12.0-cp35-cp35m-win_amd64.whl
* 验证
```angular2
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```
* 参考资料 
Anaconda介绍、安装及使用教程(https://zhuanlan.zhihu.com/p/32925500)
PyCharm 安装教程（Windows）{http://www.runoob.com/w3cnote/pycharm-windows-install.html}
Jupyter Notebook介绍、安装及使用教程{https://www.jianshu.com/p/91365f343585}
手把手教你如何安装Tensorflow（Windows和Linux两种版本）{https://blog.csdn.net/Cs_hnu_scw/article/details/79695347}
TensorFlow学习笔记1：入门{http://www.jeyzhang.com/tensorflow-learning-notes.html}
推荐书籍：《TensorFlow实战Google深度学习框架》
