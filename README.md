# HuaqingyuanjianTraining
## Day 1
### 练习git基本操作，同时将Pycharm与GitHub连接起来，
![1.
png](../Screenshot/Day1/1.png)
![2.png](../Screenshot/Day1/2.png)
![3.png](../Screenshot/Day1/3.png)
![4.png](../Screenshot/Day1/4.png)
![5.png](../Screenshot/Day1/5.png)
![6.png](../Screenshot/Day1/6.png)
![7.png](../Screenshot/Day1/7.png)
![8.png](../Screenshot/Day1/8.png)
![9.png](../Screenshot/Day1/9.png)
![10.png](../Screenshot/Day1/10.png)
![11.png](../Screenshot/Day1/11.png)
### 使用遥感图像处理工具（如rasterio、Pillow等）对多波段TIFF遥感数据进行处理，生成可视化的真彩色图像。
1.理解遥感图像中不同波段（如B02、B03、B04）与颜色（蓝、绿、红）之间的对应关系；
2.实现RGB图像的合成与归一化处理，确保输出图像在 0-255 的显示范围内；
3.利用Pillow将NumPy图像数组保存为常见图像格式（如.jpg），并进行可视化展示；
实现：
1.使用rasterio.open().read()实现对多波段遥感数据的批量读取
2.使用numpy对红绿蓝段进行归一化处理
3.使用PIL.Image.fromarray()将结果保存为jpeg格式以及用matplotlib显示图像
注：遥感图像波段值常常不在 0~255 范围内，必须归一化才能正确显示。

## Day2
### 深度学习基础
欠拟合：训练训练数据集表现不好，验证表现不好
过拟合：训练数据训练过程表现得很好，在我得验证过程表现不好
卷积过程[nn_conv.py](2022/Day2/Fundamentals_of_Deep_Learning/nn_conv.py)
#### 积运算的输出计算
5*5的输入数据 3*3的卷积核 步长1 填充1，输出5x5 
输出尺寸=⌊ N+2P−K /S ⌋+1
#### 图片卷积
[nn_conv2d.py](2022/Day2/Fundamentals_of_Deep_Learning/nn_conv2d.py)
#### tensorboard使用
使用tensorboard命令打开
tensorboard --logdir= 自己的绝对路径
![2_2.png](../Screenshot/Day2/2_2.png)
![2_3.png](../Screenshot/Day2/2_3.png)
#### 池化层
代码里面是最大池化，还有平均池化
[pooling_layer.py](2022/Day2/Fundamentals_of_Deep_Learning/pooling_layer.py)
