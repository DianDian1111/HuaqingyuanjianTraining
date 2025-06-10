# HuaqingyuanjianTraining
## Day 1
一.练习git基本操作，同时将Pycharm与GitHub连接起来，
![1.png](../Screenshot/Day1/1.png)
![2.png](../Screenshot/Day1/2.png)
![3.png](../Screenshot/Day1/3.png)
![4.png](../Screenshot/Day1/4.png)
![5.png](../Screenshot/Day1/5.png)
![6.png](../Screenshot/Day1/6.png)

二.使用遥感图像处理工具（如rasterio、Pillow等）对多波段TIFF遥感数据进行处理，生成可视化的真彩色图像。
1.理解遥感图像中不同波段（如B02、B03、B04）与颜色（蓝、绿、红）之间的对应关系；
2.实现RGB图像的合成与归一化处理，确保输出图像在 0-255 的显示范围内；
3.利用Pillow将NumPy图像数组保存为常见图像格式（如.jpg），并进行可视化展示；
实现：
1.使用rasterio.open().read()实现对多波段遥感数据的批量读取
2.使用numpy对红绿蓝段进行归一化处理
3.使用PIL.Image.fromarray()将结果保存为jpeg格式以及用matplotlib显示图像
注：遥感图像波段值常常不在 0~255 范围内，必须归一化才能正确显示。
