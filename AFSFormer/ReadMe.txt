1.数据集的预处理：
数据集需要自己制作，将数据集裁剪为1024*1024大小的图像，对应的mask图像也需要裁剪为1024*1024大小。
可以做数据增广，随机翻转等操作。
将处理好的数据放在data文件下

2.训练：
训练与测试过程需要用命令行运行：
python train_supervision.py -c config/flame/FireFormer
(train_supervision.py为训练代码，命令行参数-c代表要训练的模型）

3.测试
python flame_test.py -c config/flame/FireFormer -o fig_results/flame/FireFormer -t 'lr' --rgb
(flame_test.py为测试代码，使用别的数据集的话需要自己修改对应代码。
-c 代表需要测试对应模型的路径
-o 代表将模型推理结果的图像输出到哪个路径中
-t 代表测试阶段是否需要进行图像增广，如果需要，设置为:d4；如不需要,设置为:lr
--rgb 代表将推理的图像以rgb形式导出