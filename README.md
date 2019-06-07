### Annotations.mat
在[官网](http://sunai.uoc.edu/emotic/download.html)上下载的数据集标注

### mat2dict.py
将Annotations.mat转换为Python的字典，并保存为二进制文件Annotations.pkl，以便训练时数据的加载

### show_data.py
展示数据集中的数据信息（主要目的是展示该字典的结构及从中获取数据信息的方法）

### mode.py
神经网络模型

### resnet50_places365.pth.tar
在[github](https://github.com/CSAILVision/places365) 下载的基于Places数据集预训练的ResNet50模型参数

### train.py
训练神经网络并将训练后参数保存在net.pth中

### test.py
在测试集上进行测试并展示结果（Jaccard coefficient的频率分布直方图）