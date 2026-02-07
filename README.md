# 纸张边缘检测算法
技术路线选型：SAM＋canny边缘检测算法+OpenCV 轮廓检测与角点筛选 
## 项目文件结构介绍
### history_code
存放了调研过程中的历史代码
可以快速验证和复现结果
（需要修改代码中的图片路径）
### sam_weights
存放sam权重文件（base版）
### test
保存了测试样例和效果展示


## Quick start：

#### 1.拉取项目包
```
  git clone https://github.com/helloGitHub-art/documentDivision.git
```
    
#### 2.安装环境
```
  python -m venv myVenv
```
    
#### 3.激活环境
```
  myVenv/scripts/activate
```

#### 4.安装必要库
```
  python install -r requirements.txt
```

#### 5.修改路径
SAM-canny-final.py（批量图片处理）：将图片目录改为真实目录
SAM-canny-test.py（单个图片处理）：
将图片文件路径改为真实路径
另外修改输出的保存目录

#### 6.运行
```
  python SAM-canny-final.py
```

## 结果展示
可见test\result目录下的结果
![图片1](\test\result\49e155ad-6fd150513ccb9e48fe819baf7933cc39_sam_annotated.png "image1")
![图片2](\test\result\4654fad7-8037e6e94a80daf1be21e691bdd88caf_sam_annotated.png "image")
![图片3](\test\result\d9619fe3-ca0d4e0dd63a66c1dd02089d630b53fc_sam_annotated.png "image3")
![图片4](\test\result\image1_sam_annotated.png "image4")
![图片5](\test\result\image4_sam_annotated.png "image5")
![图片6](\test\result\image5_sam_annotated.png "image6")




