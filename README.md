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


## TODO list：

1.拉取项目包
git clone xxxxxx
2.安装环境
python -m venv myVenv
3.激活环境
myVenv/scripts/activate
4.安装必要库
python install -r requirements.txt
5.修改路径:
SAM模型路径
测试样例路径
    SAM-canny-final.py（批量图片处理）：将图片目录改为真实目录
    SAM-canny-test.py（单个图片处理）：将图片文件路径改为真实路径
输出目录
6.python SAM-canny-final.py




