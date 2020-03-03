# 函数名称：select_knn.py
# 函数功能：选择任务卸载位置（本地 or MEC/cloud）。
# 考虑每个服务请求任务的延迟要求和任务大小，找到最符合应该卸载的平台
#
# 输入参数：训练样本，预测样本
# 输出参数：预测样本应该卸载的平台
#
# 运行环境：Windows 10 Python 3.6.2
# 修改日期      版本号      修改人                          修改内容
# 2018/09/01   v0.0.0     梁颖杰                   创建
# 2018/09/25   v0.0.1     梁颖杰                   添加注释，修改训练样本参数


from sklearn import neighbors
from config import *
import numpy as np
class select_knn():
    knn = neighbors.KNeighborsClassifier()  # 取得knn分类器
    train_data=''
    def __init__(self):
        self.train_data=[] #初始化训练样本
        return
    def get_train_data(self,path):     #文件操作，读取训练样本
        f=open(path,'r')
        line_list=f.readlines()
        f.close()
        train_set = []
        lables = []
        for i in range(0, len(line_list)):
            line = line_list[i].strip('\n')
            line = line.split(',')
            a=float(line[0])
            b=float(line[1])
            c=float(line[2])
            train_set.append([a,b])
            lables.append(c)
        self.train_data = [np.array(train_set),np.array(lables)]  # data对应着时延和任务大小
    def train(self):
        self.knn.fit(self.train_data[0], self.train_data[1])  # 导入数据进行训练
    def predict(self,task_time,task_cpusize):    #对预测数据集进行卸载位置预测
        task_point=[[task_time, task_cpusize]]
        final_result = int(self.knn.predict(task_point))
        return final_result
