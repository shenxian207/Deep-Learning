"""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869
    2017 Loic Landrieu, Martin Simonovsky
"""
import argparse
import numpy as np
import sys
sys.path.append("./learning")
from metrics import *  ##metrics库是一个对矩阵值进行相关处理的库

#建立解析对象
parser = argparse.ArgumentParser(description='Evaluation function for S3DIS')  #S3DIS的评估功能

#增加parser属性
#name of flags参数，为选项参数或位置参数；default指定参数默认值，help描述选项作用
parser.add_argument('--odir', default='./results/s3dis/best', help='Directory to store results')#存储结果的目录
parser.add_argument('--cvfold', default='123456', help='which fold to consider')#考虑哪个折叠

#将parser赋给实例args
args = parser.parse_args()

#建立混淆矩阵
C = ConfusionMatrix
C.number_of_labels = 13
C.confusion_matrix=np.zeros((C.number_of_labels, C.number_of_labels))#13x13的矩阵的值都赋0

#定义字典，实体用对应的数字表示
class_map = {0:'ceiling', 1:'floor', 2:'wall', 3:'column', 4:'beam', 5:'window', 6:'door', 7:'table', 8:'chair', 9:'bookcase', 10:'sofa', 11:'board', 12:'clutter'}

#保存数据，np.load函数用于读取数据
#“+”号后面是读取的路径
if '1' in args.cvfold:
    C.confusion_matrix+=np.load(args.odir+'/cv1/pointwise_cm.npy')
if '2' in args.cvfold:
    C.confusion_matrix+=np.load(args.odir+'/cv2/pointwise_cm.npy')
if '3' in args.cvfold:
    C.confusion_matrix+=np.load(args.odir+'/cv3/pointwise_cm.npy')
if '4' in args.cvfold:
    C.confusion_matrix+=np.load(args.odir+'/cv4/pointwise_cm.npy')
if '5' in args.cvfold:
    C.confusion_matrix+=np.load(args.odir+'/cv5/pointwise_cm.npy')
if '6' in args.cvfold:
    C.confusion_matrix+=np.load(args.odir+'/cv6/pointwise_cm.npy')
   
    
#get_overall_accuracy函数，返回矩阵对角线上所有值的和与矩阵所有值的比值
#get_mean_class_accuracy函数，返回 “每一行对角线上的值与 “该行的所有值的和与1中的较大值” 的比值” 的和除以矩阵值的总数的商
#get_inersection_union_per_class函数，返回值为矩阵对角线的值与“该矩阵对角线值对应行和列所有值之和”的比值矩阵，类型为float类型
#“%” 表示将其后的值转化为百分数的形式
print("\nOverall accuracy : %3.2f %%" % (100 * np.mean(ConfusionMatrix.get_overall_accuracy(C))))
print("Mean accuracy    : %3.2f %%" % (100 * np.mean(ConfusionMatrix.get_mean_class_accuracy(C))))
print("Mean IoU         : %3.2f %%\n" % (100 * np.mean(ConfusionMatrix.get_intersection_union_per_class(C))))
print("     Classe :  mIoU")
for c in range(0,C.number_of_labels):
    print ("   %8s : %6.2f %%" %(class_map[c],100*ConfusionMatrix.get_intersection_union_per_class(C)[c]))
