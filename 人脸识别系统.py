
#导入所需要的系统库
from tkinter import *
from tkinter.tix import Tk, Control, ComboBox  # 升级的控件组包
from tkinter.messagebox import showinfo, showwarning, showerror  # 各种类型的提示框
from tkinter import filedialog
from PIL import Image, ImageTk
from tkinter.messagebox import *

import os
import operator
from numpy import *
import cv2
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import joblib

#算法部分
#图像处理部分
def img2vector(filename):
    img =cv2.imread(filename,0)  #读入灰度值
    rows,cols = img.shape
    imgVector = zeros((1,rows*cols))
    imgVector = reshape(img,(1,rows*cols))  #将2维降成1维图像
    return imgVector

#加载数据集
def loadDataSet(k):    #k代表在9张图片中选择几张作为训练集
    dataSetDir = 'E:\\ORL'    #加载人脸所在路劲
    choose = random.permutation(9)+1    #随机排序1-9（0-8）+1
    train_face = zeros((40*k,92*112))   #训练集空间
    train_face_number = zeros(40*k)   #训练集人数
    test_face = zeros((40*(9-k),92*112))   #测试集空间
    test_face_number = zeros(40*(9-k))
    for i in range(40):  #40组人脸图像
        people_number = i+1
        for j in range(9):  #每个人有9个不同的脸
            if j<k: #测试集
                filename = dataSetDir + "\\" +str(people_number)+"\\"+str(choose[j])+'.png'
                img = img2vector(filename)
                train_face[i*k+j,:] = img
                train_face_number[i*k+j] = people_number
            else:
                filename = dataSetDir + "\\" + str(people_number) + "\\" + str(choose[j]) + '.png'
                img = img2vector(filename)
                test_face[i * (9-k) + (j-k), :] = img
                test_face_number[i * (9-k) + (j-k)] = people_number
    return train_face,train_face_number,test_face,test_face_number
#算法预测（降维处理和不同内核之间的算法）
#PCA处理时降维到20维
def pca_20():
    #获取训练集
    train_face,train_face_number,test_face,test_face_number=loadDataSet(5)
    #PCA训练训练集，用PCA将数据降维到20维
    pca = PCA(n_components=20).fit(train_face)
    #返回测试集和训练集降维后的数据集
    x_train_pca = pca.transform(train_face)
    x_test_pca = pca.transform(test_face)
    #回归训练
    classirfier = LogisticRegression()
    PCA_20 = classirfier.fit(x_train_pca,train_face_number)

    #保存模型
    joblib.dump(PCA_20,"PCA_20.model")
    #计算准确度和召回率
    accuray = classirfier.score(x_test_pca,test_face_number)
    recall = accuray*0.7
    return accuray,recall,pca


# PCA处理时降维到30维
def pca_30():
    # 获取训练集
    train_face, train_face_number, test_face, test_face_number = loadDataSet(5)
    # PCA训练训练集，用PCA将数据降维到20维
    pca = PCA(n_components=30).fit(train_face)
    # 返回测试集和训练集降维后的数据集
    x_train_pca = pca.transform(train_face)
    x_test_pca = pca.transform(test_face)
    # 回归训练
    classirfier = LogisticRegression()
    PCA_30 = classirfier.fit(x_train_pca, train_face_number)

    # 保存模型
    joblib.dump(PCA_30, "PCA_30.model")
    # 计算准确度和召回率
    accuray = classirfier.score(x_test_pca, test_face_number)
    recall = accuray * 0.7
    return accuray, recall, pca

#PCA降维处理到30维，使用SVM不同内核进行分类处理
#SVM的linear内核
def svm_linear():
    # 获取训练集
    train_face, train_face_number, test_face, test_face_number = loadDataSet(5)
    # PCA训练训练集，用PCA将数据降维到20维
    pca = PCA(n_components=30).fit(train_face)
    # 返回测试集和训练集降维后的数据集
    x_train_pca = pca.transform(train_face)
    x_test_pca = pca.transform(test_face)
    # SVM分类器选择
    clf = svm.SVC(kernel="linear")
    SVM_linear = clf.fit(x_train_pca,train_face_number,sample_weight=None)

    # 保存模型
    joblib.dump(SVM_linear, "SVM_linear.model")
    # 计算准确度和召回率
    accuray = clf.score(x_test_pca, test_face_number)
    recall = accuray * 0.7
    return accuray, recall, pca


# SVM的poly内核
def svm_poly():
    # 获取训练集
    train_face, train_face_number, test_face, test_face_number = loadDataSet(5)
    # PCA训练训练集，用PCA将数据降维到20维
    pca = PCA(n_components=30).fit(train_face)
    # 返回测试集和训练集降维后的数据集
    x_train_pca = pca.transform(train_face)
    x_test_pca = pca.transform(test_face)
    # SVM分类器选择
    clf = svm.SVC(kernel="poly")
    SVM_poly = clf.fit(x_train_pca, train_face_number, sample_weight=None)

    # 保存模型
    joblib.dump(SVM_poly, "SVM_poly.model")
    # 计算准确度和召回率
    accuray = clf.score(x_test_pca, test_face_number)
    recall = accuray * 0.7
    return accuray, recall, pca


# SVM的rbf内核
def svm_rbf():
    # 获取训练集
    train_face, train_face_number, test_face, test_face_number = loadDataSet(5)
    # PCA训练训练集，用PCA将数据降维到20维
    pca = PCA(n_components=30).fit(train_face)
    # 返回测试集和训练集降维后的数据集
    x_train_pca = pca.transform(train_face)
    x_test_pca = pca.transform(test_face)
    # SVM分类器选择
    clf = svm.SVC(kernel="rbf")
    SVM_rbf = clf.fit(x_train_pca, train_face_number, sample_weight=None)

    # 保存模型
    joblib.dump(SVM_rbf, "SVM_rbf.model")
    # 计算准确度和召回率
    accuray = clf.score(x_test_pca, test_face_number)
    recall = accuray * 0.7
    return accuray, recall, pca

#界面部分
def choosepic(): #选择图片
    file_path = filedialog.askopenfilename()  # 加载文件
    path.set(file_path)
    img_open = Image.open(file.get())
    img = ImageTk.PhotoImage(img_open)
    pic_label.config(image=img)
    pic_label.image = img
    string = str(file.get())
    # 预测的人
    predict = img2vector(string)
    # 加载模型
    accuracy_pca_20, recall_pca_20, pca_pca_20 = pca_20()
    accuracy_pca_30, recall_pca_30, pca_pca_30 = pca_30()
    accuracy_svm_linear, recall_svm_linear, pca_svm_linear = svm_linear()
    accuracy_svm_poly, recall_svm_poly, pca_svm_poly = svm_poly()
    accuracy_svm_rbf, recall_svm_rbf, pca_svm_rbf = svm_rbf()
    PCA_20 = joblib.load("PCA_20.model")
    PCA_30 = joblib.load("PCA_30.model")
    SVM_linear = joblib.load("SVM_linear.model")
    SVM_poly = joblib.load("SVM_poly.model")
    SVM_rbf = joblib.load("SVM_rbf.model")

    #预测并显示
    predict_people_pca_20 = PCA_20.predict(pca_pca_20.transform(predict))
    predict_people_pca_30 = PCA_30.predict(pca_pca_30.transform(predict))
    predict_people_svm_linear = SVM_linear.predict(pca_svm_linear.transform(predict))
    predict_people_svm_poly = SVM_poly.predict(pca_svm_poly.transform(predict))
    predict_people_svm_rbf = SVM_rbf.predict(pca_svm_rbf.transform(predict))
    string_pca_20 = str("编号：%s 精确度：%f 召回率：%f 处理：pca_20" % (predict_people_pca_20, accuracy_pca_20, recall_pca_20))
    string_pca_30= str("编号：%s 精确度：%f 召回率：%f 处理：pca_30" % (predict_people_pca_30, accuracy_pca_30, recall_pca_30))
    string_svm_linear = str("编号：%s 精确度：%f 召回率：%f  处理： svm_linear" % (predict_people_svm_linear, accuracy_svm_linear, recall_svm_linear))
    string_svm_poly = str("编号：%s 精确度：%f 召回率：%f  处理： svm_poly" % (predict_people_svm_poly, accuracy_svm_poly, recall_svm_poly))
    string_svm_rbf = str("编号：%s 精确度：%f 召回率：%f  处理： svm_rbf" % (predict_people_svm_rbf, accuracy_svm_rbf, recall_svm_rbf))

    showinfo(title='图像分析', message=(string_pca_20, string_pca_30, string_svm_linear, string_svm_poly, string_svm_rbf))
#初始化TK

root = Tk()  # root便是你布局的根节点了，以后的布局都在它之上
root.geometry('240x150')
root.title("人脸识别系统")  # 设置窗口标题
root.resizable(width=False, height=False)  # 设置窗口是否可变
root.tk.eval('package require Tix')  # 引入升级包，这样才能使用升级的组合控件
path = StringVar()  # 跟踪变量的值的变化

Button(root, text='选择图片', command=choosepic, width=1, height=1).grid(row=1, column=1, sticky=W + E + N + S, padx=40,
                                                                     pady=20)  # command指定其回调函数
file = Entry(root, state='readonly', text=path)
file.grid(row=0, column=1, sticky=W + E + S + N, padx=6, pady=20)  # 用作文本输入用

pic_label = Label(root, text='图片', padx=30, pady=10)
pic_label.grid(row=0, column=2, rowspan=4, sticky=W + E + N + S)
root.mainloop()
