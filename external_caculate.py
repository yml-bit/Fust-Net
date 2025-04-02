import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import scipy.io as scio
import random
import pandas as pd
from openpyxl import load_workbook
import h5py

# from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def read_total(file_path):
    with open(file_path, "r") as tf:
        lines = tf.read().split("\n")
    # 过滤空白行
    lines = [line for line in lines if line.strip()]
    # 清洗数据并转换为浮点数列表
    matrix_data = []
    for line in lines:
        # 移除开头和结尾的方括号
        line_cleaned = line.strip()[1:-1].strip()
        numbers = line_cleaned.split()
        matrix_data.append([float(num) for num in numbers])
    # 将列表转换为numpy数组
    matrix = np.array(matrix_data)
    return matrix

def read_subject(file_path):
    with open(file_path, "r") as tf:
        lines = tf.read().split("\n")
    # 过滤空白行
    # lines = [line for line in lines if line.strip()]
    # 清洗数据并转换为浮点数列表
    matrix_list = []
    matrix_data = []
    for line in lines:
        # 移除开头和结尾的方括号
        line_cleaned = line.strip()[1:-1].strip()
        numbers = line_cleaned.split()
        if len(numbers)==0:
            matrix = np.array(matrix_data)
            matrix_list.append(matrix)
            matrix_data = []
        else:
            matrix_data.append([float(num) for num in numbers])
    return matrix_list

#提取每个患者第一组实验数据作为代表
def conform1(file_path):
    with open(file_path, "r") as tf:
        lines = tf.read().split("\n")
    # 过滤空白行
    # lines = [line for line in lines if line.strip()]
    # 清洗数据并转换为浮点数列表
    matrix_list = []
    matrix_data = []
    for line in lines:
        # 移除开头和结尾的方括号
        line_cleaned = line.strip()[1:-1].strip()
        numbers = line_cleaned.split()
        if len(numbers)==0:
            matrix = np.array(matrix_data)
            matrix_list.append(matrix[0,:])
            matrix_data = []
        else:
            matrix_data.append([float(num) for num in numbers])
    matrixs=np.array(matrix_list)
    return matrixs

def conform3(file_path):
    with open(file_path, "r") as tf:
        lines = tf.read().split("\n")
    # 过滤空白行
    # lines = [line for line in lines if line.strip()]
    # 清洗数据并转换为浮点数列表
    matrix_list = []
    matrix_data = []
    for line in lines:
        # 移除开头和结尾的方括号
        line_cleaned = line.strip()[1:-1].strip()
        numbers = line_cleaned.split()
        if len(numbers)==0:
            matrix = np.array(matrix_data)
            matrix_list.append(matrix[2,:])
            matrix_data = []
        else:
            matrix_data.append([float(num) for num in numbers])
    matrixs=np.array(matrix_list)
    return matrixs

def conform5(file_path):
    with open(file_path, "r") as tf:
        lines = tf.read().split("\n")
    # 过滤空白行
    # lines = [line for line in lines if line.strip()]
    # 清洗数据并转换为浮点数列表
    matrix_list = []
    matrix_data = []
    for line in lines:
        # 移除开头和结尾的方括号
        line_cleaned = line.strip()[1:-1].strip()
        numbers = line_cleaned.split()
        if len(numbers)==0:
            matrix = np.array(matrix_data)
            matrix_list.append(matrix[4,:])
            matrix_data = []
        else:
            matrix_data.append([float(num) for num in numbers])
    matrixs=np.array(matrix_list)
    return matrixs

def conform7(file_path):
    with open(file_path, "r") as tf:
        lines = tf.read().split("\n")
    # 过滤空白行
    # lines = [line for line in lines if line.strip()]
    # 清洗数据并转换为浮点数列表
    matrix_list = []
    matrix_data = []
    for line in lines:
        # 移除开头和结尾的方括号
        line_cleaned = line.strip()[1:-1].strip()
        numbers = line_cleaned.split()
        if len(numbers)==0:
            matrix = np.array(matrix_data)
            matrix_list.append(matrix[6,:])
            matrix_data = []
        else:
            matrix_data.append([float(num) for num in numbers])
    matrixs=np.array(matrix_list)
    return matrixs

def conform_last(file_path):
    data_list=read_subject(file_path)
    matrix_data = []
    for sub_data in data_list:
        matrix_data.append(sub_data[-1, :])
    matrixs=np.array(matrix_data)
    return matrixs

def metric(data):
    # print(data)
    if isinstance(data,list):
        i=0
        for sub_data in data:
            print("patient：",i)
            i=i+1
            true_values = sub_data[:,0]  # 请替换为实际真实值数组
            predicted_values=sub_data[:,1]
            absolute_errors = np.abs(predicted_values - true_values) # 计算绝对误差
            squared_errors = (predicted_values - true_values) ** 2# 计算平方误差
            # 计算误差的方差（这里的误差指的是平方误差，通常在计算均方误差时使用）
            variance_of_errors = np.var(squared_errors)
            # 输出结果
            print("Absolute Errors: ", absolute_errors) #
            print("Squared Errors: ", squared_errors)
            print("Variance of Errors: ", variance_of_errors)#求解误差的方差，以此来衡量预测值与真实值误差的波动情况
    else:
        true_values = data[:, 0]  # 请替换为实际真实值数组
        predicted_values = data[:, 1]
        absolute_errors = np.abs(predicted_values - true_values)  # 计算绝对误差
        # squared_errors = (predicted_values - true_values) ** 2  # 计算平方误差
        # 计算误差的方差（这里的误差指的是平方误差，通常在计算均方误差时使用）
        variance_of_errors = np.var(absolute_errors)
        # std_deviation = np.std(absolute_errors)
        f1 = r2_score(true_values, predicted_values)
        r2 = r2_score(true_values, predicted_values)
        correlation_coefficient, p_value = pearsonr(true_values, predicted_values)

        # 输出结果
        print("Absolute Errors: ", absolute_errors.mean())
        # print("Squared Errors: ", squared_errors.mean())
        print("Variance of Errors: ", variance_of_errors)
        print(f"Coefficient of Determination (f1):",f1)
        print(f"Coefficient of Determination (R²):",r2)
        print("Pearson Correlation Coefficient: ", correlation_coefficient)
        print("p-value: ", p_value)
        print("\n ")

def T_test(data):
    if isinstance(data,list):
        i=0
        for sub_data in data:
            print("patient：",i)
            i=i+1
            true_values = sub_data[:,0]  # 请替换为实际真实值数组
            predicted_values=sub_data[:,1]
            t_statistic, p_value = ttest_ind(true_values, predicted_values, equal_var=True)
            # 输出t统计量和p值
            print("t-statistic: ", t_statistic)
            print("p-value: ", p_value)
            # 根据p值判断差异是否显著
            alpha = 0.05  # 设置显著性水平
            if p_value <= alpha:
                print("There is a statistically significant difference between male and female scores.")
            else:
                print(
                    "There is no statistically significant difference between male and female scores at α={:.2f} level.".format(
                        alpha))

    else:
        true_values = data[:, 0]  # 请替换为实际真实值数组
        predicted_values = data[:, 1]
        # 使用ttest_ind函数进行独立样本t检验
        t_statistic, p_value = ttest_ind(true_values, predicted_values, equal_var=True)
        # 输出t统计量和p值
        print("t-statistic: ", t_statistic)
        print("p-value: ", p_value)
        # 根据p值判断差异是否显著
        alpha = 0.05  # 设置显著性水平
        if p_value <= alpha:
            print("There is a statistically significant difference between male and female scores.")
        else:
            print(
                "There is no statistically significant difference between male and female scores at α={:.2f} level.".format(
                    alpha))

def dot_plot(data):
    true_values = data[:, 0]  # 请替换为实际真实值数组
    predicted_values = data[:, 1]
    # 计算绝对误差
    errors = np.abs(predicted_values - true_values)

    # 创建散点图并根据误差大小着色
    fig, ax = plt.subplots()
    scatter = ax.scatter(true_values, predicted_values, c=errors, cmap='viridis', s=50, alpha=0.8)

    # 添加颜色条表示误差范围
    cbar = fig.colorbar(scatter, ax=ax, label='Absolute Error')
    # 添加对角虚线
    plt.plot([np.min(true_values), np.max(true_values)],
             [np.min(true_values), np.max(true_values)],
             'r--', label='Perfect Prediction Line')
    # 设置轴标签和标题
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    plt.legend()
    ax.set_title('Predictions vs Actual with Absolute Error')
    plt.show()

def histogrism(data):
    true_values = data[:, 0]  # 请替换为实际真实值数组
    predicted_values = data[:, 1]
    absolute_errors = np.abs(predicted_values - true_values)
    # 定义等级区间边界
    bins = [0, 1, 2, 3, 4, np.inf]
    bin_labels = ['0-1', '1-2', '2-3', '3-4', '>4']
    # 对数据进行离散化并计算频率分布
    counts, bin_edges = np.histogram(absolute_errors, bins=bins)
    # 计算每个等级所占百分比
    percentages = counts / counts.sum() * 100
    # 绘制柱状图，显示百分比
    fig, ax = plt.subplots()
    ax.bar(bin_labels, percentages, width=0.5)
    ax.set_xlabel('Grade Ranges')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Distribution of Data Across Grades')
    # 在柱子上方显示百分比数值
    for rect in ax.patches:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    (rect.get_x() + rect.get_width() / 2, height),
                    ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=45)  # 可选，让标签倾斜以便更好显示
    plt.show()

def globle_caculate():
    inpath="./GradientShap/GradientShap_weight1.txt"
    outpath = inpath.replace("weight", "nor")
    total=0
    for i in range(2,7):
        input_path=inpath.replace("1",str(i))
        with open(input_path, "r") as tf:
            lines = tf.read().split("\n")
        total+=len(lines)-1
    print(total)
    print("finished!")

def caculate_index():
    dot_path = ["model2_dur_energy","model3_dur_energy_sdr", "model4_dur_energy_size_sdr",
                "model5_dur_energy_area_size_sdr", "model6_dis_dur_energy_area_size_sdr",
                "model7_dis_dur_energy_area_size_sdr_avp_seq"]
    # his_path=["model1_dur_loss","model2_dur_energy_loss","model3_dur_energy_sdr_loss","model4_dur_energy_size_sdr_loss",
    #             "model5_dur_energy_area_size_sdr_loss","model6_dis_dur_energy_area_size_sdr_loss"]
    # input_path = "./GradientShap_weight1.txt"
    i=1
    for file in dot_path:
        # if i<5:
        #     i=i+1
        #     continue
        path = "./external_results/" + file + ".txt"
        conf = read_total(path)
        data = read_subject(path)  # 没有使用

        # 暂且不讨论每次刺激的情况
        # conf=conform1(path)
        # conf=conform3(path)
        # conf = conform5(path)
        # conf = conform7(path)
        print(file)
        metric(conf)
        dot_plot(conf)
        histogrism(conf)
        a = 1

def total_test():
    dot_path = ["model2_dur_energy", "model3_dur_energy_sdr", "model4_dur_energy_size_sdr",
                "model5_dur_energy_area_size_sdr", "model6_dis_dur_energy_area_size_sdr",
                "model7_dis_dur_energy_area_size_sdr_avp_seq"]
    # his_path=["model1_dur_loss","model2_dur_energy_loss","model3_dur_energy_sdr_loss","model4_dur_energy_size_sdr_loss",
    #             "model5_dur_energy_area_size_sdr_loss","model6_dis_dur_energy_area_size_sdr_loss"]
    # input_path = "./GradientShap_weight1.txt"
    i = 1
    for file in dot_path:
        # if i<5:
        #     i=i+1
        #     continue
        file_path = "./external_results/" + file + ".txt"

        # total = read_total(file_path)
        # metric(total)
        # dot_plot(total)
        # histogrism(total)

        conf = conform1(file_path)
        metric(conf)
        dot_plot(conf)
        histogrism(conf)

        conf = conform3(file_path)
        metric(conf)
        dot_plot(conf)
        histogrism(conf)

        # conf = conform5(file_path)
        # metric(conf)
        # dot_plot(conf)
        # histogrism(conf)

        # conf = conform7(file_path)
        # metric(conf)
        # dot_plot(conf)
        # histogrism(conf)
        #
        # conf = conform_last(file_path)
        # metric(conf)
        # dot_plot(conf)
        # histogrism(conf)


if __name__ == '__main__':
    caculate_index()
    # total_test() #各个次数的刺激,暂且用不上
    a=1