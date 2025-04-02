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
import shutil
from scipy import stats
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def read_total(file_path):
    with open(file_path, "r") as tf:
        matrix_data = []
        for line in tf:
            line = line.strip()
            if not line or line.startswith("Mean Error:"):
                continue
            try:
                line_cleaned = line[1:-1].strip()
                numbers = [float(num) for num in line_cleaned.split()]
                matrix_data.append(numbers)
            except (IndexError, ValueError):
                continue
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

def metric(data,record,mm):
    # f_test = open(record, "a")
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
        # 初始化Bootstrap参数
        n_bootstrap = 1000
        metrics = {
            'MAE': [],
            'Variance': [],
            'R2': [],
            'Pearson': []
        }

        # 高效Bootstrap计算
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(true_values), size=len(true_values), replace=True)
            sample_true = true_values[idx]
            sample_pred = predicted_values[idx]

            absolute_errors = np.abs(sample_pred - sample_true)
            metrics['MAE'].append(np.mean(absolute_errors))
            metrics['Variance'].append(np.var(absolute_errors))
            metrics['R2'].append(r2_score(sample_true, sample_pred))
            metrics['Pearson'].append(pearsonr(sample_true, sample_pred)[0])

        # 计算点估计值
        mae_point = np.mean(np.abs(predicted_values - true_values))
        var_point = np.var(np.abs(predicted_values - true_values))
        r2_point = r2_score(true_values, predicted_values)
        pearson_point, p_value = pearsonr(true_values, predicted_values)

        # 计算置信区间（95%）
        ci = {}
        for metric in metrics:
            sorted_values = np.sort(metrics[metric])
            ci_low = sorted_values[int(0.025 * n_bootstrap)]
            ci_high = sorted_values[int(0.975 * n_bootstrap)]
            ci[metric] = (ci_low, ci_high)

        # 写入文件，指定UTF-8编码
        with open(record, "a", encoding='utf-8') as f_test:
            f_test.write(mm + "\n")
            # MAE
            f_test.write(f"Mean Absolute Error: {mae_point:.4f}\n")
            f_test.write(f"95% CI: [{ci['MAE'][0]:.4f}, {ci['MAE'][1]:.4f}]\n")

            # Variance
            f_test.write(f"Error Variance: {var_point:.4f}\n")
            f_test.write(f"95% CI: [{ci['Variance'][0]:.4f}, {ci['Variance'][1]:.4f}]\n")

            # R²
            f_test.write(f"R² Score: {r2_point:.4f}\n")
            f_test.write(f"95% CI: [{ci['R2'][0]:.4f}, {ci['R2'][1]:.4f}]\n")

            # Pearson
            f_test.write(f"Pearson Correlation: {pearson_point:.4f}\n")
            f_test.write(f"95% CI: [{ci['Pearson'][0]:.4f}, {ci['Pearson'][1]:.4f}]\n")
            f_test.write(f"Associated p-value: {p_value:.4e}\n\n")

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

def total_test():
    file_path="./test/m6_7.txt"#读取划分好的文件
    # total=read_total(file_path)
    # metric(total)
    # dot_plot(total)
    # histogrism(total)

    #subjects
    # subject=read_subject(file_path)

    # conf=conform1(file_path)
    # metric(conf)
    # dot_plot(conf)
    # histogrism(conf)

    # conf=conform3(file_path)
    # metric(conf)
    # dot_plot(conf)
    # histogrism(conf)

    # conf=conform5(file_path)
    # metric(conf)
    # dot_plot(conf)
    # histogrism(conf)

    # conf=conform7(file_path)
    # metric(conf)
    # dot_plot(conf)
    # histogrism(conf)

    # conf=conform_last(file_path)
    # metric(conf)
    # dot_plot(conf)
    # histogrism(conf)

def nor_weight(input_path,output_path):
    with open(input_path, "r") as tf:
        lines = tf.read().split("\n")
    f_weight = open(output_path, "w")
    # 清洗数据并转换为浮点数列表
    matrix_list = []
    matrix_data = []
    for line in lines:
        # 移除开头和结尾的方括号
        line_cleaned = line.strip()[2:-2].strip()
        numbers = line_cleaned.split()
        if len(numbers) == 0:
            break
        abc=np.array([float(num) for num in numbers])
        abc=abc-abc.min()+0.01
        weight = abc/abc.sum()
        # weight = abc / abc.max()
        f_weight.write(str(weight) + '\n')
    print("finished!")

def fix_scientific_notation(num):
    """
    修复不完整的科学计数法字符串。
    假设缺失的指数部分为 '0'。
    """
    if 'e+' in num and not num.endswith('e+'):  # 修复形如 '-1.62159860e+' 的情况
        return num + '0'
    elif 'e-' in num and not num.endswith('e-'):  # 修复形如 '-1.62159860e-' 的情况
        return num + '0'
    elif 'e' in num and len(num.split('e')) != 2:  # 处理其他不完整情况
        return num + '0'
    return num

def safe_float_conversion(numbers):
    """
    安全地将字符串列表转换为浮点数列表。
    跳过无法修复的无效值。
    """
    valid_numbers = []
    for num in numbers:
        try:
            # 修复科学计数法后尝试转换为浮点数
            fixed_num = fix_scientific_notation(num)
            valid_numbers.append(float(fixed_num))
        except ValueError:
            print(f"Warning: Could not convert '{num}' to float. Skipping this value.")
    return valid_numbers

def globle_nor_weight():
    models=["GradientShap_weight3","GradientShap_weight4","GradientShap_weight5",
            "GradientShap_weight6","GradientShap_weight7","GradientShap_weight8"]
    for model in models:
        if model !="GradientShap_weight8":
            continue
        script_dir = os.path.dirname(os.path.abspath(__file__))
        inpath = os.path.join(script_dir, 'GradientShape',model+".txt")
        # inpath=os.path.join("./GradientShape",model+".txt")#"./GradientShap/GradientShap_weight1.txt"
        outpath = inpath.replace("weight", "nor")
        max=0
        min=0
        for i in range(2,8):
            # input_path=inpath.replace("1",str(i))
            with open(inpath, "r") as tf:
                lines = tf.read().split("\n")
            # for line in lines:
                # 移除开头和结尾的方括号
            line=lines[-2]
            line_cleaned = line.strip()[2:-2].strip()
            numbers = line_cleaned.split()
            if len(numbers) == 0:
                break
            valid_numbers = safe_float_conversion(numbers)
            abc = np.array(valid_numbers)
            # abc=np.array([float(num) for num in numbers])
            # abc=abc-abc.min()+0.01
            if max<abc.max():
                max=abc.max()
            if min>abc.min():
                min=abc.min()
        for i in range(2, 8):
            output_path = outpath.replace("1", str(i))
            input_path = inpath.replace("1", str(i))
            with open(input_path, "r") as tf:
                lines = tf.read().split("\n")
            f_weight = open(output_path, "w")
            for line in lines:
                # 移除开头和结尾的方括号
                line_cleaned = line.strip()[2:-2].strip()
                numbers = line_cleaned.split()
                if len(numbers) == 0:
                    break
                valid_numbers = safe_float_conversion(numbers)
                abc = np.array(valid_numbers)
                # abc = np.array([float(num) for num in numbers])
                abc = abc - min + 0.01
                weight = abc / (max-min)
                weight[weight>1]=1
                # weight = abc / abc.max()
                f_weight.write(str(weight) + '\n')
            print("finished!")

def globle_caculate():
    inpath="./GradientShap/GradientShap_nor2.txt"
    outpath = inpath.replace("weight", "nor")
    total=0
    for i in range(2,7):
        input_path=inpath.replace("2",str(i))
        with open(input_path, "r") as tf:
            lines = tf.read().split("\n")
        total+=len(lines)-1
    print(total)
    print("finished!")

def caculate_index_internal():
    cross = 5
    models = ["model1_dur", "model2_dur_energy", "model3_dur_energy_sdr", "model4_dur_energy_size_sdr",
              "model5_dur_energy_area_size_sdr","model6_ang_dur_energy_area_size_sdr",
              "model6_ang_dur_energy_area_size_sdr2","model7_dis_dur_energy_area_size_sdr_avp_seq",
              "model8_ang_dis_dur_energy_area_size_sdr_avp_seq"]
    for j in range(len(models)):
        if j < 1:
            continue
        mm = models[j]
        aa=[]
        for i in range(cross):
            fold = "./fold_" + str(i + 1)
            record = fold + '/internal_results/' + "test"+str(i + 1)+".txt"
            # if os.path.exists(record):
            #     os.remove(record)
            # else:
            #     pass  # 文件不存在，跳过删除操作
            # f_test = open(record, "a")
            in_results = fold + '/internal_results/'+mm+ ".txt"
            conf = read_total(in_results)
            metric(conf,record,mm)
            aa+=[conf]
        x_all = np.vstack(aa)
        dot_plot(x_all)
        histogrism(x_all)
        # a = 1

def caculate_index_external():
    cross = 5
    models = ["model1_dur", "model2_dur_energy", "model3_dur_energy_sdr", "model4_dur_energy_size_sdr",
              "model5_dur_energy_area_size_sdr","model6_ang_dur_energy_area_size_sdr",
              "model6_ang_dur_energy_area_size_sdr2","model7_dis_dur_energy_area_size_sdr_avp_seq",
              "model8_ang_dis_dur_energy_area_size_sdr_avp_seq"]
    for j in range(len(models)):
        if j < 1:
            continue
        mm = models[j]
        aa = []
        for i in range(cross):
            if i !=3:
                continue
            fold = "./fold_" + str(i + 1)
            record = fold + '/external_results/' + "test"+str(i + 1)+".txt"
            # if os.path.exists(record):
            #     os.remove(record)
            # else:
            #     pass  # 文件不存在，跳过删除操作
            in_results = fold + '/external_results/'+mm+ ".txt"
            conf = read_total(in_results)
            # metric(conf,record,mm)
            dot_plot(conf)
            histogrism(conf)

            aa += [conf]
        # x_all = np.vstack(aa)
        # dot_plot(x_all)
        # histogrism(x_all)
        a = 1

if __name__ == '__main__':
    # total_test() #各个次数的刺激,暂且用不上
    # caculate_index_internal()
    # caculate_index_external()
    globle_nor_weight()
    a=1