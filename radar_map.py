import numpy as np
import matplotlib.colors as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MaxNLocator

# data = [{"Duration": 0.48454759, "Energy": 0.68870451, "SDR": 0.61428491},
#            {"Duration": 0.46066348, "Energy": 0.79886045, "SDR": 0.66861613,"Size": 0.35830502},
#            {"Duration": 0.47308118, "Energy": 0.99191841, "SDR": 0.64703548, "Size": 0.24549836, "Area": 0.61039081},
#         {"Duration": 4.58037973e-01, "Energy": 1.00000000e+00, "SDR": 7.69921216e-01, "Size": 1.45238378e-04,
#          "Area": 2.44012515e-01, "Course": 7.88227514e-01},
#            {"Duration": 0.59923701, "Energy": 0.61286139, "SDR": 0.62427609, "Size": 0.63447653, "Area": 0.57836626,"Course": 0.60508618}]

data = [{"Duration": 0.00111526, "Energy": 1, "SDR": 0.04749312},
           {"Duration": 1.41774393e-01, "Energy": 9.67507513e-01, "SDR": 1.0,"Size": 5.49533493e-04},
           {"Duration": 1.63060519e-01, "Energy": 9.98403318e-01, "SDR": 1.0, "Size": 5.27654254e-04, "Area": 6.04504405e-01},
        {"Duration": 3.47607140e-01, "Energy": 9.49399150e-01, "SDR": 1.0, "Size": 7.93816027e-04,
         "Area": 4.23887058e-01, "Course": 3.19613718e-01},
           {"Duration": -1.23077416e-06, "Energy": 1.83245608e-06, "SDR": 3.00919525e-06, "Size": -1.22989646e-06, "Area": -5.99886235e-07,"Course": 1.0}]


# 获取所有键（即雷达图的各个轴标签）
all_keys = ['Duration', 'Energy', 'SDR', 'Size', 'Area', 'Course']

# 初始化 figure 和 polar subplot
fig = plt.figure(figsize=(6, 6), dpi=100)
ax = fig.add_subplot(111, projection='polar')


# 计算角度和标签
angles = np.linspace(0, 2 * np.pi, len(all_keys), endpoint=False)
labels = list(all_keys)

# 计算所有数据集中存在的最小值和最大值
min_value = np.nanmin([np.array([record.get(key, np.nan) for key in all_keys]) for record in data])
max_value = np.nanmax([np.array([record.get(key, np.nan) for key in all_keys]) for record in data])
# colors = ['b','lightpink', 'lightsalmon','lightskyblue', 'lightgreen']  #
colors = ['orange','midnightblue', 'lightpink','dodgerblue', 'aquamarine']  #dodgerblue
# colors = ['orange','lightpink', 'midnightblue','aquamarine', 'dodgerblue']  #dodgerblue
# colors = ['navy','lightpink', 'forestgreen','lightskyblue', 'slategray']  #
# colors = ['darkslategrey', 'mediumturquoise', 'slategray', 'royalblue','steelblue']  #
# colors = ['b', 'm','g', 'c']  #
# colors = ['Purple','b', 'm','g', 'c']  #
# colors = ['Purple', 'r','b', 'c']  #

# forestgreen darkgreen green g seagreen mediumseagreen darkslategrey
# mediumaquamarine turquoise lightseagreen mediumturquoise
# darkslategrey teal darkcyan c darkturquoise cadetblue steelblue slategray
# lightsteelblue midnightblue navy darkblue indigo royalblue

# 创建一个空的Line2D对象列表用于图例
legend_lines = []
legend_labels = []
# 创建一个空的Patch列表用于填充图形并添加到图例
patches = []
for group_index, record in enumerate(data):
    values = np.array([record.get(key, 0) for key in all_keys])

    # 绘制填充的扇形区域，设置填充颜色的透明度
    patch = ax.fill(angles, values, color=colors[group_index], alpha=0.25)  # 调整填充颜色的透明度使之更浅
    # 添加小圆点以标记数据点的位置
    for angle, value in zip(angles, values):
        ax.scatter(angle, value, color=colors[group_index], s=30, alpha=0.45, zorder=10)  # 调整s参数控制圆点半径，zorder提高圆点的层叠顺序
    patches.append(patch[0])# 绘制线条，设置线条颜色的透明度
    legend_labels.append(f"Set{group_index + 2}")
    line, = ax.plot(np.append(angles, angles[0]), np.append(values, values[0]),
                    '-', linewidth=1, color=colors[group_index], alpha=0.25)  # 调整线条颜色的透明度使之更浅

ax.legend(patches, legend_labels, loc='upper right', bbox_to_anchor=(1.05, 1.08), ncol=1, fancybox=True, shadow=True, framealpha=0.8)
ax.set_thetagrids(angles * 180 / np.pi, labels)
ax.set_theta_zero_location('N')
ax.set_rlim(min(0, min_value), max(max_value, 1) * 1.1)  # 更新这里的rlim设置
ax.set_rlabel_position(0)

# 使用patches创建图例
ax.legend(patches, legend_labels, loc='upper right', bbox_to_anchor=(1.05, 1.08), ncol=1, fancybox=True, shadow=True, framealpha=0.8)

ax.spines['polar'].set_visible(False)
# ax.grid(False)  # 可选，根据需求决定是否显示网格线

# plt.title("Weight Map")
plt.show()