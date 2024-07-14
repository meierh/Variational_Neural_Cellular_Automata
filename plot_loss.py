import pandas as pd
import matplotlib.pyplot as plt
import os

# 读取数据文件
file_path = os.path.join(os.path.dirname(__file__), 'results', 'results.csv')
df = pd.read_csv(file_path)

# 设置图形大小
plt.figure(figsize=(14, 8))

# 绘制柱状图
bars = plt.bar(df['model'], df['loss'], color='cornflowerblue', edgecolor='black')

# 为每个柱子顶部标注数值
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 设置标签和标题
plt.xticks(rotation=0, ha='center', fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel('Model Variations', fontsize=12, fontweight='bold')
plt.ylabel('Loss', fontsize=12, fontweight='bold')
plt.title('Loss for Different Model Variations', fontsize=14, fontweight='bold')

# 添加网格线
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 显示图形
plt.tight_layout()
plt.show()