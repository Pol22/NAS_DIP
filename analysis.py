import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np


file_path = 'train_median.log'
# METHOD = 'MEAN'
METHOD = 'MEDIAN'

top_generation = []
top_psnr = []
train_generation = []
train_psnr = []
values = []
top_flag = False
train_flag = False

with open(file_path, 'r') as f:
    for line in f.readlines():
        if top_flag:
            if line.startswith('['):
                top_psnr[-1].append(float(line.split()[0][1:-1]))
            else:
                top_flag = False

        if train_flag:
            if line.startswith('['):
                train_psnr[-1].append(float(line.split()[0][1:]))
            else:
                train_flag = False

        if 'Top of generation' in line:
            top_generation.append(int(line.split()[-1]))
            top_psnr.append([])
            top_flag = True

        if 'Train generation' in line:
            train_generation.append(int(line.split()[-1]))
            train_psnr.append([])
            train_flag = True

        if METHOD in line:
            value = line.split()[-1]
            values.append(float(line.split()[-1]))

sns.boxplot(data=top_psnr, color='cyan', linewidth=0.5)
plt.xticks([], [])
plt.xlabel('Generation')
plt.ylabel('PSNR')
plt.title('Top distribution per generation')
plt.savefig(f'top_{METHOD.lower()}.png')

plt.clf()
sns.boxplot(data=train_psnr, color='cyan', linewidth=0.5)
plt.xticks([], [])
plt.xlabel('Generation')
plt.ylabel('PSNR')
plt.title('Train distribution per generation')
plt.savefig(f'train_{METHOD.lower()}.png')

plt.clf()
plt.plot(top_generation, values)
plt.xlabel('Generation')
plt.ylabel('PSNR')
plt.title(f'{METHOD} per generation')
plt.savefig(f'{METHOD.lower()}.png')
