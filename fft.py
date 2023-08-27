import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
import seaborn
import pandas as pd

data_key = 'KDD_Cup_SDWPF'
data_value = 'KDD_Cup_SDWPF'
datadict = {data_key: data_value}
#采样点选择1400个，因为设置的信号频率分量最高为600赫兹，根据采样定理知采样频率要大于信号频率2倍，所以这里设置采样频率为1400赫兹（即一秒内有1400个采样点，一样意思的）
df = pd.read_csv('../../data/wpf_data_ID1/wpf_data_ID1.csv', usecols=['OT'])

# x_data = np.array(df)[0:50]
x_data = np.array(df)[0:2000]
data = np.nan_to_num(x_data, nan=np.nanmean(x_data))
y = np.squeeze(data)
x = np.arange(0, len(y))


yf = abs(fft(y))                # 取绝对值
# 处理直流分量
yf[0] = yf[0]/len(y)
# 归一化处理,画图不画直流分量
# yf[1:] = yf[1:]/len(x) * 2
yf = yf/len(x) * 2

xf = np.arange(len(y))        # 频率

# plt.subplot(111)
# plt.plot(x, y, linewidth=0.5)
# plt.title('{}'.format(item), fontsize=7)
# plt.tick_params(axis='both', which='major', labelsize=7)
# plt.xlabel('values', fontdict=[])
# 设置画布大小
plt.figure(dpi=170, figsize=(6, 6))
plt.subplots_adjust(wspace=0.15, hspace=0.5)
plt.subplot(2, 1, 1)
# 设置坐标科学计数显示
plt.ticklabel_format(style='sci', scilimits=(-2, 2), axis='both')
title = '{} - Series'.format(datadict[data_key])
plt.title(title, fontsize=18, color='darkviolet')
plt.tick_params(axis='both', which='major', labelsize=18)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Calibri'
plt.plot(x, y, 'r', linewidth=1)

plt.subplot(2, 1, 2)
plt.ticklabel_format(style='sci', scilimits=(-2, 2), axis='both')
title = '{} - FFT(two sides)'.format(datadict[data_key])
plt.title(title, fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Calibri'
plt.plot(xf, yf, 'b', linewidth=1)


# plt.legend()
path_pic = './FFTpic/{}.pdf'.format(data_key)
plt.savefig(path_pic, bbox_inches='tight')
plt.show()