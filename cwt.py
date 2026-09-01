import numpy as np
import pywt
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

def CWT(data, wavelet='morl', total_scales=500, sample_interval=1e-3):
    fc = pywt.central_frequency(wavelet)  # 小波基函数的中心频率
    c_param = 2 * fc * total_scales
    scales = c_param / np.arange(total_scales, 1, -1)
    cwt_coefficients, frequencies = pywt.cwt(data, scales, wavelet, sample_interval)  # 计算连续小波变换
    cwt_coefficients = np.real(cwt_coefficients[419:, :])
    # return cwt_coefficients[419:, :], frequencies[419:]
    return cwt_coefficients, frequencies[419:]
    # return cwt_coefficients


def PlotCwt(time_data, time_fre_data, frequencies):
    """
    绘制小波变换结果
    :param time_data: 原始数据
    :param time_fre_data: 小波变换结果
    :param frequencies: 小波变换频率
    """
    # 绘制原始信号
    plt.figure(figsize=(12, 6))  # 设置图像大小
    plt.subplot(1, 2, 1)
    plt.plot(time_data, np.arange(len(time_data)), color='black', linewidth=2)
    plt.ylim(0, len(time_data))
    plt.gca().invert_yaxis()
    plt.xlabel('Amplitude')
    plt.ylabel('Time (ms)')
    plt.title('Original Signal')

    # 绘制CWT时频分析图
    plt.subplot(1, 2, 2)
    plt.pcolormesh(frequencies, np.arange(0, time_fre_data.shape[1]),
                   np.rot90(np.real(time_fre_data)), shading='auto', cmap='RdGy') # RdBu
    plt.xlim([frequencies[-1], frequencies[0]])
    plt.xlabel('Frequency (Hz)')
    plt.title('CWT Spectrogram')
    plt.colorbar(label='Amplitude')
    plt.gca().set_yticks([])  # 移除 y 轴刻度
    # plt.tight_layout()  # 自动调整子图参数
    plt.show()


def toMinMaxSeis(d, h=0.4, center=0.5):
    dmax = np.max(np.abs(d))
    ck = h / (dmax + 1e-16)
    cb = center
    re = ck * d + cb
    return re, ck, cb  # re=ck*d+cb


syn = np.load('syn.npy')
[nTime, nTrace] = syn.shape
syn = toMinMaxSeis(syn)[0]-0.5
syn_cwt = np.zeros((80, nTime, nTrace))
for i in range(nTrace):
    syn_cwt[:, :, i], freq = CWT(syn[:, i])
    PlotCwt(syn[:, i], syn_cwt[:, :, i], freq)
    if i % 100 == 0:
        print(i)
# np.save('syn_cwt.npy', np.float32(syn_cwt))
print(1)

