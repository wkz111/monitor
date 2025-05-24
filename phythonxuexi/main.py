import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import stft
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ======================== 数据生成 ========================
# 模拟声发射信号（采样率2 MHz）
fs = 2e6
t = np.arange(0, 0.02, 1 / fs)

# 正常工况信号（20 kHz正弦波叠加噪声）
signal_normal = 0.5 * np.sin(2 * np.pi * 20e3 * t) + 0.1 * np.random.randn(len(t))

# 磨损工况信号（随机冲击脉冲）
impulse_times = [5000, 12000, 18000]  # 冲击时间点
signal_wear = 0.2 * np.random.randn(len(t))
for idx in impulse_times:
    signal_wear[idx:idx + 200] += 2.0 * np.exp(-np.linspace(0, 5, 200))

# ======================== PDF生成 ========================
with PdfPages('seal_ring_analysis.pdf') as pdf:
    # ----------------- 时域波形对比图 -----------------
    plt.figure(figsize=(12, 6))
    plt.plot(t[:6000] * 1000, signal_normal[:6000], '#2F5496', label='Normal Operation')
    plt.plot(t[:6000] * 1000, signal_wear[:6000], '#ED7D31', linestyle='--', label='Severe Wear')
    plt.xlabel('Time (ms)', fontsize=12)
    plt.ylabel('Amplitude (V)', fontsize=12)
    plt.title('Acoustic Emission Signal Comparison', fontweight='bold', pad=20)
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, linestyle=':')
    plt.tight_layout(pad=2.0)  # 增强布局紧凑性
    pdf.savefig(bbox_inches='tight')  # 自动裁剪空白区域
    plt.close()

    # ----------------- STFT时频分析图 -----------------
    f, t_stft, Zxx = stft(signal_wear, fs=fs, nperseg=1024, noverlap=768)
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(t_stft, f / 1e3, 20 * np.log10(np.abs(Zxx)),
                   shading='gouraud', cmap='jet', vmin=-40, vmax=40)
    plt.colorbar(label='Intensity (dB)', extend='both')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Frequency (kHz)', fontsize=12)
    plt.ylim(20, 400)  # 聚焦有效频段
    plt.title('Short-Time Fourier Transform Analysis', fontweight='bold', pad=20)
    plt.tight_layout(pad=2.0)
    pdf.savefig(dpi=300, metadata={'Title': 'STFT Analysis'})  # 设置分页元数据
    plt.close()

    # ----------------- 混淆矩阵热图 -----------------
    # 模拟分类结果
    labels = ['Normal', 'Mild', 'Severe']
    y_true = np.random.choice([0, 1, 2], 300, p=[0.4, 0.3, 0.3])
    y_pred = np.random.choice([0, 1, 2], 300, p=[0.4, 0.3, 0.3])

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues',
                cbar_kws={'label': 'Classification Rate'},
                annot_kws={"size": 12, "color": "#404040"})
    plt.xticks(np.arange(3) + 0.5, labels, rotation=45, ha='right')
    plt.yticks(np.arange(3) + 0.5, labels, rotation=0)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.title('Confusion Matrix (Normalized)', fontweight='bold', pad=20)
    plt.tight_layout(pad=2.0)
    pdf.savefig(bbox_extra_artists=[plt.gca()], pad_inches=0.1)  # 精确控制边距
    plt.close()

    # ================= 设置PDF文档元数据 =================
    meta = pdf.infodict()
    meta['Title'] = 'Seal Ring Wear Analysis Report'
    meta['Author'] = 'Industrial Monitoring Laboratory'
    meta['Subject'] = 'Acoustic Emission and Temperature Fusion Monitoring'
    meta['Keywords'] = 'AE FBG C4.5 DecisionTree'
    meta['CreationDate'] = np.datetime64('2025-05-23')