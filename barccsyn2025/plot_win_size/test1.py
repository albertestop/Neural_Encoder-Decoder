import matplotlib.pyplot as plt


x = [33, 66, 99]
model_perf = [0.7943, 0.8031, 0.8259]
rec_corr = [0.275, 0.45, 0.255]
rec_corr_mask = [0.035, 0.26, 0.255]


fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, model_perf, label="Encoder performance")
ax.plot(x, rec_corr, linestyle="--", label="Decoder masked video correlation")
ax.plot(x, rec_corr_mask, linestyle="dashdot", label="Decoder video correlation")
ax.scatter(x, model_perf)
ax.scatter(x, rec_corr, marker='x')
ax.scatter(x, rec_corr_mask, marker='o')


ax.set_xlabel("% window size")
ax.set_title("Window size influence on model performance")
ax.legend()

plt.tight_layout()
plt.grid(linestyle = '--', linewidth = 0.5)
plt.savefig('barccsyn2025/plot_win_size/freq.png')