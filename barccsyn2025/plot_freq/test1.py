import matplotlib.pyplot as plt


x = [1, 2, 3, 4, 5, 6, 7, 8]
model_perf = [0.1360, 0.1957, 0.2256, 0.2300, 0.2311, 0.2328, 0.2326, 0.2104]
rec_corr = [0.071, 0.139, 0.202, 0.205, 0.213, 0.235, 0.198, 0.194]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, model_perf, label="Encoder performance")
ax.plot(x, rec_corr, linestyle="--", label="Decoder video correlation")
ax.scatter(x, model_perf)
ax.scatter(x, rec_corr, marker='x')

ax.set_xlabel("Frequency (Hz)")
ax.set_title("Neural recordings frequency influence on model performance")
ax.legend()

plt.tight_layout()
plt.grid(linestyle = '--', linewidth = 0.5)
plt.savefig('barccsyn2025/plot_freq/freq.png')