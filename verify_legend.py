import matplotlib.pyplot as plt
import numpy as np

def mock_plot_bands(axis=None, label=None, color='black'):
    if axis is None:
        fig, ax = plt.subplots()
    else:
        ax = axis

    # Simulate plotting multiple bands
    bands = np.random.rand(5, 10) # 10 bands, 5 k-points
    
    # Logic extracted from the fix:
    real_label = label
    for i in range(10): # 10 bands
        ax.plot(bands[:, i], color=color, label=real_label if i == 0 else None)

    return ax

fig, ax = plt.subplots()
mock_plot_bands(axis=ax, label='Material 1', color='red')
mock_plot_bands(axis=ax, label='Material 2', color='blue')
ax.legend()

# Check handles and labels
handles, labels = ax.get_legend_handles_labels()
print(f"Legend labels: {labels}")
if labels == ['Material 1', 'Material 2']:
    print("Verification successful!")
else:
    print(f"Verification FAILED: {labels}")
    exit(1)
plt.close(fig)
