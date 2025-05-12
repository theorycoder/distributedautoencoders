import numpy as np
import matplotlib.pyplot as plt

# Plot settings
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern",
    "font.size": 16
})

# Load data from txt file
data = np.loadtxt("accuracy_vs_const.txt", skiprows=1)
constants = data[:, 0]
accuracies = data[:, 1]

# Plot
fig = plt.figure(figsize=(8,6))
plt.plot(constants, accuracies, '-o', linewidth=3, markersize=10)
plt.grid(True, which="both")
plt.xlabel('$c_{j,i}$', fontsize=20)
plt.ylabel(r'Average accuracy at $\epsilon=10$', fontsize=20)
#plt.title('Effect of Stabilization Constant on DP-DA Accuracy')
plt.tight_layout()
plt.savefig('accuracy_vs_constant.pdf')

