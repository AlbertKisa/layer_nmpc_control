import numpy as np
import matplotlib.pyplot as plt

Qc = 5.0
kappa = 0.1
r_i = 2

d = np.arange(0, 101)
cost = Qc / (1 + np.exp(kappa * (d - 2 * r_i)))

plt.figure(figsize=(8, 6))
plt.plot(d,
         cost,
         label=r'$\frac{Qc}{1 + e^{\kappa \cdot (d(k+s|k,i) - 2r_i)}}$',
         color='b')
plt.xlabel('d(k+s|k,i)', fontsize=12)
plt.ylabel('d', fontsize=12)
plt.title('Plot of d vs. d(k+s|k,i)', fontsize=14)
plt.grid(True)
plt.legend()
plt.show()
