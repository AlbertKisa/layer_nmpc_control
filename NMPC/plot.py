import matplotlib.pyplot as plt
import numpy as np
from utils import GenerateHexagonVertices

if __name__ == "__main__":
    x, y = GenerateHexagonVertices(0.25)
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, marker="o")
    plt.grid(True)
    plt.axis('equal')
    plt.show()
