import numpy as np
import matplotlib.pyplot as plt

def stereographic_from_north(u, v):
    denom = u**2 + v**2 + 1
    x = 2*u / denom
    y = 2*v / denom
    z = (u**2 + v**2 - 1) / denom
    return x, y, z

def stereographic_from_south(u, v):
    denom = u**2 + v**2 + 1
    x = 2*u / denom
    y = 2*v / denom
    z = (-u**2 - v**2 + 1) / denom
    return x, y, z

# Create grid in parameter space
U, V = np.meshgrid(np.linspace(-2, 2, 50), np.linspace(-2, 2, 50))

# Get chart points
Xn, Yn, Zn = stereographic_from_north(U, V)
Xs, Ys, Zs = stereographic_from_south(U, V)

# Plot
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(Xn, Yn, Zn, color='lightblue', alpha=0.8)
ax.set_title("Chart from North Pole")

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(Xs, Ys, Zs, color='lightgreen', alpha=0.8)
ax2.set_title("Chart from South Pole")

plt.show()
