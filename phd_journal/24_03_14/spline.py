import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

points = (
    (0, 0),
    (0, 1.25),
    (1, 2),
    (1, 3),
    (3, 3),
    (3, 4),
    (5, 4),
    (8, 4),
    (9, 5),
    (15, 5),
    (15, 7),
    (19, 8),
    (20, 12),
    (24, 12),
    (24, 13),
    (27, 19),
    (29, 19),
    (30, 19),
    (30, 24),
)

x, y = zip(*points)

tck, um = interpolate.splprep([x, y], s=0, k=3)

xnew, ynew = interpolate.splev(
    np.linspace(0, 1, 100), tck
)

plt.plot(x, y, 'x', xnew, ynew, x, y, 'b')
plt.show()
