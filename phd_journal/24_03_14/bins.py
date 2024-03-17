from math import floor, ceil


def is_good_developmental_age_estimate(estimate: float, mmse: int) -> bool:
    """
    Outputs a MMSE approximation given the developmental age estimated by an EEG model.
    """
    assert 0 <= mmse <= 30, "MMSE must be between 0 and 30"
    assert 0 <= estimate, "Developmental age estimate must be positive"

    if estimate < 1.25:
        return 0 <= mmse <= estimate/2
    elif estimate < 2:
        return floor((4 * estimate / 15) - (1 / 3)) <= mmse <= ceil(estimate/2)
    elif estimate < 5:
        return (4*estimate/15) - (1/3) <= mmse <= 2*estimate + 5
    elif estimate < 7:
        return 2*estimate - 6 <= mmse <= (4*estimate/3) + (25/3)
    elif estimate < 8:
        return (4*estimate/5) + (47/5) <= mmse <= (4*estimate/3) + (25/3)
    elif estimate < 12:
        return (4 * estimate / 5) + (47 / 5) <= mmse <= (4 * estimate / 5) + (68 / 5)
    elif estimate < 13:
        return (4 * estimate / 7) + (92 / 7) <= mmse <= (4 * estimate / 5) + (68 / 5)
    elif estimate < 19:
        return (4 * estimate / 7) + (92 / 7) <= mmse <= 30
    elif estimate >= 19:
        return mmse >= 29


accurate = []
inaccurate = []
for estimate_ in range(0, 250):
    estimate = estimate_ / 10
    for mmse in range(0, 31):
        if is_good_developmental_age_estimate(estimate, mmse):
            accurate.append((estimate, mmse))
        else:
            inaccurate.append((estimate, mmse))

# Plot points
# accurate green, inaccurate red
# x-axis: developmental age estimate
# y-axis: MMSE
import matplotlib.pyplot as plt
accurate_x, accurate_y = zip(*accurate)
plt.xlabel('Developmental Age Estimate (years)')
plt.ylabel('Acceptable MMSE (unit)')
plt.xticks((0, 1, 2, 5, 7, 8, 12, 13, 19, 25))
plt.xlim(0, 25.1)
plt.ylim(-0.5, 30.5)
plt.grid(linestyle='--', alpha=0.4)
plt.scatter(accurate_x, accurate_y, color='g', marker='|', alpha=0.3)  # square markers
# remove box around plot
plt.box(False)
plt.show()
