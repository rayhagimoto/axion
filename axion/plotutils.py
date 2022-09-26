import numpy as np

def normalize_lines(ax):
    ydata = np.array([line.get_ydata()/np.max(line.get_ydata()) for line in ax.lines], dtype=object)
    for j, line in enumerate(ax.lines):
        line.set_ydata(ydata[j])

def normalize_line(ax, i):
    line = ax.get_lines()[i]
    ydata = line.get_ydata()/np.max(line.get_ydata())
    line.set_ydata(ydata)

def get_line(ax, idx):
    """Returns the line instance in an ax at a given index.
    """

    lines = ax.get_lines()
    line = lines[idx]

    return line


def get_line_xy_data(ax, idx):
    """Get x and y data from a line given the ax on which
    it was plotted and its index.
    """

    line = get_line(ax, idx)

    y = np.array(line.get_ydata(), dtype=float)
    x = np.array(line.get_xdata(), dtype=float)

    return x, y


def smooth_line(ax, idx, w, sigma):
    """Smooth a line instance with a moving average, then a Gaussian
    filter"""

    line = get_line(ax, idx)
    x, y = get_line_xy_data(ax, idx)

    def moving_average(y, w):
        return np.convolve(y, np.ones(w), 'valid') / w

    y_ma = y
    if w > 0:
        y_ma = moving_average(y, w)
        _x = moving_average(x, w)
        line.set_xdata(_x)
    
    from scipy.ndimage import gaussian_filter
    y_smoothed = gaussian_filter(y_ma, sigma)
    line.set_ydata(y_smoothed)