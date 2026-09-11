"""Utility functions for monodepth2 training."""


def readlines(filename):
    """Read all non-empty lines from a file."""
    with open(filename, 'r') as f:
        return [line.rstrip('\n') for line in f.readlines()
                if len(line.rstrip('\n')) > 0]


def sec_to_hm_str(t):
    """Convert time in seconds to a string hh:mm:ss."""
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def sec_to_hm(t):
    """Convert time in seconds to (h, m) tuple."""
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    return h, m


def normalize_image(x):
    """Normalize image tensor for visualization, mapping to [0,1]."""
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min + 1e-7)
