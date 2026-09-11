# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import numpy as np
from collections import Counter


def readlines(filename):
    """Read all lines from a text file and return as a list."""
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


class KITTIObject(object):
    """Load KITTI object detection labels."""
    def __init__(self, root_dir, split='training'):
        self.root_dir = root_dir
        self.split = split


def load_velodyne_points(filename):
    """Load 3D point cloud from KITTI velodyne file."""
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points


def sub2ind(matrixSize, rowSub, colSub):
    """Convert row, col matrix subscripts to linear indices."""
    m, n = matrixSize
    return rowSub * n + colSub


def generate_depth_map(calib_dir, velo_filename, cam=2, vel_depth=False):
    """Generate a depth map from velodyne points (KITTI specific).
    
    Not used for endoscopic data — returns None.
    """
    return None
