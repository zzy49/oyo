import yaml
import os
import numpy as np

def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'v6_config.yaml')
    
    if not os.path.exists(config_path):
        return get_default_config()
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    camera = config['camera']
    config['camera']['K'] = np.array([
        [camera['fx'], 0, camera['cx']],
        [0, camera['fy'], camera['cy']],
        [0, 0, 1]
    ], dtype=np.float64)
    
    return config

def get_default_config():
    return {
        'camera': {
            'fx': 500.0,
            'fy': 500.0,
            'cx': 640.0,
            'cy': 360.0,
            'K': np.array([[500.0, 0, 640.0], [0, 500.0, 360.0], [0, 0, 1]], dtype=np.float64)
        },
        'data': {
            'image_dir': r'E:\data1\monodepth2\cs\2',
            'model_path': r'C:\Users\Administrator\tmp\endo_v3_stage2_v3\models\weights_19',
            'output_dir': r'E:\data1\monodepth2\v6_cs2_test'
        },
        'pipeline': {
            'n_frames': 5,
            'min_static_ratio': 0.05,
            'motion_threshold': 2.0,
            'reprojection_error': 8.0
        },
        'loop_closure': {
            'keyframe_interval': 10,
            'min_frame_gap': 30,
            'min_matches': 25
        },
        'bundle_adjustment': {
            'weight_sequential': 1.0,
            'weight_loop': 2.0,
            'weight_smooth': 0.3
        }
    }