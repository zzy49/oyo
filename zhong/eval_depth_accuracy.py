#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""4 模型深度精度评估 —— AbsRel, RMSE, δ<1.25, Ratio.

关键: 直接复用验证过的 v6_pipeline/utils.py 中的 predict_depth()。
"""

import os, sys, json
import numpy as np
from PIL import Image
import torch
import importlib

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from v6_pipeline.utils import ModelManager, predict_depth

_LITEMONO_DIR = r'E:\data1\Lite-Mono-main'
_MANYDEPTH_DIR = r'E:\data1\ManyDepth-main'

DEVICE = torch.device('cuda')

WARPED_RGB_DIR = r'F:\dataset\c1_transverse1_t1_v2\generated\rgb_warped'
GT_DEPTH_DIR   = r'F:\dataset\c1_transverse1_t1_v2\depth'
OUT_DIR = os.path.join(PROJECT_DIR, 'zhong', 'vo_depth_compare')
os.makedirs(OUT_DIR, exist_ok=True)

GT_SCALE = 100.0 / 65535.0  # uint16 → mm


# ═══════════════════════════════════════════════
# 模型加载 (复用已验证方式)
# ═══════════════════════════════════════════════

def load_manydepth(model_path, device):
    """ManyDepth 单帧推理: 用 mono_encoder.pth / mono_depth.pth (标准 ResNet)."""
    sys.path.insert(0, _MANYDEPTH_DIR)
    from networks.resnet_encoder import ResnetEncoder as ManyResnetEncoder
    from networks.depth_decoder import DepthDecoder as ManyDepthDecoder
    sys.path.pop(0)

    encoder = ManyResnetEncoder(18, False)
    loaded = torch.load(os.path.join(model_path, "mono_encoder.pth"), map_location=device)
    for key in ['height', 'width', 'use_stereo']:
        if key in loaded:
            del loaded[key]
    encoder.load_state_dict(loaded)
    encoder.eval().to(device)
    encoder.depth_range = (2.0, 100.0)  # ManyDepth 训练 min_depth/max_depth

    decoder = ManyDepthDecoder(encoder.num_ch_enc, scales=range(4))
    dec_state = torch.load(os.path.join(model_path, "mono_depth.pth"), map_location=device)
    decoder.load_state_dict(dec_state, strict=False)
    decoder.eval().to(device)
    return encoder, decoder


def load_litemono(model_path, device):
    """Lite-Mono 正确加载."""
    _orig_path = sys.path.copy()
    _cached_layers = sys.modules.get('layers', None)
    sys.path.insert(0, _LITEMONO_DIR)
    try:
        loader_layers = importlib.machinery.SourceFileLoader(
            '_litemono_layers', os.path.join(_LITEMONO_DIR, 'layers.py'))
        spec_layers = importlib.util.spec_from_loader('_litemono_layers', loader_layers)
        mod_layers = importlib.util.module_from_spec(spec_layers)
        sys.modules['_litemono_layers'] = mod_layers
        sys.modules['layers'] = mod_layers
        spec_layers.loader.exec_module(mod_layers)

        litemono_networks = os.path.join(_LITEMONO_DIR, 'networks')
        loader_enc = importlib.machinery.SourceFileLoader(
            '_litemono_enc', os.path.join(litemono_networks, 'depth_encoder.py'))
        spec_enc = importlib.util.spec_from_loader('_litemono_enc', loader_enc)
        mod_enc = importlib.util.module_from_spec(spec_enc)
        sys.modules['_litemono_enc'] = mod_enc
        spec_enc.loader.exec_module(mod_enc)
        LiteMono = mod_enc.LiteMono

        loader_dec = importlib.machinery.SourceFileLoader(
            '_litemono_dec', os.path.join(litemono_networks, 'depth_decoder.py'))
        spec_dec = importlib.util.spec_from_loader('_litemono_dec', loader_dec)
        mod_dec = importlib.util.module_from_spec(spec_dec)
        sys.modules['_litemono_dec'] = mod_dec
        spec_dec.loader.exec_module(mod_dec)
        LitemonoDepthDecoder = mod_dec.DepthDecoder
    finally:
        sys.path = _orig_path
        if _cached_layers is not None:
            sys.modules['layers'] = _cached_layers

    encoder = LiteMono(model='lite-mono', height=192, width=640)
    encoder.feed_size = (640, 192)  # Lite-Mono 训练输入尺寸 (width, height)
    encoder.depth_range = (2.0, 100.0)  # Lite-Mono 训练 min_depth/max_depth
    state_dict = torch.load(os.path.join(model_path, "encoder.pth"), map_location=device)
    for key in ['height', 'width', 'use_stereo']:
        if key in state_dict:
            del state_dict[key]
    encoder.load_state_dict(state_dict, strict=True)
    encoder.eval().to(device)

    decoder = LitemonoDepthDecoder(num_ch_enc=np.array([48, 80, 128]), scales=range(3))
    dec_state = torch.load(os.path.join(model_path, "depth.pth"), map_location=device)
    decoder.load_state_dict(dec_state, strict=True)
    decoder.eval().to(device)
    return encoder, decoder


# ═══════════════════════════════════════════════
# 指标
# ═══════════════════════════════════════════════

def compute_metrics(gt, pred, use_median_scale):
    mask = (gt > 0) & (gt < 500) & (pred > 0)
    g = gt[mask]
    p = pred[mask]
    if len(g) < 100:
        return None
    scale = np.median(g) / np.median(p) if use_median_scale else 1.0
    p_s = p * scale
    absrel = float(np.mean(np.abs(g - p_s) / g))
    rmse = float(np.sqrt(np.mean((g - p_s) ** 2)))
    delta = float(np.mean(np.maximum(g / p_s, p_s / g) < 1.25))
    # 比值用原始预测深度（不 median scaling），反映物理尺度偏差；越接近 1 越好
    ratio_val = float(np.mean(p) / np.mean(g))
    return dict(absrel=absrel, rmse=rmse, delta=delta, ratio=ratio_val, scale=float(scale))


# ═══════════════════════════════════════════════
def main():
    frame_files = sorted([f for f in os.listdir(WARPED_RGB_DIR) if f.endswith('.png')])
    print(f'帧数: {len(frame_files)}  Device: {DEVICE}')

    all_results = {}

    for model_name, model_path, model_type in [
        ('Monodepth2', r'C:\Users\Administrator\tmp\c3vd_md2_multi\models\weights_19', 'md2'),
        ('ManyDepth',  r'C:\Users\Administrator\tmp\c3vd_manydepth_multi\models\weights_19', 'manydepth'),
        ('Lite-Mono',  r'C:\Users\Administrator\tmp\c3vd_litemono_multi\models\weights_19', 'litemono'),
        ('Ours',       r'e:\data1\monodepth2\models\depth', 'md2'),
    ]:
        need_scale = (model_name != 'Ours')

        print(f'\n{"="*60}')
        print(f'  [{model_name}] 加载模型... need_scale={need_scale}')
        print(f'{"="*60}')

        if model_type == 'manydepth':
            encoder, decoder = load_manydepth(model_path, DEVICE)
            nb = 0  # mono_depth → sigmoid disparity
        elif model_type == 'litemono':
            encoder, decoder = load_litemono(model_path, DEVICE)
            nb = 0  # sigmoid disparity
        else:
            mgr = ModelManager(model_path=model_path, device=DEVICE)
            encoder, decoder, _, motion_encoder = mgr.load_model()
            nb = getattr(decoder, 'num_bins', 0)
            if model_name == 'Monodepth2':
                encoder.depth_range = (2.0, 100.0)  # Monodepth2 训练 min_depth/max_depth
            print(f'  num_bins={nb}, motion_encoder={"有" if motion_encoder is not None else "无"}')

        frame_metrics = []
        for fi, fname in enumerate(frame_files):
            img_path = os.path.join(WARPED_RGB_DIR, fname)
            fidx = int(fname.split('_')[1].split('.')[0])
            gt_path = os.path.join(GT_DEPTH_DIR, f'{fidx:04d}_depth.tiff')
            gt_mm = np.array(Image.open(gt_path)).astype(np.float32) * GT_SCALE

            try:
                # 所有模型统一用 predict_depth
                pred = predict_depth(img_path, encoder, decoder, DEVICE, feed_h=256, feed_w=320,
                                     min_depth=1.0, max_depth=500.0, num_bins=nb)
            except Exception as e:
                print(f'  [WARN] Frame {fidx}: {e}')
                continue

            met = compute_metrics(gt_mm, pred, use_median_scale=need_scale)
            if met:
                frame_metrics.append(met)

            if (fi + 1) % 30 == 0:
                print(f'  进度: {fi+1}/{len(frame_files)}')

        if frame_metrics:
            keys = ['absrel', 'rmse', 'delta', 'ratio', 'scale']
            summary = {k + '_mean': float(np.mean([m[k] for m in frame_metrics])) for k in keys}
            summary['n_frames'] = len(frame_metrics)
            all_results[model_name] = summary
            print(f'  n={len(frame_metrics)}, AbsRel={summary["absrel_mean"]:.4f}, '
                  f'RMSE={summary["rmse_mean"]:.1f}mm, '
                  f'δ<1.25={summary["delta_mean"]:.4f}, '
                  f'Ratio={summary["ratio_mean"]:.3f}, '
                  f'Scale={summary["scale_mean"]:.2f}')
        else:
            all_results[model_name] = {'error': 'no valid frames'}

        del encoder, decoder
        torch.cuda.empty_cache()

    # ── 打印表格 ──
    print(f'\n{"="*80}')
    print(' 表1 深度估计精度对比')
    print(f'{"="*80}')
    print(f'{"方法":<14} {"AbsRel ↓":>9} {"RMSE ↓":>9} {"δ<1.25 ↑":>9} {"Ratio ↑":>9} {"尺度因子":>9}')
    print('-' * 64)
    for name in ['Monodepth2', 'ManyDepth', 'Lite-Mono', 'Ours']:
        r = all_results.get(name, {})
        if 'error' in r:
            print(f'{name:<14} {"—":>9} {"—":>9} {"—":>9} {"—":>9} {"—":>9}')
        else:
            print(f'{name:<14} {r["absrel_mean"]:>9.4f} {r["rmse_mean"]:>9.1f} '
                  f'{r["delta_mean"]:>9.4f} {r["ratio_mean"]:>9.3f} {r["scale_mean"]:>9.2f}')

    out_path = os.path.join(OUT_DIR, 'depth_accuracy.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n结果: {out_path}')


if __name__ == '__main__':
    main()
