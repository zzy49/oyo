#!/usr/bin/env python
"""诊断 4 模型深度预测是否各异。"""

import os, sys, json, numpy as np
from PIL import Image

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import torch
from torchvision import transforms

from v6_pipeline.utils import ModelManager

_LITEMONO_DIR = r'E:\data1\Lite-Mono-main'
_MANYDEPTH_DIR = r'E:\data1\ManyDepth-main'

DEVICE = torch.device('cuda')
FEED_H, FEED_W = 256, 320

WARPED_RGB_DIR = r'F:\dataset\c1_transverse1_t1_v2\generated\rgb_warped'
GT_DEPTH_DIR   = r'F:\dataset\c1_transverse1_t1_v2\depth'
GT_SCALE = 100.0 / 65535.0

MODELS = [
    ('Monodepth2', r'C:\Users\Administrator\tmp\c3vd_md2_full\models\weights_19', 'md2'),
    ('ManyDepth',  r'C:\Users\Administrator\tmp\c3vd_manydepth_full\models\weights_4', 'manydepth'),
    ('Lite-Mono',  r'C:\Users\Administrator\tmp\c3vd_litemono_full\models\weights_19', 'litemono'),
    ('Ours',       r'e:\data1\monodepth2\models\depth', 'md2'),
]


def load_manydepth(model_path, device):
    sys.path.insert(0, _MANYDEPTH_DIR)
    from networks.resnet_encoder import ResnetEncoder
    from networks.depth_decoder import DepthDecoder
    sys.path.pop(0)
    enc = ResnetEncoder(18, False)
    loaded = torch.load(os.path.join(model_path, "encoder.pth"), map_location=device)
    enc.load_state_dict({k: v for k, v in loaded.items() if k in enc.state_dict()}, strict=False)
    enc.eval().to(device)
    dec = DepthDecoder(num_ch_enc=enc.num_ch_enc, scales=range(4))
    dec.load_state_dict(torch.load(os.path.join(model_path, "depth.pth"), map_location=device))
    dec.eval().to(device)
    return enc, dec


def load_litemono(model_path, device):
    import importlib
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
    enc = LiteMono(model='lite-mono', height=192, width=640)
    state_dict = torch.load(os.path.join(model_path, "encoder.pth"), map_location=device)
    for key in ['height', 'width', 'use_stereo']:
        if key in state_dict:
            del state_dict[key]
    enc.load_state_dict(state_dict)
    enc.eval().to(device)
    dec = LitemonoDepthDecoder(num_ch_enc=np.array([48, 80, 128]), scales=range(3))
    dec.load_state_dict(torch.load(os.path.join(model_path, "depth.pth"), map_location=device))
    dec.eval().to(device)
    return enc, dec


def predict_md2(img_path, encoder, decoder, device):
    from layers import depth_from_bins, make_bin_centers
    img = Image.open(img_path).convert('RGB')
    o_w, o_h = img.size
    img_r = img.resize((FEED_W, FEED_H), Image.LANCZOS)
    t = transforms.ToTensor()(img_r).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = encoder(t)
        outputs = decoder(feats)
        nb = getattr(decoder, 'num_bins', 0)
        if nb > 0:
            bins = outputs[("bins", 0)]
            residual = outputs.get(("residual", 0), None)
            centers = make_bin_centers(1.0, 500.0, nb)
            b_up = torch.nn.functional.interpolate(bins, (o_h, o_w), mode='bilinear', align_corners=False)
            r_up = torch.nn.functional.interpolate(residual, (o_h, o_w), mode='bilinear', align_corners=False) if residual is not None else None
            depth = depth_from_bins(b_up, centers, r_up)
        else:
            disp = outputs[("disp", 0)]
            d_up = torch.nn.functional.interpolate(disp, (o_h, o_w), mode='bilinear', align_corners=False)
            depth = 1.0 / (d_up + 1e-8)
    return depth.cpu().numpy().squeeze()


def predict_manydepth(img_path, encoder, decoder, device):
    img = Image.open(img_path).convert('RGB')
    o_w, o_h = img.size
    img_r = img.resize((FEED_W, FEED_H), Image.LANCZOS)
    t = transforms.ToTensor()(img_r).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = encoder(t)
        outputs = decoder(feats)
        disp = outputs[("disp", 0)]
        d_up = torch.nn.functional.interpolate(disp, (o_h, o_w), mode='bilinear', align_corners=False)
        depth = 1.0 / (d_up + 1e-8)
    return depth.cpu().numpy().squeeze()


def predict_litemono(img_path, encoder, decoder, device):
    img = Image.open(img_path).convert('RGB')
    o_w, o_h = img.size
    img_r = img.resize((FEED_W, FEED_H), Image.LANCZOS)
    t = transforms.ToTensor()(img_r).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = encoder(t)
        outputs = decoder(feats)
        disp = outputs[("disp", 0)]
        d_up = torch.nn.functional.interpolate(disp, (o_h, o_w), mode='bilinear', align_corners=False)
        depth = 1.0 / (d_up + 1e-8)
    return depth.cpu().numpy().squeeze()


def compute_metrics(gt, pred, use_median_scale=True):
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
    return absrel, rmse, delta, scale


# ═══ 诊断: 取前 3 帧, 对比 4 模型原始输出 ═══
frame_files = sorted([f for f in os.listdir(WARPED_RGB_DIR) if f.endswith('.png')])[:3]
test_frames = [os.path.join(WARPED_RGB_DIR, f) for f in frame_files]
test_gts = [os.path.join(GT_DEPTH_DIR, f'{int(f.split("_")[1].split(".")[0]):04d}_depth.tiff') for f in frame_files]

results = {}

for name, path, mtype in MODELS:
    print(f'\n── {name} ──')
    if mtype == 'manydepth':
        enc, dec = load_manydepth(path, DEVICE)
    elif mtype == 'litemono':
        enc, dec = load_litemono(path, DEVICE)
    else:
        mgr = ModelManager(model_path=path, device=DEVICE)
        enc, dec, _, me = mgr.load_model()

    # 打印 encoder / decoder 关键参数
    nb = getattr(dec, 'num_bins', 0)
    print(f'  decoder.num_bins = {nb}')
    print(f'  decoder.num_ch_enc = {getattr(dec, "num_ch_enc", "N/A")}')
    print(f'  encoder type = {type(enc).__name__}')

    preds = []
    for img_path, gt_path in zip(test_frames, test_gts):
        gt = np.array(Image.open(gt_path)).astype(np.float32) * GT_SCALE
        if mtype == 'manydepth':
            pred = predict_manydepth(img_path, enc, dec, DEVICE)
        elif mtype == 'litemono':
            pred = predict_litemono(img_path, enc, dec, DEVICE)
        else:
            pred = predict_md2(img_path, enc, dec, DEVICE)
        # 打印原始统计
        mask = (gt > 0) & (gt < 500) & (pred > 0)
        raw_median = float(np.median(pred[mask])) if mask.sum() > 0 else 0
        need_scale = (name != 'Ours')
        s = np.median(gt[mask]) / np.median(pred[mask]) if (need_scale and mask.sum() > 0) else 1.0
        print(f'  Frame {os.path.basename(img_path)}: pred_median={raw_median:.3f}, GT_median={np.median(gt[mask]):.2f}, scale={s:.2f}')
        met = compute_metrics(gt, pred, use_median_scale=need_scale)
        if met:
            preds.append(met)

    if preds:
        absrels = [p[0] for p in preds]
        rmses = [p[1] for p in preds]
        deltas = [p[2] for p in preds]
        scales = [p[3] for p in preds]
        print(f'  → AbsRel avg={np.mean(absrels):.4f}, RMSE={np.mean(rmses):.1f}, δ={np.mean(deltas):.4f}, scale={np.mean(scales):.2f}')
    results[name] = preds

    del enc, dec
    torch.cuda.empty_cache()

# 两两对比 pred 是否一致
print('\n\n── 跨模型预测一致性检查 ──')
for i in range(len(test_frames)):
    print(f'\n  Frame {os.path.basename(test_frames[i])}:')
    # 快速重跑所有模型获取原始预测
    all_raw = {}
    gt = np.array(Image.open(test_gts[i])).astype(np.float32) * GT_SCALE
    for name, path, mtype in MODELS:
        if mtype == 'manydepth':
            enc, dec = load_manydepth(path, DEVICE)
            pred = predict_manydepth(test_frames[i], enc, dec, DEVICE)
        elif mtype == 'litemono':
            enc, dec = load_litemono(path, DEVICE)
            pred = predict_litemono(test_frames[i], enc, dec, DEVICE)
        else:
            mgr = ModelManager(model_path=path, device=DEVICE)
            enc, dec, _, me = mgr.load_model()
            pred = predict_md2(test_frames[i], enc, dec, DEVICE)
        mask = (gt > 0) & (gt < 500) & (pred > 0)
        all_raw[name] = pred[mask]
        print(f'    {name}: mean={pred[mask].mean():.3f}, std={pred[mask].std():.3f}, median={np.median(pred[mask]):.3f}')
        del enc, dec
        torch.cuda.empty_cache()

    # 检查是否完全一致
    names = ['Monodepth2', 'ManyDepth', 'Lite-Mono', 'Ours']
    for a, b in [('Monodepth2', 'ManyDepth'), ('Monodepth2', 'Lite-Mono'), ('ManyDepth', 'Lite-Mono')]:
        diff = np.abs(all_raw[a] - all_raw[b]).max()
        print(f'    max|{a} - {b}| = {diff:.6f}')
