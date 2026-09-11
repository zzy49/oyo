#!/usr/bin/env python
"""用EACH模型的 ORIGINAL 推理代码做深度预测，确保没有适配层问题"""
import sys, os, json
import numpy as np, torch
from PIL import Image
import torchvision.transforms as T
import importlib

DEV = torch.device('cuda')
_MD2_DIR = r'E:\data1\monodepth2'
_MD_DIR = r'E:\data1\ManyDepth-main'
_LM_DIR = r'E:\data1\Lite-Mono-main'

IMG_DIR = r'F:/dataset/c1_transverse1_t1_v2/generated/rgb_warped'
GT_DIR = r'F:/dataset/c1_transverse1_t1_v2/depth'
GT_SCALE = 100.0 / 65535.0

def compute_metrics(gt, pred, need_median_scale):
    mask = (gt > 0) & (gt < 500) & (pred > 0)
    g = gt[mask]; p = pred[mask]
    if len(g) < 100: return None
    s = np.median(g) / np.median(p) if need_median_scale else 1.0
    ps = p * s
    return dict(absrel=float(np.mean(np.abs(g - ps) / g)),
                rmse=float(np.sqrt(np.mean((g - ps)**2))),
                delta=float(np.mean(np.maximum(g/ps, ps/g) < 1.25)),
                ratio=float(np.mean(ps)/np.mean(g)), scale=float(s))


# ═══ 1. Monodepth2 (用原始 repo 的 predict) ═══
print('=== Monodepth2 ===')
sys.path.insert(0, _MD2_DIR)
from networks.resnet_encoder import ResnetEncoder
from networks.depth_decoder import DepthDecoder
from layers import disp_to_depth as md2_disp_to_depth
sys.path.pop(0)

enc = ResnetEncoder(18, False)
e_state = torch.load(r'C:\Users\Administrator\tmp\c3vd_md2_full\models\weights_19\encoder.pth', map_location=DEV)
for k in ['height','width','use_stereo']:
    if k in e_state: del e_state[k]
enc.load_state_dict(e_state); enc.eval().to(DEV)

dec = DepthDecoder(enc.num_ch_enc, scales=range(4))
d_state = torch.load(r'C:\Users\Administrator\tmp\c3vd_md2_full\models\weights_19\depth.pth', map_location=DEV)
missing, unexpected = dec.load_state_dict(d_state, strict=False)
print(f'  decoder: {len(missing)} missing, {len(unexpected)} unexpected')
dec.eval().to(DEV)

# 用原始 monodepth2 推理方式
img = Image.open(os.path.join(IMG_DIR, 'frame_0000.png')).convert('RGB')
o_w, o_h = img.size
t = T.Compose([T.Resize((256, 320), Image.LANCZOS), T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])(img).unsqueeze(0).to(DEV)
with torch.no_grad():
    feats = enc(t)
    outs = dec(feats)
    disp = outs[("disp", 0)]
    disp_up = torch.nn.functional.interpolate(disp, (o_h,o_w), mode='bilinear', align_corners=False)
    _, depth_md2 = md2_disp_to_depth(disp_up, 1.0, 500.0)
    depth_md2 = depth_md2.cpu().numpy().squeeze()
    print(f'  depth: min={depth_md2.min():.2f}, max={depth_md2.max():.1f}, mean={depth_md2.mean():.2f}, med={np.median(depth_md2):.2f}')

del enc, dec; torch.cuda.empty_cache()


# ═══ 2. ManyDepth (mono_*.pth) ═══
print('\n=== ManyDepth ===')
sys.path.insert(0, _MD_DIR)
from networks.resnet_encoder import ResnetEncoder as MDEnc
from networks.depth_decoder import DepthDecoder as MDDec
from layers import disp_to_depth as md_disp_to_depth
sys.path.pop(0)

enc2 = MDEnc(18, False)
e2 = torch.load(r'C:\Users\Administrator\tmp\c3vd_manydepth_full\models\weights_4\mono_encoder.pth', map_location=DEV)
for k in ['height','width','use_stereo']:
    if k in e2: del e2[k]
enc2.load_state_dict(e2); enc2.eval().to(DEV)

dec2 = MDDec(enc2.num_ch_enc, scales=range(4))
d2 = torch.load(r'C:\Users\Administrator\tmp\c3vd_manydepth_full\models\weights_4\mono_depth.pth', map_location=DEV)
missing, unexpected = dec2.load_state_dict(d2, strict=False)
print(f'  decoder: {len(missing)} missing, {len(unexpected)} unexpected')
dec2.eval().to(DEV)

with torch.no_grad():
    feats2 = enc2(t)
    outs2 = dec2(feats2)
    disp2 = outs2[("disp", 0)]
    disp_up2 = torch.nn.functional.interpolate(disp2, (o_h,o_w), mode='bilinear', align_corners=False)
    _, depth_md = md_disp_to_depth(disp_up2, 1.0, 500.0)
    depth_md = depth_md.cpu().numpy().squeeze()
    print(f'  depth: min={depth_md.min():.2f}, max={depth_md.max():.1f}, mean={depth_md.mean():.2f}, med={np.median(depth_md):.2f}')

del enc2, dec2; torch.cuda.empty_cache()


# ═══ 3. Lite-Mono ═══
print('\n=== Lite-Mono ===')
_orig = sys.path.copy(); _cached = sys.modules.get('layers')
sys.path.insert(0, _LM_DIR)
try:
    l1 = importlib.machinery.SourceFileLoader('_lm_layers', os.path.join(_LM_DIR,'layers.py'))
    s1 = importlib.util.spec_from_loader('_lm_layers', l1)
    m1 = importlib.util.module_from_spec(s1)
    sys.modules['_lm_layers'] = m1; sys.modules['layers'] = m1
    s1.loader.exec_module(m1)

    ln = os.path.join(_LM_DIR, 'networks')
    l2 = importlib.machinery.SourceFileLoader('_lm_enc', os.path.join(ln,'depth_encoder.py'))
    s2 = importlib.util.spec_from_loader('_lm_enc', l2)
    m2 = importlib.util.module_from_spec(s2)
    sys.modules['_lm_enc'] = m2; s2.loader.exec_module(m2)

    l3 = importlib.machinery.SourceFileLoader('_lm_dec', os.path.join(ln,'depth_decoder.py'))
    s3 = importlib.util.spec_from_loader('_lm_dec', l3)
    m3 = importlib.util.module_from_spec(s3)
    sys.modules['_lm_dec'] = m3; s3.loader.exec_module(m3)
finally:
    sys.path = _orig
    if _cached: sys.modules['layers'] = _cached

enc3 = m2.LiteMono(model='lite-mono', height=192, width=640)
e3 = torch.load(r'C:\Users\Administrator\tmp\c3vd_litemono_full\models\weights_19\encoder.pth', map_location=DEV)
for k in ['height','width','use_stereo']:
    if k in e3: del e3[k]
enc3.load_state_dict(e3); enc3.eval().to(DEV)

dec3 = m3.DepthDecoder(num_ch_enc=np.array([48,80,128]), scales=range(3))
d3 = torch.load(r'C:\Users\Administrator\tmp\c3vd_litemono_full\models\weights_19\depth.pth', map_location=DEV)
missing, unexpected = dec3.load_state_dict(d3, strict=False)
print(f'  decoder: {len(missing)} missing, {len(unexpected)} unexpected')
dec3.eval().to(DEV)

t3 = T.Compose([T.Resize((256, 320), Image.LANCZOS), T.ToTensor()])(img).unsqueeze(0).to(DEV)
with torch.no_grad():
    feats3 = enc3(t3)
    outs3 = dec3(feats3)
    disp3 = outs3[("disp", 0)]
    disp_up3 = torch.nn.functional.interpolate(disp3, (o_h,o_w), mode='bilinear', align_corners=False)
    _, depth_lm = m1.disp_to_depth(disp_up3, 1.0, 500.0)
    depth_lm = depth_lm.cpu().numpy().squeeze()
    print(f'  disp key: {list(outs3.keys())}')
    print(f'  depth: min={depth_lm.min():.2f}, max={depth_lm.max():.1f}, mean={depth_lm.mean():.2f}, med={np.median(depth_lm):.2f}')

del enc3, dec3; torch.cuda.empty_cache()

# ═══ Comparison ═══
gt = np.array(Image.open(os.path.join(GT_DIR, '0000_depth.tiff'))).astype(np.float32) * GT_SCALE
print('\n=== Per-model metrics (1 frame) ===')
for name, pred, need_s in [('MD2', depth_md2, True), ('ManyDepth', depth_md, True), ('LiteMono', depth_lm, True)]:
    m = compute_metrics(gt, pred, need_s)
    if m:
        print(f'  {name}: AbsRel={m["absrel"]:.4f}, RMSE={m["rmse"]:.1f}, d={m["delta"]:.4f}, Scale={m["scale"]:.2f}')

print(f'\nmax|MD2-MD|={np.abs(depth_md2-depth_md).max():.4f}')
print(f'max|MD2-LM|={np.abs(depth_md2-depth_lm).max():.4f}')
