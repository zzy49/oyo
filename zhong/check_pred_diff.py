#!/usr/bin/env python
"""对比三个基线模型的原始预测值是否真的不同"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from PIL import Image

_MANYDEPTH_DIR = r'E:\data1\ManyDepth-main'
_LITEMONO_DIR = r'E:\data1\Lite-Mono-main'
FRAME = r'F:/dataset/c1_transverse1_t1_v2/generated/rgb_warped/frame_0000.png'
GT = r'F:/dataset/c1_transverse1_t1_v2/depth/0000_depth.tiff'
DEV = torch.device('cuda')

# ── 1. Monodepth2 ──
from v6_pipeline.utils import ModelManager, predict_depth
mgr = ModelManager(model_path=r'C:\Users\Administrator\tmp\c3vd_md2_full\models\weights_19', device=DEV)
enc1, dec1, _, _ = mgr.load_model()
nb1 = getattr(dec1, 'num_bins', 0)
pred1 = predict_depth(FRAME, enc1, dec1, DEV, 256, 320, num_bins=nb1)

# ── 2. ManyDepth ──
sys.path.insert(0, _MANYDEPTH_DIR)
from networks.resnet_encoder import ResnetEncoder as ManyRE
from networks.depth_decoder import DepthDecoder as ManyDD
sys.path.pop(0)
enc2 = ManyRE(18, False)
loaded2 = torch.load(r'C:\Users\Administrator\tmp\c3vd_manydepth_full\models\weights_4\mono_encoder.pth', map_location=DEV)
for k in ['height','width','use_stereo']:
    loaded2.pop(k, None)
enc2.load_state_dict(loaded2); enc2.eval().to(DEV)
dec2 = ManyDD(enc2.num_ch_enc, scales=range(4))
dec2.load_state_dict(torch.load(r'C:\Users\Administrator\tmp\c3vd_manydepth_full\models\weights_4\mono_depth.pth', map_location=DEV), strict=False)
dec2.eval().to(DEV)
pred2 = predict_depth(FRAME, enc2, dec2, DEV, 256, 320, num_bins=0)

# ── 3. Lite-Mono ──
import importlib
_orig = sys.path.copy()
_cached = sys.modules.get('layers')
sys.path.insert(0, _LITEMONO_DIR)
try:
    ll = importlib.machinery.SourceFileLoader('_ll', os.path.join(_LITEMONO_DIR,'layers.py'))
    spec_l = importlib.util.spec_from_loader('_ll', ll)
    mod_l = importlib.util.module_from_spec(spec_l)
    sys.modules['_ll'] = mod_l; sys.modules['layers'] = mod_l
    spec_l.loader.exec_module(mod_l)
    ln = os.path.join(_LITEMONO_DIR, 'networks')
    le = importlib.machinery.SourceFileLoader('_le', os.path.join(ln,'depth_encoder.py'))
    spe = importlib.util.spec_from_loader('_le', le)
    me = importlib.util.module_from_spec(spe)
    sys.modules['_le'] = me; spe.loader.exec_module(me)
    ld = importlib.machinery.SourceFileLoader('_ld', os.path.join(ln,'depth_decoder.py'))
    spd = importlib.util.spec_from_loader('_ld', ld)
    md = importlib.util.module_from_spec(spd)
    sys.modules['_ld'] = md; spd.loader.exec_module(md)
finally:
    sys.path = _orig
    if _cached: sys.modules['layers'] = _cached

enc3 = me.LiteMono(model='lite-mono', height=192, width=640)
s3 = torch.load(r'C:\Users\Administrator\tmp\c3vd_litemono_full\models\weights_19\encoder.pth', map_location=DEV)
for k in ['height','width','use_stereo']: s3.pop(k, None)
enc3.load_state_dict(s3); enc3.eval().to(DEV)
dec3 = md.DepthDecoder(num_ch_enc=np.array([48,80,128]), scales=range(3))
dec3.load_state_dict(torch.load(r'C:\Users\Administrator\tmp\c3vd_litemono_full\models\weights_19\depth.pth', map_location=DEV))
dec3.eval().to(DEV)
pred3 = predict_depth(FRAME, enc3, dec3, DEV, 256, 320, num_bins=0)

# ── 对比 ──
gt = np.array(Image.open(GT)).astype(np.float32) * (100.0 / 65535.0)
mask = (gt > 0) & (gt < 500)

for label, p in [('MD2', pred1), ('MD', pred2), ('LM', pred3)]:
    v = p[mask]
    print(f'{label}: mean={v.mean():.3f} med={np.median(v):.3f} std={v.std():.3f} min={v.min():.3f} max={v.max():.3f}')

diff_12 = np.abs(pred1[mask] - pred2[mask]).max()
diff_13 = np.abs(pred1[mask] - pred3[mask]).max()
diff_23 = np.abs(pred2[mask] - pred3[mask]).max()
print(f'\nmax|MD2-MD|={diff_12:.6f}, max|MD2-LM|={diff_13:.6f}, max|MD-LM|={diff_23:.6f}')

# 如果 max diff 都接近 0 → 三个模型输出完全一致 → 有 bug
if diff_12 < 0.01:
    print('\n⚠️  所有三个模型输出像素级完全一致！')
else:
    print('\n✓ 模型输出不同')

del enc1, enc2, enc3, dec1, dec2, dec3
torch.cuda.empty_cache()
