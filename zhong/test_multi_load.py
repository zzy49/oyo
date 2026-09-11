"""快速验证多源数据加载"""
import sys, os
sys.path.insert(0, r'e:\data1\monodepth2\huifu\monodepth2-master')

fpath = r'e:\data1\monodepth2\huifu\monodepth2-master\splits\multi_c3vd\train_files.txt'
with open(fpath) as f:
    lines = [l.strip() for l in f if l.strip()]
print(f'Total lines: {len(lines)}')
endo = sum(1 for l in lines if l.startswith('@endo@'))
scared = sum(1 for l in lines if l.startswith('@scared@'))
real = len(lines) - endo - scared
print(f'EndoSLAM={endo}, SCARED={scared}, Real={real}')

from datasets.c3vd_dataset import C3VDDataset

# 测试三层来源的前几个
for prefix, label in [('@endo@', 'EndoSLAM'), ('@scared@', 'SCARED'), (None, 'Real Colon')]:
    if prefix:
        test_lines = [l for l in lines if l.startswith(prefix)][:2]
    else:
        test_lines = [l for l in lines if not l.startswith('@')][:2]
    if not test_lines:
        print(f'{label}: 无数据')
        continue
    print(f'\n{label} ({len(test_lines)} entries):')
    try:
        ds = C3VDDataset(
            data_path='F:/zhuan', filenames=test_lines,
            height=192, width=640, frame_idxs=[0, -1, 1],
            num_scales=4, is_train=True, img_ext='.png')
        for i in range(len(test_lines)):
            item = ds[i]
            c0 = item[("color", 0, -1)]
            print(f'  [{i}] {test_lines[i]} -> color shape={c0.shape}')
            if "depth_gt" in item:
                print(f'       depth_gt shape={item["depth_gt"].shape}')
    except Exception as e:
        print(f'  ERROR: {e}')
        import traceback; traceback.print_exc()

print('\nDONE')
