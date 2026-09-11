import io

files = [
    (r'e:\data1\manydepth-master\manydepth\options.py',
     '"cityscapes_preprocessed", "c3vd_full", "multi_c3vd"',
     '"cityscapes_preprocessed", "c3vd_full", "multi_c3vd", "multi_c3vd_gt"'),
    (r'E:\data1\Lite-Mono-main\options.py',
     '"benchmark", "c3vd_full", "multi_c3vd"',
     '"benchmark", "c3vd_full", "multi_c3vd", "multi_c3vd_gt"'),
]

for p, old, new in files:
    s = io.open(p, encoding='utf-8').read()
    if old in s:
        s = s.replace(old, new)
        io.open(p, 'w', encoding='utf-8').write(s)
        print('fixed:', p)
    else:
        print('NOT FOUND:', p)
