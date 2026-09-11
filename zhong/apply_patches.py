"""批量修补 ManyDepth 和 Lite-Mono 的 c3vd_dataset.py 支持多源路由"""
import os

TARGETS = [
    r'e:\data1\manydepth-master\manydepth\datasets\c3vd_dataset.py',
    r'E:\data1\Lite-Mono-main\datasets\c3vd_dataset.py',
]

for fp in TARGETS:
    print(f"\n{'='*60}")
    print(f"Patching: {fp}")
    with open(fp, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Add cv2 import
    old_import = 'import numpy as np\nimport PIL.Image as pil'
    new_import = 'import numpy as np\nimport cv2\nimport PIL.Image as pil'
    if old_import in content:
        content = content.replace(old_import, new_import)
        print("  Added cv2 import")

    # 2. Replace __init__ - use the exact text from the file
    old_init = (
        '    def __init__(self, *args, **kwargs):\n'
        '        super(C3VDDataset, self).__init__(*args, **kwargs)\n'
        '        self.K = np.array([[600.0, 0, 512.0, 0],\n'
        '                           [0, 600.0, 384.0, 0],\n'
        '                           [0, 0, 1, 0],\n'
        '                           [0, 0, 0, 1]], dtype=np.float32)'
    )
    new_init = (
        '    def __init__(self, *args, **kwargs):\n'
        '        super(C3VDDataset, self).__init__(*args, **kwargs)\n'
        '\n'
        '        # ── Multi-source data roots ──\n'
        "        self._endo_root = r'E:\\\\data1\\\\monodepth2\\\\EndoSLAM'\n"
        "        self._scared_root = r'E:\\\\data1\\\\monodepth2\\\\scared_extracted'\n"
        '\n'
        '        self.K = np.array([[600.0, 0, 512.0, 0],\n'
        '                           [0, 600.0, 384.0, 0],\n'
        '                           [0, 0, 1, 0],\n'
        '                           [0, 0, 0, 1]], dtype=np.float32)'
    )
    if old_init in content:
        content = content.replace(old_init, new_init)
        print("  Patched __init__")
    else:
        print("  __init__ NOT MATCHED!")

    # 3. Replace check_depth
    old_cd = (
        '    def check_depth(self):\n'
        '        folder, frame_index, _ = self._parse_line(0)\n'
        '        depth_path = os.path.join(self.data_path, folder, "depth",\n'
        '                                  "{:04d}_depth.tiff".format(frame_index))\n'
        '        return os.path.isfile(depth_path)'
    )
    new_cd = (
        '    def check_depth(self):\n'
        '        """Check if depth is available, considering multi-source."""\n'
        '        folder, frame_index, _ = self._parse_line(0)\n'
        "        if folder.startswith('@endo@'):\n"
        '            return False\n'
        "        if folder.startswith('@scared@'):\n"
        '            subdir = folder[8:]\n'
        "            fname = 'frame_{:06d}_depth.npy'.format(frame_index)\n"
        '            return os.path.isfile(os.path.join(\n'
        '                self._scared_root, subdir, fname))\n'
        '        depth_path = os.path.join(self.data_path, folder, "depth",\n'
        '                                  "{:04d}_depth.tiff".format(frame_index))\n'
        '        return os.path.isfile(depth_path)'
    )
    if old_cd in content:
        content = content.replace(old_cd, new_cd)
        print("  Patched check_depth")
    else:
        # Try single quote version
        old_cd_single = (
            '    def check_depth(self):\n'
            '        folder, frame_index, _ = self._parse_line(0)\n'
            "        depth_path = os.path.join(self.data_path, folder, 'depth',\n"
            "                                  '{:04d}_depth.tiff'.format(frame_index))\n"
            '        return os.path.isfile(depth_path)'
        )
        if old_cd_single in content:
            content = content.replace(old_cd_single, new_cd.replace('"depth"', "'depth'").replace(
                '"{:04d}_depth.tiff"', "'{:04d}_depth.tiff'"))
            print("  Patched check_depth (single-quote)")
        else:
            print("  check_depth NOT MATCHED!")

    # 4. Replace get_color
    old_gc = (
        '    def get_color(self, folder, frame_index, side, do_flip):\n'
        '        img_path = os.path.join(self.data_path, folder, "rgb",\n'
        '                                "{:04d}.png".format(frame_index))\n'
        '        color = self.loader(img_path)\n'
        '        if do_flip:\n'
        '            color = color.transpose(pil.FLIP_LEFT_RIGHT)\n'
        '        return color'
    )
    new_gc = (
        '    def get_color(self, folder, frame_index, side, do_flip):\n'
        '        """Load RGB, routing to multi-source."""\n'
        "        if folder.startswith('@endo@'):\n"
        '            scene = folder[6:]\n'
        '            img_path = os.path.join(\n'
        "                self._endo_root, scene, 'Frames_jpg',\n"
        "                'image_{:04d}.jpg'.format(frame_index))\n"
        '            if not os.path.isfile(img_path):\n'
        '                img_path = os.path.join(\n'
        "                    self._endo_root, scene, 'Frames',\n"
        "                    'image_{:04d}.png'.format(frame_index))\n"
        '            if not os.path.isfile(img_path):\n'
        "                color = pil.Image.new('RGB', (self.width, self.height))\n"
        '            else:\n'
        '                color = self.loader(img_path)\n'
        '            if do_flip:\n'
        '                color = color.transpose(pil.FLIP_LEFT_RIGHT)\n'
        '            return color\n'
        '\n'
        "        if folder.startswith('@scared@'):\n"
        '            subdir = folder[8:]\n'
        "            fname = 'frame_{:06d}.png'.format(frame_index)\n"
        '            img_path = os.path.join(self._scared_root, subdir, fname)\n'
        '            if not os.path.isfile(img_path):\n'
        "                color = pil.Image.new('RGB', (self.width, self.height))\n"
        '            else:\n'
        '                color = self.loader(img_path)\n'
        '            if do_flip:\n'
        '                color = color.transpose(pil.FLIP_LEFT_RIGHT)\n'
        '            return color\n'
        '\n'
        '        img_path = os.path.join(self.data_path, folder, "rgb",\n'
        '                                "{:04d}.png".format(frame_index))\n'
        '        color = self.loader(img_path)\n'
        '        if do_flip:\n'
        '            color = color.transpose(pil.FLIP_LEFT_RIGHT)\n'
        '        return color'
    )
    if old_gc in content:
        content = content.replace(old_gc, new_gc)
        print("  Patched get_color")
    else:
        print("  get_color NOT MATCHED!")

    # 5. Replace get_depth
    old_gd = (
        '    def get_depth(self, folder, frame_index, side, do_flip):\n'
        '        depth_path = os.path.join(self.data_path, folder, "depth",\n'
        '                                  "{:04d}_depth.tiff".format(frame_index))\n'
        '        if not os.path.isfile(depth_path):\n'
        '            return np.zeros((self.height, self.width), dtype=np.float32)\n'
        '        depth_img = pil.open(depth_path)\n'
        '        depth_img = depth_img.resize((self.width, self.height), pil.NEAREST)\n'
        '        depth = np.array(depth_img, dtype=np.float32)\n'
        '        depth = np.clip(depth, 1.0, 500.0)\n'
        '        if do_flip:\n'
        '            depth = np.fliplr(depth)\n'
        '        return depth'
    )
    new_gd = (
        '    def get_depth(self, folder, frame_index, side, do_flip):\n'
        '        """Load depth, routing to multi-source."""\n'
        "        if folder.startswith('@endo@'):\n"
        '            return np.zeros((self.height, self.width), dtype=np.float32)\n'
        '\n'
        "        if folder.startswith('@scared@'):\n"
        '            subdir = folder[8:]\n'
        "            depth_fname = 'frame_{:06d}_depth.npy'.format(frame_index)\n"
        '            depth_path = os.path.join(self._scared_root, subdir, depth_fname)\n'
        '            if not os.path.isfile(depth_path):\n'
        '                return np.zeros((self.height, self.width), dtype=np.float32)\n'
        '            depth = np.load(depth_path)\n'
        '            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)\n'
        '            depth = np.clip(depth, 1.0, 500.0)\n'
        '            if depth.shape[0] != self.height or depth.shape[1] != self.width:\n'
        '                depth = cv2.resize(depth, (self.width, self.height),\n'
        '                                   interpolation=cv2.INTER_NEAREST)\n'
        '            if do_flip:\n'
        '                depth = np.fliplr(depth)\n'
        '            return depth\n'
        '\n'
        '        depth_path = os.path.join(self.data_path, folder, "depth",\n'
        '                                  "{:04d}_depth.tiff".format(frame_index))\n'
        '        if not os.path.isfile(depth_path):\n'
        '            return np.zeros((self.height, self.width), dtype=np.float32)\n'
        '        depth_img = pil.open(depth_path)\n'
        '        depth_img = depth_img.resize((self.width, self.height), pil.NEAREST)\n'
        '        depth = np.array(depth_img, dtype=np.float32)\n'
        '        depth = np.clip(depth, 1.0, 500.0)\n'
        '        if do_flip:\n'
        '            depth = np.fliplr(depth)\n'
        '        return depth'
    )
    if old_gd in content:
        content = content.replace(old_gd, new_gd)
        print("  Patched get_depth")
    else:
        print("  get_depth NOT MATCHED!")

    with open(fp, 'w', encoding='utf-8') as f:
        f.write(content)

    # 验证
    with open(fp, 'r', encoding='utf-8') as f:
        verify = f.read()
    for keyword in ['_endo_root', '_scared_root', '@endo@', '@scared@',
                     'import cv2', 'frames_jpg', 'frame_{:06d}_depth.npy']:
        if keyword.lower() in verify.lower():
            print(f"  VERIFIED: {keyword}")
        else:
            print(f"  MISSING: {keyword}")

print(f"\n{'='*60}")
print("Done! All patches applied.")
