"""多源路由补丁: 修改 c3vd_dataset.py 支持 @endo@ 和 @scared@ 前缀
适用于 Monodepth2 / ManyDepth / Lite-Mono 三份基线
"""

# 在 get_color 方法开头添加多源路由
GET_COLOR_PATCH = '''
    def get_color(self, folder, frame_index, side, do_flip):
        """Load RGB image, routing to correct data source."""
        # ── 多源路由 ──
        if folder.startswith('@endo@'):
            scene = folder[6:]  # e.g. "UnityCam/Colon"
            img_path = os.path.join(
                self._endo_root, scene, 'Frames_jpg',
                'image_{:04d}.jpg'.format(frame_index))
            if not os.path.isfile(img_path):
                # 尝试 Frames/ 目录（PNG格式）
                img_path = os.path.join(
                    self._endo_root, scene, 'Frames',
                    'image_{:04d}.png'.format(frame_index))
            if not os.path.isfile(img_path):
                color = Image.new('RGB', (self.width, self.height))
            else:
                color = self.loader(img_path)
            if do_flip:
                color = color.transpose(pil.FLIP_LEFT_RIGHT)
            return color

        if folder.startswith('@scared@'):
            subdir = folder[8:]  # e.g. "dataset_1/keyframe_1"
            fname = 'frame_{:06d}.png'.format(frame_index)
            img_path = os.path.join(self._scared_root, subdir, fname)
            if not os.path.isfile(img_path):
                color = Image.new('RGB', (self.width, self.height))
            else:
                color = self.loader(img_path)
            if do_flip:
                color = color.transpose(pil.FLIP_LEFT_RIGHT)
            return color

        # ── 默认: Real Colon (F:\\zhuan) ──
        img_path = os.path.join(self.data_path, folder, "rgb",
                                "{:04d}.png".format(frame_index))
        color = self.loader(img_path)
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color
'''

GET_DEPTH_PATCH = '''
    def get_depth(self, folder, frame_index, side, do_flip):
        """Load depth map, routing to correct data source."""
        # ── 多源路由 ──
        if folder.startswith('@endo@'):
            # EndoSLAM 无 GT 深度
            return np.zeros((self.height, self.width), dtype=np.float32)

        if folder.startswith('@scared@'):
            subdir = folder[8:]
            depth_fname = 'frame_{:06d}_depth.npy'.format(frame_index)
            depth_path = os.path.join(self._scared_root, subdir, depth_fname)
            if not os.path.isfile(depth_path):
                return np.zeros((self.height, self.width), dtype=np.float32)
            depth = np.load(depth_path)
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            depth = np.clip(depth, 1.0, 500.0)
            # 调整尺寸
            if depth.shape[0] != self.height or depth.shape[1] != self.width:
                h_scale = self.height / depth.shape[0]
                w_scale = self.width / depth.shape[1]
                import cv2
                depth = cv2.resize(depth, (self.width, self.height),
                                   interpolation=cv2.INTER_NEAREST)
            if do_flip:
                depth = np.fliplr(depth)
            return depth

        # ── 默认: Real Colon (F:\\zhuan) ──
        depth_path = os.path.join(self.data_path, folder, "depth",
                                  "{:04d}_depth.tiff".format(frame_index))
        if not os.path.isfile(depth_path):
            return np.zeros((self.height, self.width), dtype=np.float32)
        depth_img = pil.open(depth_path)
        depth_img = depth_img.resize((self.width, self.height), pil.NEAREST)
        depth = np.array(depth_img, dtype=np.float32)
        depth = np.clip(depth, 1.0, 500.0)
        if do_flip:
            depth = np.fliplr(depth)
        return depth
'''

CHECK_DEPTH_PATCH = '''
    def check_depth(self):
        """Check if depth is available (considering multi-source)."""
        folder, frame_index, _ = self._parse_line(0)
        if folder.startswith('@endo@'):
            return False  # EndoSLAM 无深度
        if folder.startswith('@scared@'):
            subdir = folder[8:]
            fname = 'frame_{:06d}_depth.npy'.format(frame_index)
            return os.path.isfile(os.path.join(
                self._scared_root, subdir, fname))
        depth_path = os.path.join(self.data_path, folder, "depth",
                                  "{:04d}_depth.tiff".format(frame_index))
        return os.path.isfile(depth_path)
'''

INIT_PATCH = '''
    def __init__(self, *args, **kwargs):
        super(C3VDDataset, self).__init__(*args, **kwargs)

        # ── 多源数据根目录 ──
        self._endo_root = r'E:\\data1\\monodepth2\\EndoSLAM'
        self._scared_root = r'E:\\data1\\monodepth2\\scared_extracted'

        # C3VD intrinsics: 1024x768 original
        self.K = np.array([[600.0, 0, 512.0, 0],
                           [0, 600.0, 384.0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
'''


# ═══════════════════════════════════════════════════════
# 输出补丁文件（供手动应用或通过脚本批量替换）
# ═══════════════════════════════════════════════════════
if __name__ == '__main__':
    print("多源路由补丁定义")
    print(f"  get_color: {len(GET_COLOR_PATCH)} 字符")
    print(f"  get_depth: {len(GET_DEPTH_PATCH)} 字符")
    print(f"  check_depth: {len(CHECK_DEPTH_PATCH)} 字符")
    print(f"  __init__: {len(INIT_PATCH)} 字符")
    print("\n使用方式: 将这些补丁替换到 c3vd_dataset.py 的对应方法中")
