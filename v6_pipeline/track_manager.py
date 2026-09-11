"""Track Manager: 长期特征跟踪系统

核心思路:
- LK 维护已有Track的帧间延续性 (保持Track ID)
- LoFTR 负责补充新点和恢复丢失区域 (创建新Track ID)
- 输出: 3D点(前一帧坐标系) + 2D点(当前帧) → 供PnP使用
"""

import numpy as np
import cv2


class TrackManager:
    """管理跨帧特征点跟踪, 维护Track ID一致性."""

    def __init__(self, min_track_length=2, proximity_thresh=10.0,
                 max_track_age=30, consolidate_min_len=3):
        self.tracks = {}       # track_id -> dict
        self.next_id = 0
        self.frame_idx = -1
        self.min_track_length = min_track_length
        self.proximity_thresh = proximity_thresh  # px, 新建Track与已有Track最小距离
        self.max_track_age = max_track_age        # 帧, 未更新Track存活时限
        self.consolidate_min_len = consolidate_min_len  # 最小track长度启用Z中值滤波
        self.active_ids = []

        self.lk_win_size = (21, 21)
        self.lk_max_level = 3

    # ═══════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════

    def initialize_frame0(self, pts_2d, depth_map, K, frame_idx=0):
        """从第一帧点集初始化所有Tracks."""
        self.frame_idx = frame_idx
        pts_3d = self._reproject_batch(pts_2d, depth_map, K)
        new_ids = []
        for i in range(len(pts_2d)):
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {
                'positions_2d': [(frame_idx, pts_2d[i].copy())],
                'positions_3d': [pts_3d[i].copy()],
                'active': True,
                'last_seen': frame_idx,
                'length': 1,
            }
            new_ids.append(tid)
        self.active_ids = new_ids
        return new_ids

    def step_frame(self, img0_path, img1_path, loftr_k0, loftr_k1,
                   depth_map_prev, depth_map_curr, K,
                   lk_quality_thresh=5.0):
        """推进一帧: LK跟踪已有Tracks + LoFTR补充新Tracks.

        Returns dict:
            pts3d_prev   — 3D点在前一帧坐标系 (供PnP)
            pts2d_curr   — 2D观测在当前帧
            track_ids    — 每个点对应的Track ID
            track_lengths — 每个Track的存活帧数
        """
        prev_frame = self.frame_idx
        self.frame_idx += 1
        curr_frame = self.frame_idx

        # 1) LK跟踪已有活跃Tracks
        self._track_existing_tracks(img0_path, img1_path, curr_frame,
                                     depth_map_curr, K, lk_quality_thresh)

        # 2) LoFTR新匹配 → 新建Tracks
        self._add_loftr_tracks(loftr_k0, loftr_k1, prev_frame, curr_frame,
                                depth_map_prev, depth_map_curr, K)

        # 3) 清理过期Tracks
        self._prune_old_tracks(curr_frame)

        # 4) 返回PnP输入: 3D=前一帧坐标系, 2D=当前帧观测
        return self._build_pnp_input(depth_map_prev, K, curr_frame)

    def get_stats(self):
        """Track统计."""
        if not self.tracks:
            return {'total': 0, 'active': 0, 'avg_length': 0, 'max_length': 0}
        lengths = [t['length'] for t in self.tracks.values()]
        return {
            'total': len(self.tracks),
            'active': len(self.active_ids),
            'avg_length': float(np.mean(lengths)),
            'max_length': int(max(lengths)),
        }

    # ═══════════════════════════════════════════════════════
    # Internal: Track lifecycle
    # ═══════════════════════════════════════════════════════

    def _track_existing_tracks(self, img0_path, img1_path, curr_frame,
                                depth_map, K, quality_thresh):
        """LK跟踪活跃Tracks从前一帧→当前帧."""
        if not self.active_ids:
            return

        prev_pts = []
        valid_ids = []
        for tid in self.active_ids:
            t = self.tracks[tid]
            _, pt = t['positions_2d'][-1]
            prev_pts.append(pt)
            valid_ids.append(tid)
        prev_pts = np.array(prev_pts, dtype=np.float32)

        tracked_pts, status = self._lk_forward_backward(
            img0_path, img1_path, prev_pts, quality_thresh)

        new_active = []
        for i, tid in enumerate(valid_ids):
            if status[i]:
                u, v = tracked_pts[i]
                pt_3d = self._reproject_single(u, v, depth_map, K)
                if pt_3d[2] <= 1e-6:
                    pt_3d = self.tracks[tid]['positions_3d'][-1].copy()

                self.tracks[tid]['positions_2d'].append((curr_frame, tracked_pts[i].copy()))
                self.tracks[tid]['positions_3d'].append(pt_3d)
                self.tracks[tid]['last_seen'] = curr_frame
                self.tracks[tid]['length'] += 1
                new_active.append(tid)
            else:
                self.tracks[tid]['active'] = False

        self.active_ids = new_active

    def _add_loftr_tracks(self, loftr_k0, loftr_k1, prev_frame, curr_frame,
                           depth_map_prev, depth_map_curr, K):
        """LoFTR新匹配→新建Tracks, 过滤与已有Track重叠的."""
        n_new = len(loftr_k0)
        if n_new == 0:
            return

        existing_pts = []
        if self.active_ids:
            for tid in self.active_ids:
                t = self.tracks[tid]
                if len(t['positions_2d']) >= 2:
                    _, pt = t['positions_2d'][-2]  # prev frame position
                    existing_pts.append(pt)
        existing_pts = np.array(existing_pts) if existing_pts else np.empty((0, 2))

        for i in range(n_new):
            if len(existing_pts) > 0:
                dists = np.linalg.norm(existing_pts - loftr_k0[i], axis=1)
                if dists.min() < self.proximity_thresh:
                    continue

            tid = self.next_id
            self.next_id += 1

            pt_3d_prev = self._reproject_single(
                loftr_k0[i, 0], loftr_k0[i, 1], depth_map_prev, K)
            pt_3d_curr = self._reproject_single(
                loftr_k1[i, 0], loftr_k1[i, 1], depth_map_curr, K)

            self.tracks[tid] = {
                'positions_2d': [
                    (prev_frame, loftr_k0[i].copy()),
                    (curr_frame, loftr_k1[i].copy()),
                ],
                'positions_3d': [pt_3d_prev, pt_3d_curr],
                'active': True,
                'last_seen': curr_frame,
                'length': 2,
            }
            self.active_ids.append(tid)

    def _prune_old_tracks(self, curr_frame):
        """清理过期Tracks."""
        to_prune = [tid for tid, t in self.tracks.items()
                    if not t['active'] and (curr_frame - t['last_seen']) > self.max_track_age]
        for tid in to_prune:
            del self.tracks[tid]

    # ═══════════════════════════════════════════════════════
    # Internal: PnP input
    # ═══════════════════════════════════════════════════════

    def _build_pnp_input(self, depth_map_prev, K, curr_frame):
        """构建PnP输入: 3D点(前一帧坐标系) + 2D点(当前帧).
        
        对长Track使用Z中值滤波: 取track历史所有depth Z的中位数,
        用当前帧2D位置 + median_Z → 3D, 消除单帧深度噪声.
        """
        pts3d_prev = []
        pts2d_curr = []
        track_ids = []
        track_lengths = []

        for tid in self.active_ids:
            t = self.tracks[tid]
            if len(t['positions_2d']) < 2:
                continue

            # 前一帧2D
            _, pt_prev = t['positions_2d'][-2]

            # Z值: 长Track用中位数滤波, 短Track用原始测量
            if t['length'] >= self.consolidate_min_len and len(t['positions_3d']) >= 3:
                Z_values = [p3d[2] for p3d in t['positions_3d'] if p3d[2] > 1e-6]
                if len(Z_values) >= 3:
                    Z = float(np.median(Z_values))
                else:
                    Z = depth_map_prev[int(np.clip(pt_prev[1], 0, depth_map_prev.shape[0]-1)),
                                      int(np.clip(pt_prev[0], 0, depth_map_prev.shape[1]-1))]
            else:
                Z = depth_map_prev[int(np.clip(pt_prev[1], 0, depth_map_prev.shape[0]-1)),
                                  int(np.clip(pt_prev[0], 0, depth_map_prev.shape[1]-1))]

            if Z <= 1e-6:
                continue

            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            X = (pt_prev[0] - cx) * Z / fx
            Y = (pt_prev[1] - cy) * Z / fy
            pt_3d = np.array([X, Y, Z], dtype=np.float32)

            # 当前帧2D
            _, pt_curr = t['positions_2d'][-1]

            pts3d_prev.append(pt_3d)
            pts2d_curr.append(pt_curr)
            track_ids.append(tid)
            track_lengths.append(t['length'])

        if not pts3d_prev:
            return {
                'pts3d_prev': np.empty((0, 3), dtype=np.float32),
                'pts2d_curr': np.empty((0, 2), dtype=np.float32),
                'track_ids': [],
                'track_lengths': np.array([], dtype=int),
            }

        return {
            'pts3d_prev': np.array(pts3d_prev, dtype=np.float32),
            'pts2d_curr': np.array(pts2d_curr, dtype=np.float32),
            'track_ids': track_ids,
            'track_lengths': np.array(track_lengths, dtype=int),
        }

    # ═══════════════════════════════════════════════════════
    # Utilities
    # ═══════════════════════════════════════════════════════

    def _lk_forward_backward(self, img0_path, img1_path, pts0, fb_thresh):
        """LK前向+后向验证."""
        img0 = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        if img0 is None or img1 is None:
            return np.zeros_like(pts0), np.zeros(len(pts0), dtype=bool)

        pts1, st_fwd, _ = cv2.calcOpticalFlowPyrLK(
            img0, img1, pts0.reshape(-1, 1, 2), None,
            winSize=self.lk_win_size, maxLevel=self.lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        if pts1 is None:
            return np.zeros_like(pts0), np.zeros(len(pts0), dtype=bool)

        valid_fwd = st_fwd.ravel().astype(bool)

        pts0_back, st_back, _ = cv2.calcOpticalFlowPyrLK(
            img1, img0, pts1[valid_fwd].reshape(-1, 1, 2), None,
            winSize=self.lk_win_size, maxLevel=self.lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        if pts0_back is None:
            return np.zeros_like(pts0), np.zeros(len(pts0), dtype=bool)

        valid_back = st_back.ravel().astype(bool)

        fb_error = np.full(len(pts0), np.inf)
        fwd_idx = np.where(valid_fwd)[0]
        back_ok = fwd_idx[valid_back]
        fb_error[back_ok] = np.linalg.norm(
            pts0[back_ok] - pts0_back[valid_back].reshape(-1, 2), axis=1)

        final_valid = fb_error < fb_thresh
        tracked = pts0.copy()
        tracked[final_valid] = pts1[final_valid].reshape(-1, 2)
        return tracked, final_valid

    def _reproject_single(self, u, v, depth_map, K):
        """像素→3D (mm)."""
        h, w = depth_map.shape
        ui = np.clip(int(round(u)), 0, w - 1)
        vi = np.clip(int(round(v)), 0, h - 1)
        Z = depth_map[vi, ui]
        if Z <= 0 or not np.isfinite(Z):
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        return np.array([(u - cx) * Z / fx, (v - cy) * Z / fy, Z], dtype=np.float32)

    def _reproject_batch(self, pts_2d, depth_map, K):
        """批量像素→3D."""
        h, w = depth_map.shape
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        pts_3d = np.zeros((len(pts_2d), 3), dtype=np.float32)
        for i in range(len(pts_2d)):
            u, v = pts_2d[i]
            ui = np.clip(int(round(u)), 0, w - 1)
            vi = np.clip(int(round(v)), 0, h - 1)
            Z = depth_map[vi, ui]
            if Z > 0 and np.isfinite(Z):
                pts_3d[i] = [(u - cx) * Z / fx, (v - cy) * Z / fy, Z]
        return pts_3d
