"""
组织运动感知PnP集成
===================
使用训练好的 MotionClassifier 对 LoFTR 匹配点做软加权,
提高 PnP 位姿估计对组织运动的鲁棒性。
"""
import cv2
import numpy as np
import torch


class MotionAwarePnP:
    """使用运动分类器加权的 PnP 求解器.

    用法:
        filter = MotionAwarePnP(model_path, device='cuda')
        R, t, weights = filter.solve_weighted_pnp(pts_3d, pts_2d, K, rgb_i, rgb_j, pts0, pts1)
    """

    def __init__(self, model_path, device='cuda', patch_size=16, n_bootstrap=5, n_samples=200):
        from motion_classifier import MotionClassifier

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.patch_size = patch_size
        self.n_bootstrap = n_bootstrap
        self.n_samples = n_samples

        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = MotionClassifier().to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    @torch.no_grad()
    def predict_weights(self, rgb_i, rgb_j, pts0, pts1):
        """Predict P(static) weight for each matched point.

        Args:
            rgb_i, rgb_j: (H, W, 3) uint8 RGB images
            pts0: (N, 2) pixel coords in frame i
            pts1: (N, 2) pixel coords in frame j

        Returns:
            weights: (N,) float32 in [0, 1], higher = more likely static
        """
        H, W = rgb_i.shape[:2]
        N = len(pts0)
        half = self.patch_size // 2

        patches_i = np.zeros((N, self.patch_size, self.patch_size, 3), dtype=np.uint8)
        patches_j = np.zeros((N, self.patch_size, self.patch_size, 3), dtype=np.uint8)

        for k in range(N):
            x_i, y_i = int(round(pts0[k, 0])), int(round(pts0[k, 1]))
            x_j, y_j = int(round(pts1[k, 0])), int(round(pts1[k, 1]))

            x0_i = max(0, x_i - half); y0_i = max(0, y_i - half)
            x1_i = min(W, x_i + half); y1_i = min(H, y_i + half)
            x0_j = max(0, x_j - half); y0_j = max(0, y_j - half)
            x1_j = min(W, x_j + half); y1_j = min(H, y_j + half)

            crop_i = rgb_i[y0_i:y1_i, x0_i:x1_i]
            crop_j = rgb_j[y0_j:y1_j, x0_j:x1_j]
            if crop_i.size > 0:
                patches_i[k] = cv2.resize(crop_i, (self.patch_size, self.patch_size))
            if crop_j.size > 0:
                patches_j[k] = cv2.resize(crop_j, (self.patch_size, self.patch_size))

        flow_dx = pts1[:, 0] - pts0[:, 0]
        flow_dy = pts1[:, 1] - pts0[:, 1]
        pos_x = pts0[:, 0] / W
        pos_y = pts0[:, 1] / H

        patch_i_t = torch.from_numpy(patches_i.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(self.device)
        patch_j_t = torch.from_numpy(patches_j.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(self.device)
        flow_t = torch.tensor(np.stack([flow_dx, flow_dy, pos_x, pos_y], axis=1),
                              dtype=torch.float32).to(self.device)

        # P(moving) → weight = 1 - P(moving) = P(static)
        p_moving = self.model.predict_proba(patch_i_t, patch_j_t, flow_t)
        weights = (1.0 - p_moving).cpu().numpy().astype(np.float32)

        return weights

    def solve_weighted_pnp(self, pts_3d, pts_2d, K, rgb_i, rgb_j, pts0, pts1):
        """Weighted PnP using bootstrap sampling.

        Args:
            pts_3d: (N, 3) 3D points
            pts_2d: (N, 2) 2D points
            K: (3, 3) camera intrinsics
            rgb_i, rgb_j: images for patch extraction
            pts0, pts1: pixel coords for patch centers

        Returns:
            R_best: (3, 3) rotation matrix
            t_best: (3, 1) translation vector
            weights: (N,) static weights
        """
        N = len(pts_3d)
        if N < 4:
            return None, None, None

        # Get weights
        weights = self.predict_weights(rgb_i, rgb_j, pts0, pts1)

        # Bootstrap: sample K times, each time selecting n_samples points
        # with probability proportional to weight
        best_R = None
        best_t = None
        best_inliers = 0
        n_select = min(self.n_samples, N)

        for _ in range(self.n_bootstrap):
            # Weighted random sampling without replacement
            prob = weights / (weights.sum() + 1e-12)
            indices = np.random.choice(N, n_select, replace=False, p=prob)

            pts_3d_s = pts_3d[indices].astype(np.float64)
            pts_2d_s = pts_2d[indices].astype(np.float64)

            ret, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_3d_s, pts_2d_s, K, None,
                iterationsCount=100,
                reprojectionError=3.0,
                confidence=0.99,
                flags=cv2.SOLVEPNP_ITERATIVE)

            if ret and inliers is not None:
                n_inl = len(inliers)
                if n_inl > best_inliers:
                    best_inliers = n_inl
                    best_R, _ = cv2.Rodrigues(rvec)
                    best_t = tvec

        if best_R is None:
            # Fallback: use all points unweighted
            ret, rvec, tvec = cv2.solvePnP(
                pts_3d.astype(np.float64), pts_2d.astype(np.float64), K, None,
                flags=cv2.SOLVEPNP_ITERATIVE)
            if ret:
                best_R, _ = cv2.Rodrigues(rvec)
                best_t = tvec

        return best_R, best_t, weights

    def solve_static_only_pnp(self, pts_3d, pts_2d, K, rgb_i, rgb_j, pts0, pts1,
                              threshold=0.5):
        """Hard filtering: only use points with P(static) > threshold."""
        weights = self.predict_weights(rgb_i, rgb_j, pts0, pts1)
        static_mask = weights > threshold

        if np.sum(static_mask) < 4:
            # Fallback to all points
            static_mask = np.ones(len(pts_3d), dtype=bool)

        pts_3d_s = pts_3d[static_mask].astype(np.float64)
        pts_2d_s = pts_2d[static_mask].astype(np.float64)

        ret, rvec, tvec = cv2.solvePnP(
            pts_3d_s, pts_2d_s, K, None,
            flags=cv2.SOLVEPNP_ITERATIVE)

        if ret:
            R, _ = cv2.Rodrigues(rvec)
            return R, tvec, weights
        return None, None, weights
