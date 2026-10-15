"""
SIFT Landmark Recognition & Verification Pipeline
Author: Afnan Khan (@Afnan-makstrix)
Course: Computer Vision - Lab Assignment 06
"""

import os
import glob
import time
import json
import csv
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class LandmarkRecognitionPipeline:
    def __init__(self, ratio_thresh=0.75, min_matches=20, min_inliers=12):
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher(cv2.NORM_L2)
        self.ratio_thresh = ratio_thresh
        self.min_matches = min_matches
        self.min_inliers = min_inliers
        self.database = {}

    def extract_sift(self, bgr_img):
        """Task 2 & 3: Grayscale conversion and SIFT feature extraction."""
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY) if len(bgr_img.shape) == 3 else bgr_img
        kps, descs = self.sift.detectAndCompute(gray, None)
        return gray, kps, descs

    def build_database(self, data_root):
        """Task 1: Load landmark reference images and extract features."""
        self.database = {}
        folders = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and d != 'unknown_landmark']
        
        for folder in sorted(folders):
            folder_path = os.path.join(data_root, folder)
            files = sorted(glob.glob(os.path.join(folder_path, "*.jpg")) + glob.glob(os.path.join(folder_path, "*.png")))
            self.database[folder] = []
            for f in files:
                img = cv2.imread(f)
                if img is None:
                    continue
                gray, kps, des = self.extract_sift(img)
                self.database[folder].append({
                    'name': folder,
                    'file': os.path.basename(f),
                    'path': f,
                    'bgr': img,
                    'gray': gray,
                    'kps': kps,
                    'des': des
                })
        return self.database

    def match_features(self, des1, des2):
        """Task 4: KNN matching (k=2) with Lowe's ratio test."""
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return []
        knn = self.bf.knnMatch(des1, des2, k=2)
        good = []
        for pair in knn:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.ratio_thresh * n.distance:
                    good.append(m)
        return good

    def recognize(self, query_bgr):
        """Task 4: Match query against database, rank candidates, apply threshold rejection."""
        gray_q, kps_q, des_q = self.extract_sift(query_bgr)
        scores = []

        for lm_name, ref_list in self.database.items():
            best_matches = []
            best_ref = None
            for ref in ref_list:
                matches = self.match_features(des_q, ref['des'])
                if len(matches) > len(best_matches):
                    best_matches = matches
                    best_ref = ref

            scores.append({
                'landmark': lm_name,
                'match_count': len(best_matches),
                'matches': best_matches,
                'ref': best_ref
            })

        scores.sort(key=lambda x: x['match_count'], reverse=True)
        top = scores[0] if scores else None
        second = scores[1] if len(scores) > 1 else None

        # Rejection decision gate
        is_recognized = False
        decision_msg = ""

        if top is None or top['match_count'] < self.min_matches:
            is_recognized = False
            decision_msg = f"Rejected: Match count ({top['match_count'] if top else 0}) below threshold {self.min_matches}"
        elif second and second['match_count'] > 0 and (top['match_count'] / second['match_count'] < 1.35) and top['match_count'] < 35:
            is_recognized = False
            decision_msg = f"Rejected: Ambiguous match between {top['landmark']} and {second['landmark']}"
        else:
            is_recognized = True
            decision_msg = f"Recognized: {top['landmark']} ({top['match_count']} reliable matches)"

        return {
            'is_recognized': is_recognized,
            'decision_message': decision_msg,
            'top_landmark': top['landmark'] if is_recognized else 'Unknown / Rejected',
            'top_score': top,
            'scores': scores,
            'query_kps': kps_q,
            'query_des': des_q
        }

    def localize(self, query_bgr, kps_q, ref_bgr, kps_ref, matches):
        """Task 5: RANSAC Homography estimation and bounding polygon projection."""
        h_q, w_q = query_bgr.shape[:2]
        h_r, w_r = ref_bgr.shape[:2]
        annotated = query_bgr.copy()

        if len(matches) < 4:
            return {'success': False, 'inliers': 0, 'image': annotated, 'msg': 'Matches < 4'}

        src_pts = np.float32([kps_ref[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kps_q[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return {'success': False, 'inliers': 0, 'image': annotated, 'msg': 'Homography failed'}

        inliers = int(mask.sum())
        if inliers < self.min_inliers or (inliers / len(matches)) < 0.20:
            self._draw_status(annotated, f"GEOMETRY UNRELIABLE: Inliers {inliers} < {self.min_inliers}", (0, 0, 255))
            return {'success': False, 'inliers': inliers, 'image': annotated, 'msg': 'Insufficient inliers'}

        det = np.linalg.det(H[:2, :2])
        if det <= 0:
            self._draw_status(annotated, "GEOMETRY UNRELIABLE: Negative Determinant", (0, 0, 255))
            return {'success': False, 'inliers': inliers, 'image': annotated, 'msg': 'Reflective transform'}

        corners = np.float32([[0, 0], [w_r, 0], [w_r, h_r], [0, h_r]]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, H)

        if not cv2.isContourConvex(np.int32(projected)):
            self._draw_status(annotated, "GEOMETRY UNRELIABLE: Non-Convex Polygon", (0, 0, 255))
            return {'success': False, 'inliers': inliers, 'image': annotated, 'msg': 'Non-convex polygon'}

        poly_area = cv2.contourArea(np.int32(projected))
        if poly_area < 0.02 * (w_q * h_q) or poly_area > 3.0 * (w_q * h_q):
            self._draw_status(annotated, "GEOMETRY UNRELIABLE: Invalid Area", (0, 0, 255))
            return {'success': False, 'inliers': inliers, 'image': annotated, 'msg': 'Degenerate area'}

        # Draw green bounding polygon
        pts_int = np.int32(projected)
        cv2.polylines(annotated, [pts_int], True, (0, 255, 128), 3, cv2.LINE_AA)
        for p in pts_int:
            cv2.circle(annotated, tuple(p[0]), 5, (0, 255, 255), -1)

        self._draw_status(annotated, f"LOCALIZED: {inliers}/{len(matches)} inliers", (0, 255, 128))
        return {'success': True, 'inliers': inliers, 'image': annotated, 'msg': 'Success', 'corners': projected}

    def _draw_status(self, img, text, color):
        h, w = img.shape[:2]
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 42), (10, 25, 20), -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
        cv2.rectangle(img, (0, 0), (w, 42), color, 2)
        cv2.putText(img, text, (15, 28), cv2.FONT_HERSHEY_DUPLEX, 0.65, color, 2)

    def run_benchmarks(self):
        """Task 6: Performance evaluation across 6 distinct conditions."""
        print("\n=== Running Performance Benchmark Suite (Afnan Khan) ===")
        eiffel_ref = self.database['eiffel_tower'][0]['bgr']
        eiffel_view3 = self.database['eiffel_tower'][2]['bgr']
        big_ben = cv2.imread("data/landmarks/unknown_landmark/big_ben_1.jpg")

        h, w = eiffel_ref.shape[:2]

        q_orig = eiffel_ref.copy()
        M = cv2.getRotationMatrix2D((w/2, h/2), 45, 0.85)
        q_rot = cv2.warpAffine(eiffel_ref, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        q_scale = cv2.resize(eiffel_ref, (w//2, h//2), interpolation=cv2.INTER_AREA)
        q_illum = np.clip(eiffel_ref.astype(np.float32) * 0.35, 0, 255).astype(np.uint8)
        q_view = eiffel_view3.copy()
        q_unknown = big_ben.copy()

        tests = [
            ("Original Image", q_orig, "eiffel_tower"),
            ("Rotated Image (45 deg)", q_rot, "eiffel_tower"),
            ("Resized Image (0.5x Scale)", q_scale, "eiffel_tower"),
            ("Reduced Brightness (0.35x)", q_illum, "eiffel_tower"),
            ("Different Viewpoint", q_view, "eiffel_tower"),
            ("Non-Database Landmark", q_unknown, "unknown")
        ]

        records = []
        gallery_items = []
        matches_vis = []

        print(f"{'Condition':<26} | {'Ground Truth':<14} | {'Predicted':<18} | {'Matches':<8} | {'Inliers':<8} | {'Decision':<10} | {'Time (ms)':<9}")
        print("-" * 105)

        for name, q_img, gt in tests:
            t0 = time.time()
            rec = self.recognize(q_img)
            top = rec['top_score']
            is_rec = rec['is_recognized']

            if is_rec and top['ref']:
                loc = self.localize(q_img, rec['query_kps'], top['ref']['bgr'], top['ref']['kps'], top['matches'])
                inliers = loc['inliers']
                ann_img = loc['image']
                dec = "RECOGNIZED"
            else:
                inliers = 0
                ann_img = q_img.copy()
                self._draw_status(ann_img, "REJECTED [UNKNOWN / NON-DATABASE]", (0, 0, 255))
                dec = "REJECTED"

            dt_ms = (time.time() - t0) * 1000
            n_matches = top['match_count'] if top else 0

            records.append({
                'Condition': name,
                'Ground_Truth': gt,
                'Predicted_Landmark': rec['top_landmark'],
                'Query_Keypoints': len(rec['query_kps']),
                'Reliable_Matches': n_matches,
                'RANSAC_Inliers': inliers,
                'Decision': dec,
                'Processing_Time_ms': round(dt_ms, 2)
            })

            gallery_items.append((name, ann_img, dec, n_matches, inliers))

            if name in ["Rotated Image (45 deg)", "Non-Database Landmark"] and top and top['ref']:
                m_vis = cv2.drawMatches(
                    q_img, rec['query_kps'],
                    top['ref']['bgr'], top['ref']['kps'],
                    top['matches'][:50], None,
                    matchColor=(0, 255, 128) if is_rec else (0, 0, 255),
                    singlePointColor=(180, 180, 180),
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
                matches_vis.append((name, m_vis, is_rec, n_matches))

            print(f"{name:<26} | {gt:<14} | {rec['top_landmark']:<18} | {n_matches:<8} | {inliers:<8} | {dec:<10} | {dt_ms:<9.1f}")

        print("-" * 105)

        # Plot 1: Database overview
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        fig.patch.set_facecolor('#064e3b')
        for ax, (lm, refs) in zip(axes, sorted(self.database.items())):
            ax.set_facecolor('#022c22')
            rgb = cv2.cvtColor(refs[0]['bgr'], cv2.COLOR_BGR2RGB)
            ax.imshow(rgb)
            ax.set_title(f"{lm.replace('_', ' ').title()}\n({len(refs)} Views, {len(refs[0]['kps'])} KPs)",
                         color='#34d399', fontsize=11, fontweight='bold', pad=8)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig("figures/task1_landmark_database.png", dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close()

        # Plot 2: Keypoints
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        fig.patch.set_facecolor('#064e3b')
        for ax, (lm, refs) in zip(axes, sorted(self.database.items())):
            ax.set_facecolor('#022c22')
            ref = refs[0]
            kp_vis = cv2.drawKeypoints(ref['bgr'], ref['kps'][:400], None, color=(0, 255, 128),
                                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            ax.imshow(cv2.cvtColor(kp_vis, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{lm.replace('_', ' ').title()}\n{len(ref['kps'])} SIFT Features",
                         color='#a7f3d0', fontsize=11, fontweight='bold', pad=8)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig("figures/task2_sift_keypoints.png", dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close()

        # Plot 3: Feature Matches
        fig, axes = plt.subplots(len(matches_vis), 1, figsize=(15, 5.5 * len(matches_vis)))
        if len(matches_vis) == 1:
            axes = [axes]
        fig.patch.set_facecolor('#064e3b')
        for ax, (c_name, m_bgr, is_rec, n_m) in zip(axes, matches_vis):
            ax.set_facecolor('#022c22')
            rgb = cv2.cvtColor(m_bgr, cv2.COLOR_BGR2RGB)
            ax.imshow(rgb)
            col = '#34d399' if is_rec else '#f87171'
            txt = f"[RECOGNIZED: {n_m} reliable matches]" if is_rec else f"[REJECTED: {n_m} spurious matches]"
            ax.set_title(f"SIFT Match Pair: {c_name} -> {txt}", color=col, fontsize=12, fontweight='bold', pad=10)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig("figures/task4_feature_matching.png", dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close()

        # Plot 4: Robustness Matrix
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.patch.set_facecolor('#064e3b')
        for ax, (c_name, ann_img, dec, n_m, n_in) in zip(axes.flatten(), gallery_items):
            ax.set_facecolor('#022c22')
            rgb = cv2.cvtColor(ann_img, cv2.COLOR_BGR2RGB)
            ax.imshow(rgb)
            col = '#34d399' if dec == 'RECOGNIZED' else '#f87171'
            ax.set_title(f"{c_name}\nStatus: {dec} | Matches: {n_m} | Inliers: {n_in}",
                         color=col, fontsize=11, fontweight='bold', pad=8)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig("figures/task6_robustness_evaluation.png", dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close()

        # Plot 5: Summary metrics
        conds = [r['Condition'] for r in records]
        matches = [r['Reliable_Matches'] for r in records]
        inliers = [r['RANSAC_Inliers'] for r in records]
        times = [r['Processing_Time_ms'] for r in records]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
        fig.patch.set_facecolor('#064e3b')

        x = np.arange(len(conds))
        width = 0.35
        ax1.set_facecolor('#022c22')
        ax1.bar(x - width/2, matches, width, label='Reliable Matches', color='#34d399')
        ax1.bar(x + width/2, inliers, width, label='RANSAC Inliers', color='#6ee7b7')
        ax1.set_title('Matches & Inliers per Condition', color='#ecfdf5', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(conds, rotation=25, ha='right', color='#a7f3d0', fontsize=9)
        ax1.tick_params(colors='#a7f3d0')
        ax1.legend(facecolor='#064e3b', edgecolor='#047857', labelcolor='#ecfdf5')
        ax1.grid(axis='y', linestyle='--', alpha=0.3, color='#047857')

        ax2.set_facecolor('#022c22')
        bars = ax2.bar(x, times, width*1.5, color='#10b981')
        ax2.set_title('Execution Latency (ms)', color='#ecfdf5', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(conds, rotation=25, ha='right', color='#a7f3d0', fontsize=9)
        ax2.tick_params(colors='#a7f3d0')
        ax2.grid(axis='y', linestyle='--', alpha=0.3, color='#047857')
        for b in bars:
            yval = b.get_height()
            ax2.text(b.get_x() + b.get_width()/2.0, yval + 1.5, f'{yval:.1f}ms', ha='center', va='bottom', color='#ecfdf5', fontsize=8)

        plt.tight_layout()
        plt.savefig("figures/task6_performance_metrics.png", dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close()

        # Save JSON & CSV
        with open("results/evaluation_metrics.json", "w", encoding="utf-8") as f:
            json.dump(records, f, indent=4)
        with open("results/evaluation_metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)

        print("\n-> Saved all figures and benchmark results in figures/ and results/")
        return records

if __name__ == '__main__':
    pipeline = LandmarkRecognitionPipeline()
    pipeline.build_database("data/landmarks")
    pipeline.run_benchmarks()
