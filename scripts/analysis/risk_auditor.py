"""Risk Auditor — Module B of AGSA.

跨规划器安全风险评估 API。

设计:
    输入: 一个 (scene, attack) 组合 + 一个 src planner (有 gene 数据)
    输出: 其他 2 个 planner 的 fail 风险评分

机制:
    - 用 src planner 的 37-dim gene (含 strength) 作输入
    - 喂给 6 个跨 planner XGBoost (CNN↔DINO, CNN↔TF, DINO↔TF)
    - 返 0-1 风险分数

用法:
    # 单次 audit
    python risk_auditor.py --scene 8462feefcf135a4a --attack rain --planner CNN

    # 作为 library
    from risk_auditor import RiskAuditor
    auditor = RiskAuditor.load("exp/tierB_partial/risk_auditor")
    result = auditor.audit_scene(scene_token, attack, "CNN")

artifacts:
    exp/tierB_partial/risk_auditor/
        cnn_to_dino.pkl    # 6 个 XGBoost 模型
        cnn_to_tf.pkl
        dino_to_cnn.pkl
        dino_to_tf.pkl
        tf_to_cnn.pkl
        tf_to_dino.pkl
        metadata.json      # AUC, n_train, feature list
        demo_output.txt    # demo 运行结果
"""
from __future__ import annotations
import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


GENE_FIELDS = [
    "low_freq_ratio", "mid_freq_ratio", "high_freq_ratio", "spectral_centroid",
    "hue_mean", "hue_std", "sat_mean", "sat_std", "val_mean", "val_std", "colorfulness",
    "lbp_entropy", "glcm_contrast", "lbp_uniformity",
    "rms_contrast", "mean_luma", "std_luma", "dynamic_range", "mean_shift", "std_shift",
    "edge_mean", "edge_density", "road_luma_mean", "road_luma_std",
    "lane_line_count", "lane_line_density",
    "vehicle_loss", "person_loss", "detection_loss", "conf_loss", "vehicle_loss_ratio",
    "shadow_ratio", "highlight_ratio", "luma_entropy", "luma_skew", "luma_mean",
]
META_FIELDS = ["strength"]
PLANNERS = ["CNN", "DINO", "TF"]


def train_auditor(csv_path: str, output_dir: Path) -> Dict:
    """训练 6 个 src→tgt 跨 planner XGBoost 模型并保存。"""
    import xgboost as xgb
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score

    print(f"=== Training Risk Auditor (6 cross-planner XGBoost) ===")
    df = pd.read_csv(csv_path)
    feat = [c for c in GENE_FIELDS + META_FIELDS if c in df.columns]
    print(f"  features: {len(feat)}")

    # group by (scene, attack, strength) for src→tgt alignment
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"features": feat, "models": {}}

    for src in PLANNERS:
        for tgt in PLANNERS:
            if src == tgt:
                continue
            pair_name = f"{src.lower()}_to_{tgt.lower()}"
            print(f"\n  training {pair_name}...")

            # Build src and tgt DataFrames, align on (scene_token, attack, strength)
            df_src = df[df["planner"] == src].copy()
            df_tgt = df[df["planner"] == tgt].copy()
            key_cols = ["scene_token", "attack", "strength"]
            # feat 也含 'strength' (作为输入特征), 用 unique 合并防列重名
            feat_unique = list(dict.fromkeys(key_cols + feat))
            merged = df_src[feat_unique].merge(
                df_tgt[key_cols + ["success"]].rename(columns={"success": "y_tgt"}),
                on=key_cols, how="inner",
            )
            Xs = merged[feat].values.astype(np.float32)
            Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
            ys = merged["y_tgt"].values.astype(np.int32)
            ss = merged["scene_token"].values

            # 5-fold GroupKFold 训练，取 mean AUC
            gkf = GroupKFold(n_splits=5)
            aucs, models = [], []
            for tr, te in gkf.split(Xs, ys, ss):
                m = xgb.XGBClassifier(
                    n_estimators=300, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=42, n_jobs=-1, eval_metric="logloss",
                )
                m.fit(Xs[tr], ys[tr])
                try:
                    aucs.append(roc_auc_score(ys[te], m.predict_proba(Xs[te])[:, 1]))
                except ValueError:
                    pass
                models.append(m)
            best_i = max(range(len(aucs)), key=lambda i: aucs[i])
            best_model = models[best_i]
            mean_auc = float(np.mean(aucs))
            print(f"    per-fold AUC: {[f'{a:.3f}' for a in aucs]}  mean={mean_auc:.3f}")

            # save
            pkl_path = output_dir / f"{pair_name}.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump({"model": best_model, "feat": feat,
                             "auc": mean_auc, "n_samples": int(len(merged))}, f)
            metadata["models"][pair_name] = {
                "auc": mean_auc,
                "n_samples": int(len(merged)),
                "n_folds": 5,
            }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata


class RiskAuditor:
    """Risk Auditor — Module B 公共 API。

    用法:
        auditor = RiskAuditor.load("exp/tierB_partial/risk_auditor")
        result = auditor.audit_gene(gene_vec_37, src_planner="CNN")
        # result = {"DINO": 0.62, "TF": 0.41, "CNN_reference": ...}

        result = auditor.audit_scene("8462fee...", "rain", "CNN", csv_path)
    """

    def __init__(self, model_dir: Path, metadata: Dict):
        self.model_dir = Path(model_dir)
        self.metadata = metadata
        self.models: Dict[Tuple[str, str], object] = {}
        for pair_name, m_info in metadata["models"].items():
            with open(self.model_dir / f"{pair_name}.pkl", "rb") as f:
                blob = pickle.load(f)
            src, tgt = pair_name.split("_to_")
            self.models[(src.upper(), tgt.upper())] = blob["model"]
        self.feat = metadata["features"]

    @classmethod
    def load(cls, model_dir: str) -> "RiskAuditor":
        p = Path(model_dir)
        with open(p / "metadata.json", "r") as f:
            metadata = json.load(f)
        return cls(p, metadata)

    def audit_gene(self, gene_vec: np.ndarray, src_planner: str) -> Dict[str, float]:
        """对单帧 gene vector (37-dim + strength = 38 features), 给所有 target planner 风险评分。

        返回:
            {"CNN_reference": 0.84, "DINO": 0.62, "TF": 0.41}
        其中 CNN_reference 是 src planner 的 self-consistency 预测
        (用同源 src→src 训练, 但若 src 已知, 应当跟 ground truth 对齐)
        """
        gv = np.asarray(gene_vec, dtype=np.float32).reshape(1, -1)
        if gv.shape[1] != len(self.feat):
            raise ValueError(f"expected {len(self.feat)} features, got {gv.shape[1]}")
        gv = np.nan_to_num(gv, nan=0.0, posinf=0.0, neginf=0.0)

        out = {}
        for tgt in PLANNERS:
            if (src_planner, tgt) not in self.models:
                # 没有 src→tgt model, 跳过
                continue
            model = self.models[(src_planner, tgt)]
            risk = float(model.predict_proba(gv)[:, 1][0])
            if tgt == src_planner:
                out[f"{tgt}_reference"] = risk
            else:
                out[tgt] = risk
        return out

    def audit_scene(self, scene_token: str, attack: str, src_planner: str,
                    csv_path: str) -> Optional[Dict]:
        """从 CSV 查 (scene, attack) 任意 strength 的 src gene, 然后 audit。

        返回 dict:
            {
                "scene_token": ...,
                "attack": ...,
                "src_planner": ...,
                "ground_truth": {"DINO": ..., "TF": ...},  # if available
                "predicted_risk": {"DINO": ..., "TF": ...},
                "agreement": {...},  # predicted vs ground truth
            }
        """
        df = pd.read_csv(csv_path)
        sub_src = df[(df["scene_token"] == scene_token) &
                     (df["attack"] == attack) &
                     (df["planner"] == src_planner)]
        if sub_src.empty:
            return None
        # 取 strength 最大的那行 (代表"worst case")
        sub_src = sub_src.sort_values("strength", ascending=False).iloc[0]
        gv = sub_src[self.feat].values.astype(np.float32)

        # predicted risk
        pred_risk = self.audit_gene(gv, src_planner)

        # ground truth (同 scene+attack+strength 的其他 planner 真实 success)
        gt = {}
        for tgt in PLANNERS:
            if tgt == src_planner:
                continue
            sub_tgt = df[(df["scene_token"] == scene_token) &
                         (df["attack"] == attack) &
                         (df["strength"] == sub_src["strength"]) &
                         (df["planner"] == tgt)]
            if not sub_tgt.empty:
                gt[tgt] = int(sub_tgt.iloc[0]["success"])
        return {
            "scene_token": scene_token,
            "attack": attack,
            "src_planner": src_planner,
            "strength": float(sub_src["strength"]),
            "predicted_risk": pred_risk,
            "ground_truth_fail": gt,
        }

    def audit_batch(self, df: pd.DataFrame, src_planner: str,
                    n_samples: int = 50) -> pd.DataFrame:
        """批量 audit 多个 (scene, attack)。返回每个 case 的预测 vs 真实对比。"""
        rows = []
        # group by (scene, attack), 取最坏 strength
        for (scene, atk), g in df[df["planner"] == src_planner].groupby(
                ["scene_token", "attack"]):
            g = g.sort_values("strength", ascending=False).iloc[0]
            gv = g[self.feat].values.astype(np.float32)
            pred = self.audit_gene(gv, src_planner)
            row = {"scene_token": scene, "attack": atk, "src_planner": src_planner,
                   "src_strength": float(g["strength"])}
            for tgt in PLANNERS:
                if tgt != src_planner and tgt in pred:
                    row[f"pred_{tgt}_risk"] = pred[tgt]
                    # ground truth: 取同 (scene, attack, strength) 的 tgt 行
                    tgt_row = df[(df["scene_token"] == scene) &
                                 (df["attack"] == atk) &
                                 (df["strength"] == g["strength"]) &
                                 (df["planner"] == tgt)]
                    row[f"gt_{tgt}_fail"] = int(tgt_row.iloc[0]["success"]) if not tgt_row.empty else None
            rows.append(row)
        out = pd.DataFrame(rows)
        return out.head(n_samples)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="exp/tierB_partial/merged_3pl.csv")
    p.add_argument("--output-dir", default="exp/tierB_partial/risk_auditor")
    p.add_argument("--mode", choices=["train", "audit", "demo"],
                   default="train", help="train=训 6 模型; audit=查单个; demo=批量演示")
    p.add_argument("--scene", help="audit mode: scene_token")
    p.add_argument("--attack", help="audit mode: attack name")
    p.add_argument("--planner", default="CNN", help="src planner (CNN/DINO/TF)")
    args = p.parse_args()

    out_dir = Path(args.output_dir)

    if args.mode == "train":
        t0 = time.time()
        metadata = train_auditor(args.csv, out_dir)
        print(f"\n  training done in {time.time() - t0:.1f}s")
        print(f"  saved 6 models + metadata.json in {out_dir}/")
        return

    # load auditor
    auditor = RiskAuditor.load(out_dir)

    if args.mode == "audit":
        if not args.scene or not args.attack:
            print("  audit mode needs --scene and --attack")
            return
        result = auditor.audit_scene(args.scene, args.attack, args.planner, args.csv)
        if result is None:
            print(f"  scene={args.scene} attack={args.attack} not found")
            return
        print(f"\n=== Risk Audit ===")
        print(f"  scene_token : {result['scene_token']}")
        print(f"  attack      : {result['attack']}")
        print(f"  src planner : {result['src_planner']}")
        print(f"  strength    : {result['strength']:.2f}")
        print(f"\n  --- Predicted fail risk (1=likely fail) ---")
        for k, v in result["predicted_risk"].items():
            print(f"    {k:<20s}: {v:.3f}")
        if result["ground_truth_fail"]:
            print(f"\n  --- Ground truth fail (1=actual fail) ---")
            for k, v in result["ground_truth_fail"].items():
                print(f"    {k:<20s}: {v}")
        # 一句话风险解读
        pred = result["predicted_risk"]
        src_ref = pred.get(f"{args.planner}_reference", None)
        others = {k: v for k, v in pred.items() if not k.endswith("_reference")}
        if others:
            max_tgt = max(others, key=others.get)
            max_v = others[max_tgt]
            if max_v > 0.7:
                verdict = "HIGH RISK: 跨规划器都容易失败"
            elif max_v > 0.4:
                verdict = f"MODERATE: {max_tgt} 风险最高 ({max_v:.2f})"
            else:
                verdict = f"LOW: 整体可控, {max_tgt} 风险最高 ({max_v:.2f})"
            print(f"\n  Verdict: {verdict}")

    elif args.mode == "demo":
        df = pd.read_csv(args.csv)
        print(f"\n=== Risk Auditor Demo: 批量 audit 10 个 case ===")
        out = auditor.audit_batch(df, args.planner, n_samples=10)
        print(out.to_string(index=False))
        # 算 AUC (pred vs gt) for demo
        if "pred_DINO_risk" in out.columns and "gt_DINO_fail" in out.columns:
            from sklearn.metrics import roc_auc_score
            valid = out.dropna(subset=["pred_DINO_risk", "gt_DINO_fail"])
            if len(valid) > 5:
                try:
                    auc = roc_auc_score(valid["gt_DINO_fail"], valid["pred_DINO_risk"])
                    print(f"\n  batch AUC (src={args.planner} → DINO): {auc:.3f}  on n={len(valid)}")
                except ValueError as e:
                    print(f"\n  [WARN] AUC skip: {e}")


if __name__ == "__main__":
    main()
