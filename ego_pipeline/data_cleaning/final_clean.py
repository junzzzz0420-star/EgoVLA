
"""Apply outlier rejection, gap filling, and camera-aware trajectory smoothing."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import numpy as np

from .cleaning_modules.block_outlier import block_hampel
from .cleaning_modules.image_lr_constraint import reject_crossed_hands
from .cleaning_modules.slerp_rot import fill_slerp_rot
from .cleaning_modules.ukf_cam_smoothing import smooth_ukf_cam


def _valid_count(pred_result) -> int:
    """Count valid hand predictions in the serialized result tuple."""
    v = pred_result[4]
    try:
        import torch
        if isinstance(v, torch.Tensor):
            v = v.cpu().numpy()
    except Exception:
        pass
    return int(np.asarray(v).astype(bool).sum())


def final_clean(
    pred_result: list,
    cam_c2w=None,
    K=None,
    pos_thresh: float = 0.15,
    rot_thresh: float = 45.0,
    max_block: int = 15,
    pos_recovery: float = 0.15,
    max_gap: int = 30,
    q: float = 0.6,
    r: float = 0.6,
    beta: float = 2.0,
    rts: float = 1.0,
    lr_margin_px: float = 0.0,
    enforce_image_lr: bool = True,
    verbose: bool = True,
) -> list:
    """Run cleanup: outlier reject → image L/R → gap fill → UKF → image L/R."""

    if verbose:
        print('[clean] rejecting short outlier blocks')
    n0 = _valid_count(pred_result)
    pred_result = block_hampel(
        pred_result,
        pos_thresh=pos_thresh,
        rot_thresh=rot_thresh,
        max_block=max_block,
        pos_recovery=pos_recovery,
    )
    if verbose:
        print(
            f'[clean] rejected={n0 - _valid_count(pred_result)} '
            f'valid_before={n0} valid_after={_valid_count(pred_result)}'
        )

    # Before fill: drop frames where left wrist is not to the left of right in image.
    if enforce_image_lr:
        if verbose:
            print('[clean] enforcing image-space left-of-right (pre-fill)')
        n_lr = _valid_count(pred_result)
        pred_result = reject_crossed_hands(
            pred_result, cam_c2w=cam_c2w, K=K, min_margin_px=lr_margin_px,
        )
        if verbose:
            print(
                f'[clean] lr_rejected={n_lr - _valid_count(pred_result)} '
                f'valid_after={_valid_count(pred_result)}'
            )

    if verbose:
        print('[clean] filling short rotation gaps with SLERP')
    n1 = _valid_count(pred_result)
    pred_result = fill_slerp_rot(pred_result, max_gap=max_gap)
    if verbose:
        print(
            f'[clean] filled={_valid_count(pred_result) - n1} '
            f'valid_before={n1} valid_after={_valid_count(pred_result)}'
        )

    if verbose:
        print('[clean] applying camera-aware UKF and RTS smoothing')
    pred_result = smooth_ukf_cam(pred_result, cam_c2w=cam_c2w,
                                 q=q, r=r, beta=beta, rts=rts)
    if verbose:
        print(f'[clean] q={q} r={r} beta={beta} rts={bool(rts)}')

    # Final gate after smoothing (UKF / fill may reintroduce mild crossings).
    if enforce_image_lr:
        if verbose:
            print('[clean] enforcing image-space left-of-right (final)')
        n_lr2 = _valid_count(pred_result)
        pred_result = reject_crossed_hands(
            pred_result, cam_c2w=cam_c2w, K=K, min_margin_px=lr_margin_px,
        )
        if verbose:
            print(
                f'[clean] lr_rejected={n_lr2 - _valid_count(pred_result)} '
                f'valid_after={_valid_count(pred_result)}'
            )

    return pred_result




def _find_latest_pth(result_dir: Path) -> Path:
    pth_files = sorted(result_dir.rglob('*.pth'), key=lambda p: p.stat().st_mtime)
    if not pth_files:
        raise FileNotFoundError(f'No prediction .pth file exists under: {result_dir}')
    return pth_files[-1]


def main() -> None:
    import joblib

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=None,
                        help='Prediction .pth file; defaults to the newest file under result/')
    parser.add_argument('--megasam', type=Path, default=None,
                        help='Optional Mega-SAM camera trajectory .npz file')
    args = parser.parse_args()

    pth_path = (args.input.resolve() if args.input is not None
                else _find_latest_pth(PROJECT_ROOT / 'result'))
    scene = pth_path.stem
    print(f'[clean] input={pth_path}')

    megasam = args.megasam if args.megasam is not None else pth_path.parent / 'megasam.npz'
    if not Path(megasam).exists():
        print(f'[clean] camera trajectory not found; smoothing without camera input: {megasam}')
        megasam = None

    pred_result = joblib.load(pth_path)
    result = final_clean(pred_result, cam_c2w=megasam)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = PROJECT_ROOT / 'output' / 'data_cleaning' / 'test' / timestamp / 'final'
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f'{scene}.pth'
    joblib.dump(result, out_path)
    print(f'[clean] output={out_path}')


if __name__ == '__main__':
    main()
