# DAS 数据接入（mcap → 管线输入）

把 DAS-Ego 头戴设备录制的 mcap（6 目鱼眼 + IMU）转成重建管线可用的
单路 pinhole mp4；支持单目/双目/六目多视图重建。

## 统一入口（推荐，即"三个开关"）

`run_multiview.py` 一条命令跑完全链（mcap → 去畸变 → 预处理 → 逐目重建
→ 多目融合 → IMU 增强 → 可视化 + Pi0.5），已有中间产物自动跳过：

```bash
cd /home/jun/Ego2LeRobot && conda activate ego2lerobot

# 六目（全 rig 融合，度量尺度来自立体三角化）
python ego_pipeline/preprocessing/das/run_multiview.py --mode rig6 \
    --mcap '<DAS-Ego_xxx.mcap>' --start 2.4 --duration 5 --out examples/out_rig6

# 双目 / 单目：改 --mode 即可
python ego_pipeline/preprocessing/das/run_multiview.py --mode stereo --mcap ... --start .. --duration .. --out ...
python ego_pipeline/preprocessing/das/run_multiview.py --mode mono   --mcap ... --start .. --duration .. --out ...
```

可视化布局随模式自动变化（`viz_multiview.py`，UI 可直接复用的渲染端）：
**mono=1 输入面板 / stereo=2 / rig6=6（3×2）**，每路输入各自叠加手部建模，
下方共享一个 IMU 增强的世界系双手 3D 面板。每次输出到
`<out>/viz/<mode>_<时间戳>/`，**不覆盖历史**，方便多次对比。

三模式同窗对比实测（c2a589d7, 2.4–7.4s，见 `examples/das_compare/`）：

| 模式 | EE 位置抖动 | 相对单目 |
| --- | --- | --- |
| mono | 0.01096 | — |
| stereo | 0.00349 | −68% |
| rig6 | 0.00265 | −76% |

## 脚本

| 脚本 | 作用 |
| --- | --- |
| `run_multiview.py` | **统一入口**：mono/stereo/rig6 三模式 |
| `export_mcap.py` | mcap → 各目 mp4 + 时间戳 npz + IMU npz + `calib_cameraN.json`（`--camera all` 导全部 6 目） |
| `undistort_ds.py` | double-sphere → pinhole 去畸变 |
| `fuse_multiview.py` | 多目融合：Umeyama 跨目配准 + 立体三角化度量尺度 + 逐帧中值投票；非参考目自动对齐帧数（±1 帧）；三角化样本 <3 时尺度回退 1.0 |
| `gap_fill.py` | 融合轨迹短时补全：≤3 帧缺失 SLERP 插值 + `observed/interpolated/invalid` 状态列 |
| `imu_enhance.py` | IMU 时间基自校准 + 重力对齐 + 陀螺去抖 + 四面板可视化 |
| `inspect_cameras.py` | 相机朝向检查工具 |

仓库根目录另有 `scripts/audit_trajectory.py`（轨迹质量审计）与
`scripts/run_gapfill_lyz.sh`（LYZ 双目全链批量：run_multiview → gap_fill → audit）。

## 轨迹质量审计与短时补全（在管线中的位置）

审计与补全的输入不是视频，是 `run_multiview` 的融合产物 `fused.parquet`
（逐帧双手世界系轨迹表），因此二者插在管道末端：

```text
ros bag (mcap)
  │  convert_egocap.py（解包：视频/IMU 解出、时间对齐）
  ▼
egocap_das_export/<录制>/           ← 双目视频 + 逐帧时间戳 + IMU(同钟) + 标定
  │  run_multiview.py（裁窗 → 逐目重建 → 融合）
  ▼
<out>/fused.parquet                 ← 待检/待补的轨迹表
  │  ★ scripts/audit_trajectory.py（体检：哪里缺、哪里跳）
  │  ★ ego_pipeline/preprocessing/das/gap_fill.py（治疗：≤3 帧小洞插值补全）
  ▼
<out>/fused_filled.parquet          ← 含 *_status 状态列，不覆盖原文件
  │  viz_multiview.py / Pi0.5 导出
  ▼
```

三步用法（选数据靠 `--parquet`：精确路径查一段，通配符批量查）：

```bash
# 1) 体检——覆盖率/连续缺失段/可疑速度与加速度/左右手间距突变/kp21 形变
python scripts/audit_trajectory.py \
  --parquet 'examples/das_app_runs/*/fused.parquet' --out-dir examples/trajectory_qc
#   （界面每次运行的产物都在 examples/das_app_runs/<运行名>/fused.parquet）

# 2) 治疗——≤3 帧缺失段插值补全（位置线性、朝向 SLERP、关节四元数归一化），
#    每帧输出 left/right_status ∈ {observed, interpolated, invalid}，
#    补出的帧绝不与真实观测混同；>3 帧的大洞与贴边缺失保持 invalid 不动
python ego_pipeline/preprocessing/das/gap_fill.py \
  --parquet <某次运行>/fused.parquet            # 输出 fused_filled.parquet

# 3) 复查——补全前后对比
python scripts/audit_trajectory.py \
  --parquet <某次运行>/fused.parquet <某次运行>/fused_filled.parquet \
  --out-dir <某处>/qc
```

精度参考（人工挖洞测试，干净段挖 1–3 帧）：位置恢复误差 1.4–2.8 mm、
旋转 0.12–6.25°，远低于跨目配准残差。注意二者目前是**手动后处理步骤**，
未挂进 run_multiview 一条龙——界面/命令行跑完出的 viz 与 Pi0.5 基于
未补全的 fused；需要补全版数据时按上面 2→3 手动执行。

## 实测效果（c2a589d7 录制，2.4–7.4s 窗口）

- 6 目中 5 目双手近满保留（第 6 目仅右手），融合后双手 75/75 帧、平均 5.9 目/帧
- 跨目配准腕点残差 ~32mm；5 目尺度收敛 0.945–0.955（互证 SLAM 尺度一致性）
- 立体三角化：单目世界 = 1.27 × 米制
- **EE 位置抖动较单目降低 74.7%**（0.0104 → 0.0026）
- IMU：重力对齐 120.5°、时间基自校准、去抖 Δp 中位 ~12mm

## 数据事实（实测）

- 话题前缀 `/robot0`；相机 `/robot0/sensor/cameraN/compressed`（h264 裸流，30 Hz），
  IMU `/robot0/sensor/imu`（200 Hz，gyro rad/s，accel **g**）。
- `camera_info.D = [fx, fy, cx, cy, xi, alpha]`，标准 double-sphere 模型
  （Usenko 3DV'18，与 basalt 实现一致）。41 个录制共用同一套出厂标定。
- `T_b_c = [tx,ty,tz, qx,qy,qz,qw]`。6 目分布：camera0/4 = 左眼簇（前/后），
  camera1/5 = 右眼簇，camera2/3 ≈ 机体 (0,0,-1)。默认参考目 **camera0**。
- h264 流从会话中途订阅开始，首帧缺 SPS/PPS —— 导出时从第一个含 SPS 的
  消息起步（`dropped_frames` 记录丢弃数）。
- 选窗建议：双手图像水平分离 ≥50px（近垂直对齐会触发管线左右身份翻转）、
  画面无第三只手、佩戴者不交叉手。

## 单步用法（内部脚本直接调用）

```bash
# 导出（文件或目录；--camera all | camera0,camera2 | camera0）
python ego_pipeline/preprocessing/das/export_mcap.py \
  --input <mcap或目录> --camera all --output /mnt/d/work/ego/das_export

# 去畸变自检（round-trip 应 < 1e-4 度）
python ego_pipeline/preprocessing/das/undistort_ds.py \
  --calib <导出>/calib_camera0.json --selftest

# 融合（多目 parquet → fused.parquet）
python ego_pipeline/preprocessing/das/fuse_multiview.py \
  --views camera0=<p0> camera1=<p1> ... --ref camera0 --stereo camera1 \
  --rig-cal <导出> --videos <窗口视频目录> --out fused.parquet
```
