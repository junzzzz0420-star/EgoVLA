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
| `fuse_multiview.py` | 多目融合：Umeyama 跨目配准 + 立体三角化度量尺度 + 逐帧中值投票 |
| `imu_enhance.py` | IMU 时间基自校准 + 重力对齐 + 陀螺去抖 + 四面板可视化 |
| `inspect_cameras.py` | 相机朝向检查工具 |

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
