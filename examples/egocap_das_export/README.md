# EgoCap 自采数据（LYZ）转换件

`egocap_*` 目录是 EgoCap 双目头戴设备录制的 ros bag（mcap）经
[`convert_egocap.py`](../../ego_pipeline/preprocessing/das/convert_egocap.py)
转换后的 DAS 兼容导出件。仓库附 `egocap_20261001_103828_bag_0` 一条作为样本，
其余录制本地转换后不随 git 分发（体积原因）。

每条录制的导出目录包含：

| 文件 | 内容 |
| --- | --- |
| `camera{0,1}.mp4` | 左/右目原始图像重编码（pinhole 直通） |
| `camera{0,1}_rect.mp4`（+`.json`） | 去畸变视频及内参（fuse 读取 K） |
| `camera{0,1}_timestamps.npz` | 逐帧 `log_time_ns`，视频与 IMU 共同时间基 |
| `imu.npz` | 陀螺 rad/s + 加计 g，与视频同一 log 时钟 |
| `calib_camera{0,1}.json` | 双目标定（`make_egocap_calibration` 生成的 fallback 标定） |

从原始 mcap 复现转换：

```bash
python ego_pipeline/preprocessing/das/convert_egocap.py \
    --input <LYZ录制目录>/bag --name <录制名>
```

转换完成后有两种方式跑完剩余管道（YOLO 手部过滤 → GeoCalib → MoGe + HaWoR
→ MegaSaM → 双目融合 → IMU 增强 → Pi0.5）：

- Web 界面：`python ego_pipeline/preprocessing/das/app.py`，数据源下拉选择
  “EgoCap已转换 · <name>/”，模式选 `stereo`；
- 命令行：`python ego_pipeline/preprocessing/das/run_multiview.py --mode stereo
  --mcap <导出目录> --start .. --duration .. --out ...`。
