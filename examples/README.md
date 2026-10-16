# Examples

## 自采数据（克隆即得）

| 目录 | 内容 |
| --- | --- |
| [`egocap_das_export/`](egocap_das_export/) | EgoCap（LYZ）双目 ros bag 转换件样本一条：双目视频 + 逐帧时间戳 + IMU（同 log 时钟）+ 标定，详见其 [README](egocap_das_export/README.md) |
| [`das_compare/`](das_compare/) | DAS-Ego 六目录制的 mono / stereo / rig6 三模式对比完整产物（multiview.mp4、指标、快照、融合 parquet） |
| [`das_pi05_preview/`](das_pi05_preview/) | 自采录制端到端产出的完整 Pi0.5 episode（`data/` state/action parquet、`meta/`、`videos/`） |

更大的数据集（其余 EgoCap 转换件、原始 mcap、批量 pipeline 中间产物）不随 git 分发。

## 公共数据集示例

从 Egocentric-100K `factory001` 抽出的 3 条视频，放在 `egovideo/`，目录结构与原数据集一致，可直接作为 `--input`。

| 文件 | 时长 | 分辨率 |
| --- | --- | --- |
| `egovideo/factory001/worker001/factory_001_worker_001_0024.mp4` | 179.8 s | 456×256 |
| `egovideo/factory001/worker002/factory_001_worker_002_0157.mp4` | 148.8 s | 456×256 |
| `egovideo/factory001/worker003/factory_001_worker_003_0187.mp4` | 159.3 s | 456×256 |

每个 clip 附带原数据集的 `.json` 元数据和对应 worker 的 `intrinsics.json`。
