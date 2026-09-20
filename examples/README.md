# Examples

从 Egocentric-100K `factory001` 抽出的 3 条视频，放在 `egovideo/`，目录结构与原数据集一致，可直接作为 `--input`。

| 文件 | 时长 | 分辨率 |
| --- | --- | --- |
| `egovideo/factory001/worker001/factory_001_worker_001_0024.mp4` | 179.8 s | 456×256 |
| `egovideo/factory001/worker002/factory_001_worker_002_0157.mp4` | 148.8 s | 456×256 |
| `egovideo/factory001/worker003/factory_001_worker_003_0187.mp4` | 159.3 s | 456×256 |

每个 clip 附带原数据集的 `.json` 元数据和对应 worker 的 `intrinsics.json`。
