# 快速上手：从零复现 pipeline 与可视化界面

写给第一次接触本项目的同事：照着做，就能在自己的机器上跑通
**ros bag（mcap）→ 视频/IMU 对齐 → 手部重建 → LeRobot/Pi0.5 训练数据** 的完整管道。

## 0. 这个框架是什么

一条把第一人称视频变成 VLA 训练数据的管道：

```
ros bag (mcap) ──解包/视频·IMU 时间对齐──> 鱼眼去畸变 ──> YOLO 手部过滤
   ──> GeoCalib 相机估计 ──> MoGe 深度 + HaWoR 手部建模 ──> MegaSaM SLAM
   ──> 多目融合（Umeyama 配准 + 立体三角化定尺度）──> IMU 增强（重力对齐+去抖）
   ──> 世界系双手 14-DoF 轨迹 ──> lerobot_v3 / Pi0.5 训练数据
```

也支持公共数据集（Egocentric-100K 等）直接作为输入。项目自带三类示例数据
（见 [`examples/`](../examples/)），克隆即得，本指南第 5 步直接用它们演示。

## 1. 机器要求

- Linux x86_64（WSL2 也可以），NVIDIA GPU（驱动支持 CUDA），显存建议 ≥ 24 GB
- 已装 Conda；系统里有 `ffmpeg`
- 磁盘：环境约 30 GB，模型权重约 10 GB

## 2. 克隆代码 + 装环境（一条脚本）

```bash
git clone https://github.com/junzzzz0420-star/EgoVLA.git
cd EgoVLA
bash scripts/create_env.sh          # 自动建 conda 环境，默认走 USTC 镜像
conda activate ego2lerobot
```

脚本中断了就**重新跑一遍**，会接着装。装完先不急着 doctor——权重还没放，
检查会报缺东西，先做第 3 步。

## 3. 放置模型资产（许可要求，不能随仓库分发）

### 3.1 HaWoR 源码（克隆官方仓库）

```bash
git clone https://github.com/ThunderVVV/HaWoR third_party/HaWoR
```

### 3.2 六个模型权重（逐条复制执行）

```bash
wget -O model/geocalib/pinhole.tar \
  https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar

wget -O model/moge2/model.pt \
  https://huggingface.co/Ruicheng/moge-2-vitl-normal/resolve/main/model.pt

wget -O model/megasam/megasam_final.pth \
  https://huggingface.co/datasets/UCBProject/Affordance2Grasp-Weights/resolve/main/MegaSAM/checkpoints/megasam_final.pth

wget -O model/hawor/hawor.ckpt \
  https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/hawor.ckpt

wget -O model/hawor/detector.pt \
  https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt

wget -O third_party/HaWoR/weights/hawor/model_config.yaml \
  https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/model_config.yaml
```

HaWoR 还依赖两个权重，按 [HaWoR 官方 README](https://github.com/ThunderVVV/HaWoR)
的下载说明放到：

```text
third_party/HaWoR/weights/external/droid.pth
third_party/HaWoR/thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth
```

### 3.3 MANO 手模型（需官网注册，约 5 分钟）

1. 到 [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/) 注册账号并接受许可；
2. 下载 `mano_v*_*.zip`，解压出 `MANO_LEFT.pkl` / `MANO_RIGHT.pkl`；
3. 放到（注意左右手是两个不同目录）：

```text
third_party/HaWoR/_DATA/data/mano/MANO_RIGHT.pkl
third_party/HaWoR/_DATA/data_left/mano_left/MANO_LEFT.pkl
```

## 4. 注册后端 + 自检

```bash
bash scripts/install_data_backends.sh    # 提示时输入 I AGREE 回车
python -m cli doctor --profile data --strict
```

`doctor` 全绿（exit 0）= 权重、MANO、后端全部就位，可以开跑。
**报缺什么就回第 3 步补什么**，它是唯一的检查入口。

## 5. 跑可视化界面 + 自带样本（10 分钟见效）

```bash
python ego_pipeline/preprocessing/das/app.py
```

浏览器打开 <http://localhost:7861>：

1. **数据源下拉**里直接选 `EgoCap已转换 · egocap_20261001_103828_bag_0/`
   （这是仓库自带的 LYZ 自采双目录制样本，克隆即有，无需配置）；
2. 模式选 `stereo`，设窗口起点 `0`、时长 `5` 秒；
3. 点运行——页面实时滚动日志，完成后页面内直接播放 `multiview.mp4`
   （双目输入叠加手部建模 + 世界系双手 3D 轨迹面板）。

想看六目效果：用 DAS-Ego 设备的 mcap（见第 6 步）选 `rig6` 模式；
三模式的量化对比（抖动 −76% 等）见 [`ego_pipeline/preprocessing/das/README.md`](../ego_pipeline/preprocessing/das/README.md)。

## 6. 跑自己的 mcap 数据

两种方式任选：

- **界面导入**（最简单）：界面"导入"输入框直接粘贴录制路径（Windows 路径
  `D:\x\y` 也可以，自动转换），EgoCap 原始 bag 会自动转换为 DAS 兼容目录，
  转完在下拉选它运行；
- **自动扫描**：启动前设环境变量（分号分隔多个 glob，注意整体用单引号）：

```bash
export EGO2LEROBOT_MCAP_GLOBS='/mnt/d/work/ego/你的目录/**/*.mcap'
python ego_pipeline/preprocessing/das/app.py
```

EgoCap 原始 bag 也可以先用命令行转换再用：
`python ego_pipeline/preprocessing/das/convert_egocap.py --input <bag目录> --name <名字>`

## 7. 跑公共数据集的批量管线（可选）

用自带的 3 条 Egocentric-100K 视频走完整批处理（预处理 → 重建 → Pi0.5），
命令见主 [README](../README.md) 的「预处理」「重建管线」「转成 Pi0.5 14 维」三节。

## 8. 常见问题

| 症状 | 处理 |
| --- | --- |
| `doctor` 报某个权重缺失 | 回第 3 步对照路径补齐（路径必须一字不差） |
| 下载慢/失败 | HuggingFace 可设镜像：`export HF_ENDPOINT=https://hf-mirror.com` 后重试 |
| 界面下拉没有自己的数据 | 检查 `EGO2LEROBOT_MCAP_GLOBS` 是否已 export 且重启了 app；mcap 需 >30MB 才显示 |
| 界面日志报 CUDA OOM | 缩短窗口时长；或 `export EGO2LEROBOT_GPU_ALLOC_GB=24`（按你的显存调） |
| HaWoR 相关报错 | 确认第 3.1/3.2 的 HaWoR 源码与三个文件都就位；仍报错联系项目维护者 |
| WSL 里访问 Windows 盘 | 路径写成 `/mnt/d/...`；界面导入框可直接粘 `D:\...` 自动转换 |
