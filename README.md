# Ego2LeRobot

把 Egocentric-100K 这类第一人称视频做成 VLA 训练用的 14 维 state/action LeRobot 数据。

路径：规格化 → YOLO 手部过滤 → GeoCalib → MoGe + HaWoR → MegaSaM → 世界系转换与轨迹清理 → `lerobot_v3` → Pi0.5 14 维格式。

下面命令用仓库 [`examples/egovideo/`](examples/egovideo/) 里的 3 条 Egocentric-100K 视频做端到端示例。

## 做什么

把 ego 视频转成**世界坐标系双手末端 14DoF 轨迹**，用来训练 VLA，再看开环测试。

<p align="center">
  <img src="docs/assets/ego_source.gif" width="48%" alt="原始 ego 视频">
  <img src="docs/assets/world_14dof.gif" width="48%" alt="世界坐标系双手末端 14DoF 轨迹">
  <br>
  <em>左：原始 ego 视频 → 右：世界系双手末端 14DoF 轨迹</em>
</p>

<p align="center">
  <img src="docs/assets/train_loss.png" width="92%" alt="训练 loss：红 ego delta EE，绿示教 delta joint">
  <br>
  <em>训练 loss。红：ego delta EE；绿：示教 delta joint</em>
</p>

<p align="center">
  <img src="docs/assets/openloop_test.png" width="92%" alt="ego 数据训练模型开环测试">
  <br>
  <em>ego 数据训练后的开环测试（GT vs inferred）</em>
</p>

原视频见 [`docs/assets/ego_source.mp4`](docs/assets/ego_source.mp4)、[`docs/assets/world_14dof.mp4`](docs/assets/world_14dof.mp4)。

## 环境

新机器先装 Conda 环境并注册第三方后端：

```bash
bash scripts/create_env.sh
conda activate ego2lerobot
bash scripts/install_data_backends.sh
python -m cli doctor --profile data --strict
```

仓库根目录已有 `.venv` 时，直接激活即可：

```bash
source .venv/bin/activate   # 或直接用 .venv/bin/python

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export EGO2LEROBOT_RAY_TMP=/data/luogz/tmp/ray
export TORCH_HOME=/data/luogz/cache/torch
export TRITON_CACHE_DIR=/data/luogz/cache/triton
export EGO2LEROBOT_GPU_ALLOC_GB=120
export RAY_local_fs_capacity_threshold=0.99
export PYTHONUNBUFFERED=1

python -m cli doctor --profile data --strict
```

权重不随仓库分发，放到 `model/` 后可用 `python -m cli doctor --profile data --strict` 检查。HaWoR 源码放到 `third_party/HaWoR/`。细节见 [`docs/installation.md`](docs/installation.md)。

| 权重 | 本地路径 | 下载 |
| --- | --- | --- |
| GeoCalib pinhole | `model/geocalib/pinhole.tar` | [GitHub Release](https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar) |
| MoGe-2 ViT-L + normal | `model/moge2/model.pt` | [Hugging Face](https://huggingface.co/Ruicheng/moge-2-vitl-normal/resolve/main/model.pt) |
| MegaSaM | `model/megasam/megasam_final.pth` | [Hugging Face](https://huggingface.co/datasets/UCBProject/Affordance2Grasp-Weights/resolve/main/MegaSAM/checkpoints/megasam_final.pth) |
| HaWoR | `model/hawor/hawor.ckpt` | [Hugging Face](https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/hawor.ckpt) |
| HaWoR config | `model/hawor/model_config.yaml` | [Hugging Face](https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/model_config.yaml) |
| HaWoR 手部检测器 | `model/hawor/detector.pt` | [Hugging Face](https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt) |

```bash
mkdir -p model/geocalib model/moge2 model/megasam model/hawor

wget -O model/geocalib/pinhole.tar \
  https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar

wget -O model/moge2/model.pt \
  https://huggingface.co/Ruicheng/moge-2-vitl-normal/resolve/main/model.pt

wget -O model/megasam/megasam_final.pth \
  https://huggingface.co/datasets/UCBProject/Affordance2Grasp-Weights/resolve/main/MegaSAM/checkpoints/megasam_final.pth

wget -O model/hawor/hawor.ckpt \
  https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/hawor.ckpt
wget -O model/hawor/model_config.yaml \
  https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/model_config.yaml
wget -O model/hawor/detector.pt \
  https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt
```

## 预处理

规格化 15 fps、不超过 456×256、60s 切段，再切掉没手、超过两只手、同侧重复的帧。

```bash
#规格化：30fps、不超过 456×256、60s 切段（原文生产参数）
python ego_pipeline/preprocessing/reprocessed/main.py \
  --ops fps+resolution+segment \
  --input  examples/egovideo \
  --output_dir examples/reprocessed \
  --max_fps 15 \
  --max_width 456 --max_height 256 \
  --scale_mode resize \
  --duration 60 \
  --codec h264 --crf 18 --preset ultrafast \
  --workers 16 \
  --overwrite \
  --verify

#YOLO 手部过滤：切掉没手、超过两只手、同侧重复
python ego_pipeline/preprocessing/filtered/main.py \
  --ops hand_filter \
  --input  examples/reprocessed \
  --output_dir examples/filtered \
  --filter_multi_hands --require_lr_pair \
  --gpus 6,7 \
  --overwrite

#校验段区间不重叠、元数据完整
python ego_pipeline/preprocessing/filtered/verify.py \
  --input  examples/reprocessed \
  --output examples/filtered
```

## 重建管线

```bash
EGO2LEROBOT_GPU_ALLOC_GB=120 \
python -m cli pipeline \
  --input  examples/filtered \
  --output examples/pipeline \
  --num_gpus 8 \
  --test \
  --long_video_threshold_s 1000000 \
  --per_video_lerobot \
  --aggregate_lerobot \
  --delete_temp yes
```

中断后续跑：

```bash
EGO2LEROBOT_GPU_ALLOC_GB=120 \
python -m cli pipeline \
  --input  examples/filtered \
  --resume examples/pipeline \
  --num_gpus 8 \
  --test \
  --long_video_threshold_s 1000000 \
  --retry_failed
```

## 转成 Pi0.5 14 维

`--action-mode absolute` 存绝对值 EE；去掉该参数则是 delta EE。

```bash
python ego_pipeline/tasks/lerobot_to_pi05.py \
  --source examples/pipeline \
  --output examples/lerobot_pi05 \
  --action-mode absolute \
  --prompt "assemble electronic products" \
  --overwrite
```

中断后续跑：

```bash
python ego_pipeline/tasks/lerobot_to_pi05.py \
  --source examples/pipeline \
  --output examples/lerobot_pi05 \
  --resume
```

## 3D 可视化

`--source-v3` 指向管线为该 episode 写出的 `lerobot_v3` 目录；跑完上面的管线后按实际路径替换。

```bash
python ego_pipeline/tasks/visualize_pi05_ee.py \
  --dataset examples/lerobot_pi05 \
  --source-v3 examples/pipeline \
  --output examples/lerobot_pi05/viz_episode_000000.mp4
```
