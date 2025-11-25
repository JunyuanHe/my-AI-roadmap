下面给你**完整的“资料 + 论文 + 代码仓库”清单**，严格对应你 6 个月规划的每一模块。
你可以直接照着这些资源学，能节省你 **80% 的寻资料时间**，同时保证都是最权威和可复现的实现。

我会按模块整理，并且给你：

* **核心论文（必须读）**
* **书/教程**
* **重要代码仓库（一定要跑）**
* **有用的工程工具链**
* **推荐的学习顺序**

---

# ⭐ 总目录（按主题）

1. NeRF & 神经隐式场
2. 可微渲染
3. 多视几何 & 3D 重建
4. 3D 生成模型 (EG3D / GET3D / DreamFusion)
5. 深度估计、SLAM、Semantic Reconstruction
6. Embodied AI / 3D 交互
7. 大规模数据集
8. 工程工具链（可直接使用）

---

# 1️⃣ **NeRF & 神经隐式场**

## 📌 必读论文

这些是整个神经场方向的核心基础，必须全部读：

* **NeRF (2020)**: Neural Radiance Fields — *最原始论文*
* **Mip-NeRF / Mip-NeRF 360** — 多尺度抗 aliasing
* **instant-NGP (2022)** — 关键的多分辨率 Hash Grid 加速
* **Neural SDF / VolSDF / NeuS** — 用 SDF 做表面
* **3D Gaussian Splatting (SIGGRAPH 2023)** — 目前最快的神经场渲染方法

## 📌 代码仓库（必跑）

* **instant-ngp（官方）**
  [https://github.com/NVlabs/instant-ngp](https://github.com/NVlabs/instant-ngp)

  > 你的第一站，训练速度极快。
* **Gaussian Splatting 官方**
  [https://github.com/graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
* **nerfstudio（最强 NeRF 框架）**
  [https://github.com/nerfstudio-project/nerfstudio](https://github.com/nerfstudio-project/nerfstudio)

  > 你要看“method implementations”，可以学到训练 pipeline 如何组织。

## 📌 教程/笔记

* **JumpStart NeRF by NerfStudio**
* **Kaolin Tutorials (NVIDIA)**
  [https://github.com/NVIDIAGameWorks/kaolin](https://github.com/NVIDIAGameWorks/kaolin)
* **Lil’s NeRF tutorial**（入门很快）

## 📌 推荐学习顺序

1. instant-ngp（上手快）
2. nerfstudio（学结构）
3. 高斯点模型（新趋势）
4. SDF-based NeRF（表面提取）

---

# 2️⃣ **可微渲染（Differentiable Rendering）**

## 📌 必读论文

* **Soft Rasterizer (SoftRas)**
* **Neural Mesh Renderer (NMR)**
* **DIB-R（NVIDIA）**
* **PyTorch3D Rendering**
* **SDF-based differentiable rendering (VolSDF / NeuS)**

## 📌 代码仓库

* **PyTorch3D（最全面）**
  [https://github.com/facebookresearch/pytorch3d](https://github.com/facebookresearch/pytorch3d)
* **NVIDIA Kaolin（工业级）**
  [https://github.com/NVIDIAGameWorks/kaolin](https://github.com/NVIDIAGameWorks/kaolin)
* **SoftRas**
  [https://github.com/ShichenLiu/SoftRas](https://github.com/ShichenLiu/SoftRas)
* **DIB-R (NVIDIA)**
  [https://github.com/NVlabs/DIB-R](https://github.com/NVlabs/DIB-R)

## 📌 教程

* PyTorch3D 官方 tutorials（非常实战）
* Kaolin_differentiable_rendering_tutorials

## 📌 推荐学习顺序

1. 跑 PyTorch3D 的 mesh → image 渲染
2. 实现一个 tiny differentiable rasterizer
3. 用可微渲染做 mesh refine

---

# 3️⃣ **多视几何 & 3D 重建**

## 📌 必读论文 / 书

* **Szeliski《Computer Vision: Algorithms and Applications》**（必读章节：SfM、MVS）
* **Hartley & Zisserman《Multiple View Geometry》**（经典）
* **COLMAP 官方论文**
* **OpenMVS / MVSNet 论文**

## 📌 工具与代码

* **COLMAP（最重要的 SfM 工具）**
  [https://github.com/colmap/colmap](https://github.com/colmap/colmap)
* **OpenMVS / OpenMVG**
* **MVSNet 系列**（深度多视重建）
  [https://github.com/YoYo000/MVSNet](https://github.com/YoYo000/MVSNet)
* **NeRF + COLMAP pipeline 示例**
  用 COLMAP 生成相机位姿 → 再送入 NeRF

## 📌 推荐学习顺序

1. 跑 COLMAP（本地照片 → sparse → dense）
2. 跑 OpenMVS（mesh）
3. 深度 MVS（MVSNet）
4. NeRF + SfM 配合使用

---

# 4️⃣ **3D 生成模型（EG3D / GET3D / DreamFusion / 3DGS-Gen）**

## 📌 必读论文

* **EG3D (CVPR 2022)**
* **GET3D (NeurIPS 2022)**
* **DreamFusion (2022)** — 文本 → 3D
* **Magic3D (2022)**
* **Fantasia3D (2023)**
* **ProlificDreamer (2024)**
* **GaussianDreamer / 3D Gaussian Splatting 生成模型**

## 📌 核心代码仓库

* **EG3D 官方**
  [https://github.com/NVlabs/eg3d](https://github.com/NVlabs/eg3d)
* **GET3D 官方**
  [https://github.com/nv-tlabs/GET3D](https://github.com/nv-tlabs/GET3D)
* **DreamFusion 复现（NeRF + SDS）**
  [https://github.com/ashawkey/stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion)
* **Magic3D unofficial**
  [https://github.com/yuyangzhang/magic3d](https://github.com/yuyangzhang/magic3d)
* **3D Gaussian Dreaming 系列**
  [https://github.com/dvlab-research/GaussianDreamer](https://github.com/dvlab-research/GaussianDreamer)

## 📌 推荐学习顺序

1. EG3D（tri-plane 原理最值得学）
2. GET3D（显式 mesh 生成）
3. DreamFusion（最重要的 Text→3D 范式）
4. GaussianDreamer（新趋势 + 最快）

---

# 5️⃣ **深度估计、SLAM、语义重建**

## 📌 必读论文

* Depth estimation:

  * **DPT**
  * **MiDaS**
  * **NeW CRFs**
* SLAM:

  * **ORB-SLAM3**
  * **DSO**
  * **DROID-SLAM (2021)** — 深度 SLAM 巅峰
* 3D 语义：

  * **PointNet++**
  * **MinkowskiNet（sparse conv）**
  * **SemanticFusion / PanopticFusion**

## 📌 代码仓库（必跑）

* **DROID-SLAM**
  [https://github.com/princeton-vl/DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM)
* **ORB-SLAM3**
  [https://github.com/UZ-SLAMLab/ORB_SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)
* **DPT/MiDaS**
  [https://github.com/isl-org/MiDaS](https://github.com/isl-org/MiDaS)
* **MinkowskiEngine（最强 sparse conv 框架）**
  [https://github.com/NVIDIA/MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)
* **Open3D（点云处理神器）**
  [https://github.com/isl-org/Open3D](https://github.com/isl-org/Open3D)

## 📌 推荐学习顺序

1. MiDaS：预测 depth
2. ORB-SLAM3：理解传统 pipeline
3. DROID-SLAM：学习深度视觉 SLAM
4. MinkowskiNet：做语义点云分割

---

# 6️⃣ **Embodied AI / 交互 / 物理模拟**

## 📌 必读论文

* **Habitat**
* **iGibson / BEHAVIOR**
* **AI2-THOR**
* **RoboSuite / RLBench（机器人操作）**
* **DiffSkill / DiffTaichi（可微物理）**

## 📌 代码仓库

* **Habitat-lab + Habitat-sim（最重要）**
  [https://github.com/facebookresearch/habitat-lab](https://github.com/facebookresearch/habitat-lab)
* **iGibson / BEHAVIOR**
  [https://github.com/StanfordVL/iGibson](https://github.com/StanfordVL/iGibson)
* **AI2-THOR**
  [https://github.com/allenai/ai2thor](https://github.com/allenai/ai2thor)
* **Isaac Gym / Isaac Sim**
* **RLBench**
  [https://github.com/stepjam/RLBench](https://github.com/stepjam/RLBench)
* **DiffTaichi（可微物理）**
  [https://github.com/yuanming-hu/difftaichi](https://github.com/yuanming-hu/difftaichi)

## 📌 推荐学习顺序

1. Habitat navigation
2. AI2-THOR object interaction
3. iGibson physics-aware tasks
4. 机器人操作（RLBench 或 Isaac）

---

# 7️⃣ **数据集**

## 📌 重建 / 室内 3D

* **ScanNet**
* **Matterport3D**
* **Replica Dataset（Habitat 官方）**
* **HM3D（高质量室内模型库）**

## 📌 Embodied 任务

* **Gibson / Stanford2D-3D**
* **THOR 官方场景**

## 📌 3D 生成

* **ShapeNet**（必备）
* **Objaverse（超大规模 3D 模型）**
* **ABO dataset（真实物体）**

---

# 8️⃣ **工程与工具链（非常重要）**

## 📌 几何处理

* **Open3D（强烈推荐）**
  点云 → mesh → ICP → RANSAC 全都有
* **Trimesh**
* **MeshLab**

## 📌 学术级图形学库

* **Taichi（数值 + 可微）**
* **PyTorch3D**
* **Kaolin**

## 📌 Diffusion / Text-to-Image

* **Stable Diffusion**
  [https://github.com/CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
* **HuggingFace Diffusers**（DreamFusion 最常用）
  [https://github.com/huggingface/diffusers](https://github.com/huggingface/diffusers)

## 📌 渲染工具

* **Blender + Blender Python**（脚本生成数据）
* **Unity + HDRP**（可做 photoreal simulation）
* **Unreal Engine + USD pipeline**（工业级）

---

# 🚀 最后：我可以把这些资源按你的 6 个月学习计划**逐周绑定到任务列表**

比如：

> 第 3 月第 2 周要读 EG3D 论文 + 跑 EG3D 官方代码 + 做 tri-plane 实验。

如果你愿意，我可以帮你把**所有资料整合成一个带跳转链接的“周任务表”**（你直接照着执行即可）。你想让我帮你做这个吗？
