# ReLi3D 适配逐函数说明

本文档基于当前项目根目录 `D:\Coding Apps\My_Project\hejunbaseline` 的代码，逐函数解释以下文件：
- `pipeline/Reli3D.py`
- `pipeline/reli3d_blender_render.py`
- `main.py`（与 ReLi3D 相关）

---

## 1) `pipeline/Reli3D.py`

### `Reli3DPipeline.__init__(...)`
- 作用：初始化 ReLi3D pipeline 配置、路径、开关和模型加载策略。
- 关键点：
  - 建立 `cache_dir`。
  - 解析 `reli3d_root/config/checkpoint`。
  - `use_official_infer=True` 时，不加载内存模型，走官方脚本推理。
  - `use_official_infer=False` 时，加载 ReLi3D 模型到 GPU/CPU。

### `_init_reli3d_imports()`
- 作用：把 `ReLi3D` 仓库加到 `sys.path`，导入其核心类和工具。
- 输出：把 `OmegaConf/Names/ReLi3DMapper/FeedForwardSystem/...` 绑定到 `self`。

### `_resolve_config_path(config_path)`
- 作用：确定配置文件路径。
- 规则：优先 CLI 指定，否则尝试 `config.yaml`，不存在再尝试 `raw.yaml`。

### `_resolve_checkpoint_path(checkpoint_path)`
- 作用：确定 checkpoint 路径。
- 规则：优先 CLI 指定，其次环境变量 `RELI3D_CHECKPOINT`，最后默认路径。

### `_load_system_cfg(config_path)`
- 作用：读取并实例化 ReLi3D 的 `FeedForwardSystem.Config`。
- 兼容：支持 `system` 或 `main_module.system` 两种配置结构。

### `_load_model()`
- 作用：加载 ReLi3D 神经网络权重并切到 eval。
- 输出：`self.model`。

### `_convert_source_view_for_reli3d(source_view)`
- 作用：把本项目相机位姿约定转换到 ReLi3D 需要的约定（开关可控）。
- 逻辑：右乘一个 Y/Z 取反的 4x4 变换矩阵。

### `_blender_to_ogl_c2w(c2w_blender)`
- 作用：导出官方 `transforms.json` 时可选把 Blender 坐标系转成 OGL。

### `_export_case_inputs(...)`
- 作用：把一个 sample 导出成 ReLi3D 官方脚本输入格式。
- 产物：
  - `rgba/*.png`（输入图+alpha）
  - `transforms.json`
  - 可选 `camera_debug.json`
  - 首次还会写 `official_infer_cmd.txt` 提示命令
- 关键可控项：
  - `export_principal_mode`（`dataset/center`）
  - `export_fov_mode`（`xy/scalar_x`）
  - `export_coord_system`（`ogl/blender`）

### `_reconstruct_mesh_official(case_dir, sample_dir, target_lighting)`
- 作用：调用官方 `demos/reli3d/infer_from_transforms.py` 重建 mesh。
- 输出：`mesh.glb` 到缓存目录；HDR 采用数据集 target lighting（用于公平 relighting 评测）。

### `_build_mapper_batch(...)`
- 作用：当不走官方脚本时，构造 ReLi3D mapper 输入 batch。
- 做法：
  - source RGB/mask 编码成 jpg bytes。
  - 计算 fov、principal，填入 mapper 需要的字段。

### `_sample_paths(batch_idx, sample_idx)`
- 作用：统一生成该 sample 的缓存路径（sample_dir/mesh/hdr）。

### `_prepare_official_meshes_for_batch(batch)`
- 作用：批量官方重建（一个 subprocess 处理多个对象），减少重复启动开销。
- 行为：
  - 对已有 mesh 的 sample 跳过重建。
  - 对缺失 mesh 的 sample 统一导出输入并 batch infer。

### `_reconstruct_mesh(...)`
- 作用：单 sample 重建入口。
- 逻辑：
  - 缓存命中直接返回。
  - 先导出 case inputs。
  - `use_official_infer=True` -> `_reconstruct_mesh_official`
  - 否则走内存模型 `self.model.get_mesh(...)`。

### `_write_hdr(target_lighting, path)`
- 作用：把数据集的 target lighting tensor 写成 `.hdr` 文件供 Blender world 使用。

### `_render_with_blender(...)`
- 作用：调用 Blender 后台脚本渲染目标视角，回收 RGB/Depth/Mask。
- 流程：
  - 生成临时 `job.json`。
  - subprocess 调 `reli3d_blender_render.py`。
  - 读取 `rgb/*.png` + `depth/*.exr`。
- 深度后处理（当前实现重点）：
  - 背景按 alpha mask 置 0。
  - 过远/异常值（>=1e6）置 0。

### `_read_exr_depth(exr_path, height, width)`
- 作用：读取 EXR 的 R 通道深度。
- 异常策略：读取失败时告警并回退全零深度（不中断整轮）。

### `__call__(batch, **kwargs)`
- 作用：pipeline 主入口，供 `main.py` 直接调用。
- 输入：`source_images/source_view/source_Ks/target_view/target_Ks/target_lighting/...`
- 输出：`(rgb_out, depth_out, mask_out)`，形状均为 `B,F,C,H,W`。
- 关键容错：
  - `Bad glTF/json contained NaN` 时删除坏 mesh，重建并重试一次。
  - 仍失败则跳过该 sample（当前 sample 输出保持零，流程继续）。

### `glob_glob(pattern)`
- 作用：简单封装 `glob.glob`，用于按 pattern 找深度文件。

---

## 2) `pipeline/reli3d_blender_render.py`

### `_enable_cycles_gpu(scene)`
- 作用：尝试启用 Cycles GPU（OPTIX/CUDA/HIP/METAL/ONEAPI），失败回退 CPU。

### `_reset_scene()`
- 作用：清空到 factory empty scene，设置渲染基础参数。
- 默认：Cycles、samples=64、RGBA PNG、开启 Z pass。

### `_setup_world(hdr_path)`
- 作用：创建 world 节点，加载 HDR 环境贴图作为全局照明。

### `_setup_depth_output(depth_dir)`
- 作用：设置 compositor 的 File Output，写 EXR 深度。
- 输出格式：`OPEN_EXR`，32-bit。

### `_set_camera_intrinsics(cam_data, width, height, fx, fy, cx, cy)`
- 作用：把 K 矩阵参数映射到 Blender 相机的 `lens/shift_x/shift_y`。

### `_cv_to_blender_c2w(c2w_cv)`
- 作用：将项目里的相机约定转换为 Blender 世界约定。

### `main()`
- 作用：Blender 脚本入口。
- 流程：
  - 读 `--job` JSON。
  - 建场景、导入 GLB、设置相机。
  - 遍历 targets 渲染 RGB + depth。
  - 写 `done.txt` 作为完成标记。

---

## 3) `main.py`（ReLi3D相关函数）

### `get_device()`
- 作用：返回 `cuda` 或 `cpu`。

### `_save_depth_raw(depth_1hw, exr_path, npy_path)`
- 作用：保存原始深度到 `.npy` 与 `.exr`，用于严格核验。

### `log_validation(dataloader, pipeline, args, metric_fn)`
- 作用：统一评估循环。
- 核心步骤：
  - 支持断点续跑（`--skip_exist`）。
  - 调 `pipeline(filtered_batch)` 得到预测。
  - 计算各指标，更新进度条。
  - 保存预测图与结果 JSON。
- ReLi3D 专用保存逻辑：
  - `pred_depths.png`（可视化）
  - `pred_depth.exr/.npy`（原始）
  - `--save_gt` 时保存 `gt_depth.exr/.npy`
- 深度鲁棒性：若 `depth_pred` 全零或非有限，当前 sample 的深度分跳过（不影响 RGB 指标）。

### `main(args)`
- 作用：根据 `--baseline` 选择 pipeline，构造 dataloader，启动 `log_validation`。
- ReLi3D 分支：
  - 初始化 `Reli3DPipeline` 并传入全部 `--reli3d_*` 参数。
  - dataloader 的 batch size 可被 `--reli3d_recon_chunk_size` 覆盖。

---

## 4) 参数入口（`main.py` 中与 ReLi3D 相关）

- 路径/模型：`--reli3d_root --reli3d_config --reli3d_checkpoint --reli3d_blender_path`
- 重建/缓存：`--reli3d_cache_dir --reli3d_texture_size --reli3d_remesh --reli3d_vertex_count`
- 相机与导出：
  - `--reli3d_no_source_view_conversion`
  - `--reli3d_export_principal_mode`
  - `--reli3d_export_fov_mode`
  - `--reli3d_export_coord_system`
  - `--reli3d_dump_camera_debug`
- 官方推理与调试：
  - `--reli3d_use_official_infer`
  - `--reli3d_export_case_inputs_dir`
  - `--reli3d_render_source_for_debug`
  - `--reli3d_recon_chunk_size`

---

## 5) 端到端流程（一句话）

`main.py` 取 batch -> `Reli3DPipeline` 用 source views/images 重建 mesh -> Blender 脚本在 target views + target HDR 下渲染 RGB/depth/mask -> 回到 `main.py` 保存与算分。
