# ReLi3D 时序版说明

本文档按**运行时先后顺序**说明 `main.py -> pipeline/Reli3D.py -> pipeline/reli3d_blender_render.py` 的调用链。

项目根目录：`D:\Coding Apps\My_Project\hejunbaseline`

---

## A. 启动阶段

1. 解析参数（`main.py`）
- 入口：`if __name__ == "__main__"`
- 读取 `--baseline ReLi3D` 以及所有 `--reli3d_*` 参数。

2. 进入 `main(args)`（`main.py`）
- 创建 `dataset = EvalDataset(...)`
- 选择 pipeline：`Reli3DPipeline(...)`
- 构建 `DataLoader`
- 进入评估循环：`log_validation(...)`

3. `Reli3DPipeline.__init__`（`pipeline/Reli3D.py`）
- 配置路径与缓存目录
- `_init_reli3d_imports()` 导入 ReLi3D 组件
- `_resolve_config_path/_resolve_checkpoint_path`
- 如果 `use_official_infer=False`：`_load_model()` 载入网络权重

---

## B. 每个 batch 的主循环

4. `log_validation` 取一个 batch（`main.py`）
- 调 `outputs = pipeline(filtered_batch)`
- 对 tuple 输出拆包：`relit, depth_pred, mask_pred`
- 用 `metric_fn(...)` 算指标
- 保存图像/深度/json

5. `Reli3DPipeline.__call__`（`pipeline/Reli3D.py`）
- 读取 batch 字段：
  - `source_images/source_view/source_Ks`
  - `target_view/target_Ks/target_lighting`
- 若 `use_official_infer=True`，先批量准备 mesh：`_prepare_official_meshes_for_batch(batch)`

---

## C. 重建阶段（mesh）

### 路径 1：官方 infer（当前默认）

6. `_prepare_official_meshes_for_batch`（批处理）
- 对 batch 内每个 sample：
  - `_sample_paths(...)` 决定缓存路径
  - 缓存无 mesh 才加入 jobs
  - `_convert_source_view_for_reli3d(...)`
  - `_export_case_inputs(...)` 导出官方输入（rgba + transforms）
- 一次 subprocess 调官方脚本：
  - `python demos/reli3d/infer_from_transforms.py ... --objects case1 case2 ...`
- 拷贝输出 mesh 到缓存
- HDR 使用 `_write_hdr(target_lighting, hdr_path)`

### 路径 2：内存模型（`use_official_infer=False`）

7. `_reconstruct_mesh(...)`
- `_build_mapper_batch(...)` 构建 mapper 输入
- `self.model.get_mesh(...)` 生成 mesh
- 导出 `mesh.glb`
- HDR 同样用 `_write_hdr(target_lighting, hdr_path)`

---

## D. 渲染阶段（Blender）

8. 回到 `__call__`，逐 sample 调 `_render_with_blender(...)`
- 组织 `job.json`（mesh/hdr/targets/K/c2w/分辨率）
- subprocess 调 Blender：
  - `blender --background --python pipeline/reli3d_blender_render.py -- --job job.json`

9. Blender 脚本 `main()`（`pipeline/reli3d_blender_render.py`）
- `_reset_scene()`：空场景 + Cycles + GPU尝试 + Z pass
- `_setup_world(hdr_path)`：加载环境光 HDR
- `_setup_depth_output(depth_dir)`：配置 EXR depth 输出
- 导入 `mesh.glb`
- 遍历 target views：
  - `_set_camera_intrinsics(...)`
  - `_cv_to_blender_c2w(...)`
  - 渲染 RGB PNG 与 depth EXR
- 写 `done.txt`

10. `_render_with_blender` 回收输出
- 读取 `rgb/*.png`、`depth/*.exr`
- `_read_exr_depth(...)` 读深度
- 深度后处理：
  - 背景按 alpha 置 0
  - 过远异常值（>=1e6）置 0
- 返回 `rgb_arr/depth_arr/mask_arr`

---

## E. 异常分支（关键）

11. `__call__` 内渲染异常处理
- 若 RuntimeError 含 `Bad glTF` / `json contained NaN`：
  - 删除坏缓存 mesh
  - 重新重建并重试一次
- 若仍失败：
  - 跳过该 sample（不中断整轮）
  - 当前 sample 输出保持零

12. 深度读取失败
- `_read_exr_depth` 捕获异常，告警并返回零深度

---

## F. 回到评估与落盘

13. `log_validation`（`main.py`）
- 用 `metric_fn` 计算每个 sample 指标
- ReLi3D 分支保存：
  - `pred_relight.png`
  - `pred_depths.png`（可视化）
  - `pred_depth.exr/.npy`（原始）
  - 可选 `gt.png` + `gt_depth.exr/.npy`
- 更新 `results.json` 与进度条

---

## G. 一句话总流程

`main.py` 读数据 -> `Reli3DPipeline` 先重建 mesh（官方/内存模型）-> Blender 脚本按 target view + target HDR 渲染 -> 回传 RGB/Depth/Mask -> `main.py` 算分并保存。

---

## H. 调试建议（按时序定位）

- 卡在重建：看 `_official_inputs` / 官方 infer stdout stderr
- 卡在渲染：看 Blender stdout/stderr、是否写出 `done.txt`
- 深度异常：看 `pred_depth.exr` 与 `gt_depth.exr` 的背景约定是否一致
- 慢：分解耗时为 reconstruct/render/metrics/json 四段
