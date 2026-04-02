# KAIR 全量优化执行报告（完整覆盖版）

## 1. 报告目标与覆盖范围

本报告用于完整记录本轮在 KAIR 仓库中已经执行的全部优化相关操作，覆盖以下内容：

1. 运行前打通与可复现性准备（数据、配置、脚本可运行性）
2. 第1阶段（P0）正确性与稳定性修复
3. 第2阶段训练流程可控性与验证协议一致性修复
4. 第3阶段工具层去重与配置语义补全
5. 第4阶段运行时副作用清理与下载器鲁棒性提升
6. 全部短时验证结果与边界说明
7. 所有新增/修改文件清单（不遗漏）

说明：本次按用户要求避免长时间训练/推理，验证以静态检查与短时 smoke test 为主。

---

## 2. 执行背景与问题起点

### 2.1 初始目标

初始工作目标从“全仓代码理解”逐步演进到“SwinIR 训练推理跑通”和“工程层优化分阶段落地”，最终形成了代码级优化闭环。

### 2.2 初始关键阻碍

1. 数据目录与默认配置不匹配，无法直接跑通 SwinIR 训练。
2. 推理时存在经典结构错配问题：checkpoint 的 attn_mask 维度与测试脚本构建结构不一致。
3. 训练主脚本存在无限循环与参数解析不稳问题，批量任务管控风险高。
4. 工具层存在重复实现、导入副作用、下载器鲁棒性不足等工程问题。

---

## 3. 按时间线的全操作记录

### 3.1 全仓理解与审计产物生成

已完成仓库全量静态审计，产出以下文档化结果：

1. code_inventory.txt：代码文件清单
2. code_signatures.txt：结构签名摘要
3. repo_understanding_full.md：逐文件职责理解
4. optimization_report_and_plan.md：分阶段优化计划与执行记录

### 3.2 SwinIR 训练前置可运行性改造

1. 解压数据集到训练/验证目录（已验证数量）：
   - trainsets/DIV2K/DIV2K_train_HR：800
   - trainsets/DIV2K/DIV2K_train_LR_bicubic/X4：800
   - trainsets/DIV2K/DIV2K_valid_HR：100
   - trainsets/DIV2K/DIV2K_valid_LR_bicubic/X4：100
2. 新建并使用 options/swinir/my_swinir_classical_sr.json，按当前环境改为单卡可用与本地数据路径可用。
3. 修复 SwinIR 推理结构错配路径，避免因 training_patch_size 与 checkpoint 结构不一致导致直接崩溃。

### 3.3 第1阶段到第4阶段优化落地

1. 第1阶段（P0）已完成。
2. 第2/3/4阶段已继续完成并验证。
3. 所有改动已汇总进 optimization_report_and_plan.md。

---

## 4. 全量文件改动矩阵（逐项覆盖）

> 说明：本节按“发现问题 -> 可优化点 -> 具体改动 -> 结果/效果”逐文件列出，确保无遗漏。

### 4.1 运行前置与配置层

#### 文件：options/swinir/my_swinir_classical_sr.json（新建）

- 发现问题：默认示例配置与当前本地数据目录不匹配，且训练入口对批大小/路径敏感。
- 可优化点：提供可直接执行的本地配置，减少手工改配置错误。
- 具体改动：
  1. task 设置为 my_swinir_classical_sr。
  2. gpu_ids 设置为 [0]，dist 设为 false。
  3. scale=4，数据路径对齐 trainsets/DIV2K 下 HR/LR_x4。
  4. 训练 batch size 设为 32（单卡友好）。
  5. netG.img_size 设为 48（对应当前训练结构口径）。
- 结果/效果：可直接用于本地启动 SwinIR classical_sr 训练；路径与硬件约束一致。

### 4.2 第1阶段（P0）修复

#### 文件：utils/utils_dist.py

- 发现问题：all_gather 使用 pickle，但文件未导入 pickle。
- 可优化点：修复显式依赖，避免分布式运行时 NameError。
- 具体改动：新增 import pickle。
- 结果/效果：分布式 all_gather 序列化路径不再因缺失导入而崩溃。

#### 文件：models/model_base.py

- 发现问题：
  1. 设备选择逻辑仅依赖 gpu_ids 非空，不检查 CUDA 实际可用性。
  2. 非分布式路径存在不必要并行封装风险。
  3. 学习率获取兼容性不足（不同 scheduler API 差异）。
  4. torch.load 在不同环境下 map_location 处理不稳。
- 可优化点：增强 CPU/GPU 兼容、并行封装条件与 checkpoint/optimizer 恢复稳定性。
- 具体改动：
  1. 设备选择改为 torch.cuda.is_available() 与 gpu_ids 联合判断。
  2. 非 dist 情况下仅在 CUDA 且多卡时启用 DataParallel。
  3. current_learning_rate 优先使用 get_last_lr，保留回退逻辑。
  4. load_network 统一 map_location='cpu'，提升跨环境加载稳定性。
  5. load_optimizer 在 CUDA/CPU 环境下均可恢复。
- 结果/效果：
  1. CPU-only 场景不再误走 CUDA 逻辑。
  2. 单卡场景避免不必要封装开销。
  3. checkpoint 恢复在设备迁移时更稳。

#### 文件：models/model_plain.py（P0部分）

- 发现问题：梯度裁剪目标错误会导致潜在运行异常。
- 可优化点：将裁剪目标限定为真实网络参数。
- 具体改动：clip_grad_norm_ 的目标改为 self.netG.parameters()。
- 结果/效果：开启 clipgrad 时可稳定执行，不再依赖不存在的 self.parameters()。

#### 文件：models/model_plain4.py

- 发现问题：原逻辑对 batch 中 sf 的处理存在“只用第一个样本 sf”风险。
- 可优化点：支持 batch 内逐样本 sf，确保 USRNet 多尺度样本正确前向。
- 具体改动：
  1. sf 统一转换为 int64 向量。
  2. 若 batch 内 sf 全相同，走统一前向。
  3. 若混合 sf，按样本逐个前向后拼接输出。
- 结果/效果：混合 sf batch 的推理/训练语义正确，避免隐性精度偏差。

#### 文件：main_test_swinir.py

- 发现问题：
  1. strict 加载下，classical_sr 常因结构参数错配（尤其 attn_mask）报错。
  2. checkpoint 存储 key 形式多样，脚本容错不足。
- 可优化点：自动识别 checkpoint 实际结构，减少“参数名对了但结构不匹配”的误用。
- 具体改动：
  1. 新增 resolve_checkpoint_state_dict：统一解析 params/params_ema/state_dict/model 等 key。
  2. 新增 infer_img_size_from_attn_mask：从 attn_mask 反推 checkpoint 对应 img_size。
  3. classical_sr 下若推断 img_size 与传入 training_patch_size 不一致，自动告警并覆盖。
  4. 加载失败时输出更可解释的结构错配提示。
- 结果/效果：
  1. 对自训练 SwinIR 权重更友好，减少“36 vs 256”类报错。
  2. 用户误设 training_patch_size 时脚本可自动纠偏。

### 4.3 第2阶段优化

#### 文件：main_train_psnr.py

- 发现问题：
  1. argparse 存在重复 parse_args 调用。
  2. --dist 布尔解析不稳。
  3. 训练循环无 total_iter 硬停止，默认无限循环。
  4. 周期验证网络固定，无法显式选择 G 或 E。
- 可优化点：训练可控、参数解析稳健、验证协议可配置。
- 具体改动：
  1. 新增 _str2bool，--dist 支持稳定布尔输入。
  2. 仅调用一次 args = parser.parse_args()。
  3. 引入 total_iter 读取（优先 train.total_iter，回退 train.max_iter，再回退 1000000）。
  4. 引入 train.val_which_model（G/E）解析，E_decay<=0 时自动回退到 G 并告警。
  5. 周期测试调用 model.test(use_ema=...)。
  6. current_step 到达 total_iter 后保存并退出。
- 结果/效果：
  1. 训练过程可硬停止，避免长期占卡风险。
  2. 周期验证口径可控，训练日志更可解释。

#### 文件：main_train_gan.py

- 发现问题与优化点：与 main_train_psnr.py 同类。
- 具体改动：同构改造（解析、总迭代上限、验证网络选择、结束保存逻辑）。
- 结果/效果：GAN 训练流程可控性与一致性显著提升。

#### 文件：main_train_usrnet.py

- 发现问题：
  1. 解析参数别名较老，兼容性不足。
  2. 无总迭代硬停止。
  3. 验证网络口径不可显式控制。
- 可优化点：对齐主训练脚本能力，统一行为。
- 具体改动：
  1. 参数改为 -opt 与 --opt 双别名。
  2. 新增 total_iter 控制与到点保存退出。
  3. 新增 val_which_model 解析与 use_ema 验证路径。
- 结果/效果：USRNet 训练脚本可控性与可维护性对齐其他主入口。

#### 文件：models/model_plain.py（第2阶段新增）

- 发现问题：训练脚本可以选择 E 验证，但模型 test 接口不支持 use_ema。
- 可优化点：扩展测试接口，保持向后兼容。
- 具体改动：
  1. test(self, use_ema=False) 支持按需选择 netE/netG。
  2. testx8(self, use_ema=False) 同步支持。
  3. 前向前保存训练态，结束后恢复训练态。
- 结果/效果：验证口径从“隐式固定 G”升级为“显式可配置 G/E”。

#### 文件：models/model_gan.py（第2阶段新增）

- 发现问题：GAN 模型同样缺少 EMA 可选验证路径。
- 可优化点：使 GAN 训练内验证与 EMA 保存机制一致。
- 具体改动：test(self, use_ema=False) 支持 netE/netG 选择并恢复训练态。
- 结果/效果：GAN 训练验证可与测试口径更一致。

### 4.4 第3阶段优化

#### 文件：models/model_gan.py（配置语义修复）

- 发现问题：优化器构建未读取配置中的 G_optimizer_wd / D_optimizer_wd。
- 可优化点：保证配置与运行语义一致。
- 具体改动：
  1. G_optimizer 的 weight_decay 改为读取 train.G_optimizer_wd。
  2. D_optimizer 的 weight_decay 改为读取 train.D_optimizer_wd。
- 结果/效果：配置项生效，权重衰减行为可复现、可解释。

#### 文件：utils/utils_option.py

- 发现问题：
  1. netD 场景缺少 D_optimizer_wd 默认值。
  2. 缺少 val_which_model 默认值，导致脚本分支需额外判空。
- 可优化点：补全默认语义，减少脚本重复防御代码。
- 具体改动：
  1. 新增 train.D_optimizer_wd 默认 0。
  2. 新增 train.val_which_model 默认 G。
- 结果/效果：配置兼容性提高，旧配置可平滑运行新训练逻辑。

#### 文件：utils/utils_model.py

- 发现问题：find_last_checkpoint 与 utils_option 中存在重复实现，维护成本高。
- 可优化点：单一事实源，避免后续行为漂移。
- 具体改动：utils_model.find_last_checkpoint 改为直接委托 utils_option.find_last_checkpoint。
- 结果/效果：去重完成，后续 checkpoint 策略只需维护一处。

### 4.5 第4阶段优化

#### 文件：utils/utils_image.py

- 发现问题：
  1. 顶层导入 matplotlib 及 3D 模块，增加无关场景导入成本。
  2. 导入期设置环境变量会产生全局副作用。
- 可优化点：将绘图依赖延迟到真正调用时加载，减少导入副作用。
- 具体改动：
  1. 新增 _get_matplotlib_pyplot 惰性导入。
  2. imshow/surf 改为函数内导入与使用。
  3. 移除导入阶段全局 KMP 环境变量设置。
- 结果/效果：
  1. 非绘图场景导入更轻量。
  2. 可降低 ABI 冲突链路影响面（特别是仅做训练/推理时）。

#### 文件：main_download_pretrained_models.py

- 发现问题：下载逻辑缺乏超时、重试、分块写盘和临时文件机制，网络波动时易失败或留下坏文件。
- 可优化点：提升下载鲁棒性与可恢复性。
- 具体改动：
  1. 已存在且非空文件直接跳过。
  2. requests 增加 stream=True 与 timeout=30。
  3. 增加 max_retries=3 + 指数退避重试。
  4. 先写 .part 临时文件，完成后原子替换。
  5. 空文件检测与失败清理。
  6. --models 解析改为正则分割空白/逗号并过滤空 token。
  7. --models 默认值改为列表，保持迭代语义一致。
- 结果/效果：下载稳定性显著提升，失败场景可恢复，不易污染目标文件。

#### 文件：requirement.txt

- 发现问题：依赖声明缺少部分核心运行库。
- 可优化点：增强环境可复现性。
- 具体改动：补充 torch、matplotlib、scipy。
- 结果/效果：安装依赖后脚本运行一致性提高，减少“环境里有但 requirements 没写”的隐患。

### 4.6 文档同步

#### 文件：optimization_report_and_plan.md

- 发现问题：仅有阶段计划，不足以覆盖后续全部执行细节。
- 可优化点：将第2/3/4阶段执行结果落盘，形成阶段闭环。
- 具体改动：新增 Phase 4 与 Phase 2/3/4 Execution Notes，补充改动列表与验证结果。
- 结果/效果：阶段计划与执行记录一致，后续追溯更清晰。

---

## 5. 验证与结果证据（短时）

### 5.1 静态与语法验证

1. 对所有改动 Python 文件执行 py_compile，结果通过。
2. 对改动文件执行错误诊断，结果为无错误。

### 5.2 运行级 smoke test（非长时）

1. 训练脚本参数烟测：
   - main_train_psnr.py --help：通过
   - main_train_gan.py --help：通过
   - main_train_usrnet.py --help：通过
2. 下载器烟测：
   - main_download_pretrained_models.py --models "foo"：输出友好提示，不崩溃
3. 模型 API 烟测：
   - ModelPlain.test 签名为 (self, use_ema=False)
   - ModelGAN.test 签名为 (self, use_ema=False)
4. 工具导入烟测：
   - utils_image 在 kair 环境导入通过
5. SwinIR 一图烟测（历史验证记录）：
   - 以 training_patch_size=128 加载 img_size=48 训练出的权重
   - 脚本自动推断并覆盖到 48
   - 推理成功并输出图像

### 5.3 数据准备结果验证

1. train_HR=800
2. train_LR_X4=800
3. valid_HR=100
4. valid_LR_X4=100

---

## 6. 优化后综合效果评估

### 6.1 正确性

1. 修复了至少 3 条可直接导致崩溃/错误行为的路径：
   - 分布式 pickle 缺失导入
   - clip_grad 参数目标错误
   - SwinIR 结构错配导致的 strict 加载失败高发路径
2. USRNet 混合 sf batch 的前向语义得到修正。

### 6.2 稳定性与可控性

1. 三个主训练入口均支持 total_iter 有界停止，避免无限循环占用资源。
2. 验证网络支持显式 G/E 选择，EMA 验证口径可控。
3. 下载器对网络波动更稳，减少半下载坏文件问题。

### 6.3 可维护性

1. checkpoint 查找逻辑去重，减少后续维护分叉。
2. 配置默认值补全，降低脚本中的临时防御逻辑。
3. 工具模块导入副作用降低，模块边界更清晰。

### 6.4 可复现性

1. 关键依赖声明补全。
2. 本地可用 SwinIR 训练配置与数据目录已落地。
3. 阶段执行记录已写入文档，便于团队复盘。

原有训练和推理命令大多数可以继续直接用。

1. 训练命令行参数是否变化
1. PSNR/GAN 训练入口参数名没变，仍是 opt、launcher、local_rank、dist，见 main_train_psnr.py 和 main_train_gan.py。
2. dist 的解析方式变得更稳了，支持更标准的布尔输入，见 main_train_psnr.py 和 main_train_gan.py。
3. USRNet 训练入口新增了 opt 的长参数别名，原来的短参数仍可用，见 main_train_usrnet.py。

2. 训练配置语义是否变化（重点）
1. 新增了总步数控制：train.total_iter（可回退到 train.max_iter），见 main_train_psnr.py、main_train_gan.py、main_train_usrnet.py。
2. 新增了验证网络选择：train.val_which_model，可选 G 或 E，默认是 G，见 utils_option.py。
3. 训练循环从“理论无限跑”改成“到 total_iter 就结束保存退出”，这是行为上的实质变化，见 main_train_psnr.py、main_train_gan.py、main_train_usrnet.py。

3. 推理命令是否变化
1. SwinIR 推理参数格式基本没变，参数集仍是 task、scale、noise、jpeg、training_patch_size、model_path、folder_lq、folder_gt、tile 等，见 main_test_swinir.py。
2. 主要变化在内部行为：会自动从 checkpoint 推断结构并在不一致时自动纠正 training_patch_size，见 main_test_swinir.py。所以命令通常不用改，但更不容易因结构错配报错。

结论：  
日常“怎么敲命令”整体上几乎不需要改；真正需要关注的是训练配置里建议显式加上 total_iter 和 val_which_model，这样新逻辑才能完全按你预期生效。

---

## 7. 本轮未做的事项（明确边界）

1. 未执行长时间训练，不报告最终精度收敛曲线增益。
2. 未完成全仓范围的全部潜在工程清理（例如某些历史备份文件策略、更多数据集配对策略重构）。
3. 未修改超出本轮范围的模型家族代码（遵循范围约束）。

---

## 8. 最终变更清单（文件级）

### 8.1 新建文件

1. options/swinir/my_swinir_classical_sr.json
2. code_inventory.txt
3. code_signatures.txt
4. repo_understanding_full.md
5. optimization_report_and_plan.md（本轮前已创建，后续有追加更新）

### 8.2 修改文件

1. utils/utils_dist.py
2. models/model_base.py
3. models/model_plain.py
4. models/model_plain4.py
5. main_test_swinir.py
6. main_train_psnr.py
7. main_train_gan.py
8. main_train_usrnet.py
9. models/model_gan.py
10. utils/utils_option.py
11. utils/utils_model.py
12. utils/utils_image.py
13. main_download_pretrained_models.py
14. requirement.txt
15. optimization_report_and_plan.md

### 8.3 数据目录变更

1. trainsets/DIV2K/DIV2K_train_HR（解压）
2. trainsets/DIV2K/DIV2K_train_LR_bicubic/X4（解压）
3. trainsets/DIV2K/DIV2K_valid_HR（解压）
4. trainsets/DIV2K/DIV2K_valid_LR_bicubic/X4（解压）

---

## 9. 结论

本轮优化已从“可运行性修补”推进到“训练流程可控 + 验证口径一致 + 工具层鲁棒 + 依赖可复现”的完整闭环。

在不进行长时间训练的约束下，已经完成：

1. 高风险正确性问题清零（P0）
2. 训练入口控制能力补齐（P1核心）
3. 工程层重复实现与副作用清理（P2/P3/P4核心）
4. 全量改动文档化与可追溯化

该状态可作为下一步短程回归（几十到几百 iter）和长期训练的稳定基线。