以下是对 `code_inventory.txt` 中 174 个文件的职责描述分析：

### 根目录 (Main Scripts)
- main_challenge_sr.py: 针对超分辨率竞赛设计的推断与提交结果生成脚本。
- main_download_pretrained_models.py: 自动从云端下载各模型预训练权重文件的实用工具。
- main_test_dncnn.py: DnCNN 高性能降噪网络的测试与性能评估主程序。
- main_test_dncnn3_deblocking.py: 使用 DnCNN-3 版本进行图像去伪影与去块效应测试。
- main_test_dpsr.py: 深度即插即用超分辨率（DPSR）算法的测试与验证脚本。
- main_test_face_enhancement.py: 集成人脸检测与修复功能的图像人脸增强测试程序。
- main_test_fdncnn.py: 灵活降噪网络 FdnCNN 的测试脚本，支持多种噪声水平。
- main_test_ffdnet.py: 快速灵活降噪网络 FFDNet 的推理与质量测试程序。
- main_test_imdn.py: 轻量级信息多重蒸馏网络 IMDN 的超分辨率测试脚本。
- main_test_ircnn_denoiser.py: 基于迭代修复型卷积神经网络的降噪器测试程序。
- main_test_msrresnet.py: 经典 MSRResNet 超分辨率模型的测试与图像生成脚本。
- main_test_rrdb.py: 基于残留残差密集块（RRDB）模型的超分辨率性能测试。
- main_test_rvrt.py: 递归视频修复互感器 RVRT 的视频增强效果测试脚本。
- main_test_srmd.py: 超分辨率退化模型（SRMD）的批量图像修复测试程序。
- main_test_swinir.py: 基于 Swin Transformer 的图像修复网络高性能测试脚本。
- main_test_usrnet.py: 展开式深度超分辨率网络 USRNet 的多功能测试程序。
- main_test_vrt.py: 视频修复互感器 VRT 的复杂视频序列增强测试脚本。
- main_train_dncnn.py: 针对 DnCNN 降噪模型的模型训练、日志记录与导出。
- main_train_drunet.py: 深度残差降噪网络 DRUNet 的标准训练流程控制脚本。
- main_train_gan.py: 基于生成对抗网络（GAN）的超分辨率模型训练主程序。
- main_train_psnr.py: 以提升 PSNR 评分为目标的传统超分辨率模型训练脚本。
- main_train_usrnet.py: 深度迭代超分辨率 USRNet 的端到端训练框架脚本。
- main_train_vrt.py: 针对大规模视频数据集的 VRT 网络分布式训练脚本。

### data 目录 (Dataset Handling)
- data/__init__.py: 数据处理模块入口，负责导出数据集相关的核心类与函数。
- data/dataset_blindsr.py: 针对盲超分辨率任务的随机退化数据集加载与增强。
- data/dataset_dncnn.py: 适配 DnCNN 训练的图像对加载与噪声合成逻辑。
- data/dataset_dnpatch.py: 专注于小块图像（Patch）的快速加载，提升降噪训练效率。
- data/dataset_dpsr.py: DPSR 所需的高低分辨率图像对及模糊核数据加载器。
- data/dataset_fdncnn.py: FdnCNN 模型训练时动态添加噪声水平图的数据处理。
- data/dataset_ffdnet.py: 配合 FFDNet 使用的图像降采样与噪声增强数据集。
- data/dataset_jpeg.py: 专门用于 JPEG 压缩伪影去除任务的数据集准备工具。
- data/dataset_l.py: 加载单通道亮度分量（L 维度）的图像数据处理逻辑。
- data/dataset_plain.py: 通用的高清/低清图像对加载器，不执行特定退化。
- data/dataset_plainpatch.py: 以 Patch 为单位的普通超分辨率训练数据加载逻辑。
- data/dataset_sr.py: 超分辨率任务标准数据集加载器，支持多种缩放因子。
- data/dataset_srmd.py: 为 SRMD 模型提供包含降采样与模糊信息的复合数据。
- data/dataset_usrnet.py: USRNet 采样与模糊卷积所需的高维度数据预处理。
- data/dataset_video_test.py: 视频修复测试阶段的连续帧序列加载与边界处理。
- data/dataset_video_train.py: 视频修复训练阶段的滑窗分帧与多帧增强加载器。
- data/select_dataset.py: 根据配置文件参数动态路由并初始化指定数据集实例。

### SR/my-usrnet/options (Experimental Configs)
- SR/my-usrnet/options/my_train_usrnet_260331_140232.json 至 153637.json (11个文件):
  USRNet 重复实验导出的配置文件，记录了从 14:02:32 到 15:36:37 间不同时间戳下的训练参数。

### models 目录 (Models & Architecture)
- models/op/__init__.py: 自定义高效算子模块入口，集成 CUDA 加速层映射。
- models/op/deform_attn.py: 视频修复中的可变形注意力机制 CUDA 算子具体实现。
- models/op/fused_act.py: 融合激活函数的高效计算算子，优化训练推理速度。
- models/op/upfirdn2d.py: 支持高质量上采样与 FIR 滤波的 2D 卷积算子。
- models/basicblock.py: 模型基础构建块合集，包含残差、密集块等组件实现。
- models/loss.py: 训练损失函数集合，包含 L1、Charbonnier 及对抗损失。
- models/loss_ssim.py: 结构相似性（SSIM）损失函数的 PyTorch 微分实现。
- models/model_base.py: 所有模型类的抽象基类，定义训练、保存、加载通用接口。
- models/model_gan.py: 基于对抗学习框架的模型控制逻辑，管理生成器与判别器。
- models/model_plain.py: 定义最基础的端到端图像修复训练逻辑与反馈循环。
- models/model_plain2.py: plain 模型的变体，支持更灵活的优化器配置与参数调整。
- models/model_plain4.py: plain 模型的高级变体，支持更复杂的前向推理与记录。
- models/model_vrt.py: 针对 VRT 架构定制的端到端视频序列修复控制模型类。
- models/network_discriminator.py: 图像 GAN 训练中常用的 PatchGAN 或经典判别器网络。
- models/network_dncnn.py: DnCNN 深度卷积神经网络结构的具体层级定义与实现。
- models/network_dpsr.py: 为 DPSR 算法设计的具有迭代思想的深度网络骨干。
- models/network_faceenhancer.py: 集成人脸特征提取与细节恢复的综合增强网络结构。
- models/network_feature.py: 预训练特征提取网络（如 VGG），用于计算感知损失。
- models/network_ffdnet.py: FFDNet 网络架构，包含噪声图拼接到图像通道的逻辑。
- models/network_imdn.py: 轻量级架构 IMDN 的具体层叠与通道蒸馏逻辑实现。
- models/network_msrresnet.py: MSRResNet 残差网络结构定义，用于高性能超分。
- models/network_rrdb.py: RRDB 模块在网络中的拓扑结构及全局衔接定义。
- models/network_rrdbnet.py: 完整的基于 RRDB 的图像生成网络，适配 ESRGAN。
- models/network_rvrt.py: 递归视频修复互感器 RVRT 的模块化组件与拓扑定义。
- models/network_srmd.py: 基于维度增强的超分辨率退化模型（SRMD）骨干结构。
- models/network_swinir.py: SwinIR 核心架构，基于移动窗口自注意力机制实现。
- models/network_unet.py: 通用 U-Net 及其变体结构，适用于多种降噪与修复任务。
- models/network_usrnet.py: 深度展开式 USRNet 及其模块化迭代过程的网络定义。
- models/network_usrnet_v1.py: USRNet 的版本 1 实现，记录了早期架构细节。
- models/network_vrt.py: VRT 的核心架构，包含多维注意力和空时融合层。
- models/select_model.py: 工厂模式实现，根据配置字符串实例化对应的 model 类。
- models/select_network.py: 工厂模式实现，根据配置字符串实例化具体的网络骨干。

### options 目录 (Global Configs)
- options/rvrt/*.json (6个文件): 涵盖 RVRT 在 REDS、Vimeo-90K、DVD 等不同任务上的训练配置。
- options/swinir/*.json (7个文件): 定义 SwinIR 在图像降噪、经典/轻量超分、GAN 增强上的配置。
- options/vrt/*.json (9个文件): 覆盖 VRT 在视频超分、去模糊、去噪等全场景下的实验参数。
- options/my_train_usrnet.json: 用户的 USRNet 训练自定义主配置文件。
- options/train_bsrgan_x4_gan.json: 针对 BSRGAN 的对抗生成模型训练超参数定义。
- options/train_bsrgan_x4_psnr.json: 针对 BSRGAN 的 PSNR 指向型训练阶段配置文件。
- options/train_dncnn.json: 经典 DnCNN 降噪网络训练所需的迭代与学习率参数。
- options/train_dpsr.json: DPSR 算法在特定规模下的训练流程控制配置文件。
- options/train_drunet.json: DRUNet 深度降噪模型在标准数据集上的实验配置。
- options/train_fdncnn.json: 灵活降噪 FdnCNN 的多层噪声训练参数配置。
- options/train_ffdnet.json: FFDNet 训练过程涉及的数据路径与超参数定义。
- options/train_imdn.json: 轻量级 IMDN 模型的训练环境设置与优化方案。
- options/train_msrresnet_gan.json: 基于对抗损失的 MSRResNet 增强训练配置。
- options/train_msrresnet_psnr.json: 旨在获得最高 PSNR 的 MSRResNet 训练配置。
- options/train_rrdb_psnr.json: 基于 RRDB 骨干的传统超分辨率训练参数配置文件。
- options/train_srmd.json: 针对多退化模型 SRMD 的专属训练控制参数。
- options/train_usrnet.json: 系统默认的 USRNet 深度展开训练全局配置文件。

### retinaface 目录 (Face Detection Component)
- retinaface/data_faces/__init__.py: 人脸数据处理包初始化，导出标注解析工具。
- retinaface/data_faces/config.py: 存储 RetinaFace 检测器的不同骨干权重配置参数。
- retinaface/data_faces/data_augment.py: 人脸检测训练专属的数据增强策略（翻转、裁剪、抖动）。
- retinaface/data_faces/wider_face.py: 适配 WIDER FACE 标注格式的高效解析与加载器。
- retinaface/facemodels/__init__.py: 人脸网络架构包入口，组织各层级引用关系。
- retinaface/facemodels/net.py: RetinaFace 主干网络的深度定义与多层特征融合实现。
- retinaface/facemodels/retinaface.py: RetinaFace 完整的检测逻辑，包含输入处理与输出聚合。
- retinaface/layers/functions/prior_box.py: 在输入图上预生成不同尺度检测框（Prior-Box）的应用。
- retinaface/layers/modules/__init__.py: 导出人脸专用损失函数模块与其依赖组件。
- retinaface/layers/modules/multibox_loss.py: 复现人脸定位与分类的多任务 MultiBox 损失。
- retinaface/layers/__init__.py: 层级组件包总入口，导出所有自定义子层。
- retinaface/utils_faces/nms/__init__.py: 非极大值抑制（NMS）算法的 Python 封装入口。
- retinaface/utils_faces/nms/py_cpu_nms.py: 经典的 CPU 版 NMS 实现，用于过滤重叠检测框。
- retinaface/utils_faces/__init__.py: 人脸工具包入口，集成边界框转化与计时逻辑。
- retinaface/utils_faces/box_utils.py: 包含边界框编解码、IoU 计算等几何运算的核心工具。
- retinaface/utils_faces/timer.py: 衡量检测各阶段运行时间的高精度计时程序。
- retinaface/retinaface_detection.py: 封装好的 RetinaFace 检测器类，供图像增强脚本调用。

### scripts 目录 (Data Scripts)
- scripts/data_preparation/create_lmdb.py: 将原始图片集高效转化为存储友好的 LMDB 数据库。
- scripts/data_preparation/extract_subimages.py: 按照指定步长裁剪大图为子图，生成训练样本。
- scripts/data_preparation/prepare_DAVIS.py: 针对 DAVIS 视频数据集进行清洗与分帧预处理。
- scripts/data_preparation/prepare_DVD.py: 针对 DVD 视频数据集进行针对性的格式转化脚本。
- scripts/data_preparation/prepare_GoPro_as_video.py: 将 GoPro 序列集转化为视频修复训练所需格式。
- scripts/data_preparation/prepare_UDM10.py: 针对 UDM10 测试集进行图像序列路径生成与组织。
- scripts/data_preparation/regroup_reds_dataset.py: 重新整理 REDS 数据集的分片文件结构与分布逻辑。
- scripts/matlab_scripts/evaluate_video_deblurring.m: 用于全量评估视频去模糊指标（PSNR/SSIM）的 MATLAB 脚本。
- scripts/matlab_scripts/generate_LR_UDM10_BD.m: 使用模糊下采样方式在 MATLAB 中生成 UDM10 低清集。
- scripts/matlab_scripts/generate_LR_Vimeo90K.m: 使用 Bicubic 下采样方式生成 Vimeo-90K 训练数据对。
- scripts/matlab_scripts/generate_LR_Vimeo90K_BD.m: 使用高斯模糊下采样生成高质量低清 Vimeo-90K 数据。

### matlab 目录 (Auxiliary Tools)
- matlab/Cal_PSNRSSIM.m: 实现精确的 PSNR 与 SSIM 评估算法，适用于学术对比。
- matlab/center_replace.m: 用于修复结果中的边缘对齐或中心位置图像替换。
- matlab/main_denoising_color.m: 基于传统或 MATLAB 接口进行彩色图像降噪的主脚本。
- matlab/main_denoising_gray.m: 专注于灰度图降噪效果评估与传统算法实现。
- matlab/modcrop.m: 对图像进行模数裁剪（Mod-Crop），使其符合缩放因子倍数。
- matlab/shave.m: 在评估前剃除图像边缘像素，避免卷积引入的伪影误差。
- matlab/zoom_function.m: 各种插值算法（双三次、双线性）的 MATLAB 缩放实现。

### utils 目录 (Utilities)
- utils/utils_alignfaces.py: 执行人脸关键点对齐与预处理的高级图像算法。
- utils/utils_blindsr.py: 盲超分场景下动态模糊核生成与降采样策略集。
- utils/utils_bnorm.py: 批量归一化（BN）层参数调整与权重修复实用函数。
- utils/utils_deblur.py: 去模糊任务中常见的维纳滤波或其他频率域算子。
- utils/utils_dist.py: 分布式训练（DDP）相关的环境变量初始化与通信工具。
- utils/utils_googledownload.py: 提供从 Google Drive 驱动器进行文件拉取的接口。
- utils/utils_image.py: 图像处理各阶段（格式转换、归一化、保存）的核心工具。
- utils/utils_lmdb.py: LMDB 读写的高层次封装，支持并发数据拉取。
- utils/utils_logger.py: 训练日志与可视化工具包装，支持屏幕输出与文件记录。
- utils/utils_mat.py: 读写 Python 与 MATLAB (.mat) 格式数据的核心接口。
- utils/utils_matconvnet.py: 兼容 MatConvNet 模型格式转换的遗留支持工具。
- utils/utils_model.py: 管理模型权重保存、指数移动平均 (EMA) 的通用方法。
- utils/utils_modelsummary.py: 生成模型结构报告并计算参数量与 FLOPs 的工具。
- utils/utils_option.py: 解析与解析 JSON 配置文件并构建实验路径的核心。
- utils/utils_params.py: 动态管理模型参数冻结、解冻与细粒度控制逻辑。
- utils/utils_receptivefield.py: 计算复杂深度网络各层理论感受野的相关函数。
- utils/utils_regularizers.py: 为网络训练提供正则化项（如 TV 损失）的辅助函数。
- utils/utils_sisr.py: 专属于单图像超分辨率（SISR）的特定数据与指标工具。
- utils/utils_video.py: 处理视频帧对齐、时间轴合并及光流可视化的函数库。
- utils/utils_videoio.py: 视频文件的高效读写与编解码适配工具。

---

### 覆盖统计
- **总数**: 174
- **已覆盖**: 174
- **未覆盖**: 无

---

### 8 条关键调用链 (入口->数据->模型->网络->工具)
1. **USRNet 训练**: [main_train_usrnet.py](main_train_usrnet.py) -> [data/dataset_usrnet.py](data/dataset_usrnet.py) -> [models/model_plain.py](models/model_plain.py) -> [models/network_usrnet.py](models/network_usrnet.py) -> [utils/utils_blindsr.py](utils/utils_blindsr.py)
2. **SwinIR 超分测试**: [main_test_swinir.py](main_test_swinir.py) -> [data/dataset_plain.py](data/dataset_plain.py) -> [models/model_plain.py](models/model_plain.py) -> [models/network_swinir.py](models/network_swinir.py) -> [utils/utils_image.py](utils/utils_image.py)
3. **视频 VRT 训练**: [main_train_vrt.py](main_train_vrt.py) -> [data/dataset_video_train.py](data/dataset_video_train.py) -> [models/model_vrt.py](models/model_vrt.py) -> [models/network_vrt.py](models/network_vrt.py) -> [utils/utils_dist.py](utils/utils_dist.py)
4. **人脸增强流程**: [main_test_face_enhancement.py](main_test_face_enhancement.py) -> [data/dataset_plain.py](data/dataset_plain.py) -> [retinaface/retinaface_detection.py](retinaface/retinaface_detection.py) -> [models/network_faceenhancer.py](models/network_faceenhancer.py) -> [utils/utils_alignfaces.py](utils/utils_alignfaces.py)
5. **GAN 超分训练**: [main_train_gan.py](main_train_gan.py) -> [data/dataset_sr.py](data/dataset_sr.py) -> [models/model_gan.py](models/model_gan.py) -> [models/network_rrdbnet.py](models/network_rrdbnet.py) -> [models/loss.py](models/loss.py)
6. **FFDNet 降噪**: [main_test_ffdnet.py](main_test_ffdnet.py) -> [data/dataset_ffdnet.py](data/dataset_ffdnet.py) -> [models/model_plain.py](models/model_plain.py) -> [models/network_ffdnet.py](models/network_ffdnet.py) -> [utils/utils_model.py](utils/utils_model.py)
7. **RVRT 视频修复**: [main_test_rvrt.py](main_test_rvrt.py) -> [data/dataset_video_test.py](data/dataset_video_test.py) -> [models/model_vrt.py](models/model_vrt.py) -> [models/network_rvrt.py](models/network_rvrt.py) -> [utils/utils_video.py](utils/utils_video.py)
8. **数据预处理**: [scripts/data_preparation/create_lmdb.py](scripts/data_preparation/create_lmdb.py) -> [scripts/data_preparation/extract_subimages.py](scripts/data_preparation/extract_subimages.py) -> 无 -> 无 -> [utils/utils_lmdb.py](utils/utils_lmdb.py)