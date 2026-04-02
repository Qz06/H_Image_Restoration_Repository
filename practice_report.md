# KAIR 训练实践问题记录

## 1. 任务背景

本次目标是基于配置文件：

`/root/autodl-tmp/KAIR/options/my_train_usrnet.json`

在 KAIR 项目中从头开始训练一个 `USRNet` 模型。

训练入口使用的是：

```bash
python main_train_usrnet.py -opt /root/autodl-tmp/KAIR/options/my_train_usrnet.json
```

在实际启动过程中，连续遇到了多个由新版 `NumPy`、`PyTorch`、`SciPy` 与旧版 KAIR 代码不兼容导致的问题。

---

## 2. 遇到的问题与解决办法

### 问题 1：`np.int` 已被新版 NumPy 删除

#### 报错现象

训练在数据喂入阶段报错：

```python
AttributeError: module 'numpy' has no attribute 'int'
```

报错位置：

- `models/model_plain4.py`
- 触发语句：
  ```python
  self.sf = np.int(data['sf'][0,...].squeeze().cpu().numpy())
  ```

#### 原因分析

`np.int` 是旧版 NumPy 中对 Python 内置 `int` 的别名，已在 NumPy 1.20+ 中弃用，并在更高版本中移除。

#### 解决办法

将 `np.int(...)` 替换为 `int(...)`。

#### 实际修改

- `models/model_plain4.py`
  - `np.int(...)` -> `int(...)`

另外，为避免项目中其他脚本以后再遇到同类问题，也顺手修复了两处潜在兼容问题：

- `utils/utils_image.py`
  - `dtype=np.int` -> `dtype=int`
- `main_test_ircnn_denoiser.py`
  - `np.int(...)` -> `int(...)`

---

### 问题 2：`torch.rfft` / `torch.irfft` 已被新版 PyTorch 删除

#### 报错现象

训练在 `USRNet` 前向传播阶段报错：

```python
AttributeError: module 'torch' has no attribute 'rfft'
```

报错位置：

- `models/network_usrnet.py`

#### 原因分析

旧版 KAIR 的 `network_usrnet.py` 使用了：

- `torch.rfft`
- `torch.irfft`

这些接口已经在较新的 PyTorch 中移除，新的实现方式改为 `torch.fft` 模块。

#### 解决办法

KAIR 仓库中已经自带了一个兼容新版 PyTorch 的实现文件：

- `models/network_usrnet_v1.py`

因此最直接、最稳妥的做法是用新版兼容实现替换原始 `network_usrnet.py`。

#### 实际修改

- 备份：
  - `models/network_usrnet.py.bak`
- 替换方式：
  - 用 `models/network_usrnet_v1.py` 覆盖 `models/network_usrnet.py`

#### 修改结果

新的实现改用了以下新版接口：

- `torch.fft.fftn`
- `torch.fft.ifftn`
- `torch.conj`

从而解决了 `torch.rfft` / `torch.irfft` 兼容性问题。

---

### 问题 3：`scipy.finfo` 调用错误

#### 报错现象

训练在生成模糊核阶段报错：

```python
AttributeError: Module 'scipy' has no attribute 'finfo'
```

报错链路：

- `data/dataset_usrnet.py`
- `utils/utils_deblur.py`
- `fspecial_gaussian(...)`

具体报错位置：

```python
h[h < scipy.finfo(float).eps * h.max()] = 0
```

#### 原因分析

`finfo` 属于 `numpy`，不是 `scipy` 顶层模块的通用属性。旧代码里写成了 `scipy.finfo(...)`，在当前 SciPy 版本下会报错。

#### 解决办法

将：

```python
scipy.finfo(float)
```

替换为：

```python
np.finfo(float)
```

#### 实际修改

- `utils/utils_deblur.py`
  - `scipy.finfo` -> `np.finfo`
- `utils/utils_blindsr.py`
  - `scipy.finfo` -> `np.finfo`

之所以连 `utils/utils_blindsr.py` 一起改，是因为其中也存在相同写法，属于同类兼容隐患。

---

## 3. 当前仍存在但未阻塞训练的问题

### `lr_scheduler.step()` 的 warning

训练启动时还有如下 warning：

```python
UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`.
UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated...
```

#### 原因分析

KAIR 当前的学习率调度实现中：

- `models/model_base.py` 的 `update_learning_rate(self, n)` 使用了：
  ```python
  scheduler.step(n)
  ```
- 而训练脚本中是在每轮优化前调用 `update_learning_rate(current_step)`

这与新版 PyTorch 推荐的调用顺序不完全一致，因此会触发 warning。

#### 影响

- 这是 warning，不是导致训练崩溃的根因
- 当前不会像前面几个兼容问题一样直接中断训练
- 可能影响学习率调度初始阶段的行为，但不妨碍先把训练跑起来

#### 当前处理状态

- 暂未修改
- 属于后续可以继续优化的兼容性问题

---

## 4. 已修改的文件列表

本次一共涉及如下文件调整：

1. `models/model_plain4.py`
   - 修复 `np.int` 兼容问题

2. `utils/utils_image.py`
   - 修复 `dtype=np.int` 兼容问题

3. `main_test_ircnn_denoiser.py`
   - 修复 `np.int` 兼容问题

4. `models/network_usrnet.py`
   - 用 `models/network_usrnet_v1.py` 的新版实现替换，修复 `torch.rfft/irfft` 问题

5. `utils/utils_deblur.py`
   - 修复 `scipy.finfo` -> `np.finfo`

6. `utils/utils_blindsr.py`
   - 修复 `scipy.finfo` -> `np.finfo`

---

## 5. 训练经验总结

### 经验 1：KAIR 是老项目，直接在新环境下跑很容易遇到兼容问题

主要集中在：

- `NumPy`
- `PyTorch`
- `SciPy`

这些底层依赖的新版本接口变更上。

### 经验 2：优先判断是“致命错误”还是“非致命 warning”

本次真正导致训练中断的是：

- `np.int`
- `torch.rfft`
- `scipy.finfo`

而不是 `lr_scheduler` warning。

### 经验 3：能复用项目内已有兼容实现时，优先复用

例如 `USRNet` 的 PyTorch 兼容问题，项目已经提供了：

- `models/network_usrnet_v1.py`

直接复用它比手工逐行改造旧版频域代码更稳妥。

### 经验 4：从头训练时要特别注意自动续训机制

`main_train_usrnet.py` 会自动从：

`path.root/task/models/`

下寻找最新 checkpoint 并继续训练。

因此如果想真正“从头开始”，需要满足以下任一条件：

1. 在配置文件中使用一个全新的 `task` 名称
2. 删除旧的任务输出目录后再重新训练

---

## 6. 建议的后续检查项

后续如果训练继续报错，优先排查以下方向：

1. 数据集路径是否正确
2. `my_train_usrnet.json` 中的 `task` 是否对应一个干净输出目录
3. `kernels/kernels_12.mat` 是否存在
4. `trainsets/trainH` 与验证集路径是否已经准备完成
5. 是否还有其他旧版 API 在训练过程中未被覆盖到

---

## 7. 当前结论

到目前为止，已经解决的主要阻塞问题包括：

- `NumPy` 的 `np.int` 兼容问题
- `PyTorch` 的 `torch.rfft/torch.irfft` 兼容问题
- `SciPy` 的 `scipy.finfo` 调用问题

当前剩余的 `lr_scheduler.step()` warning 暂时不阻塞训练，可留作后续优化项处理。
