# KAIR Optimization Report and Phase Plan (Scope-Limited)

## Scope
This review only covers:
- Common framework files (training loop, options parsing, model base, utility modules)
- SwinIR-related files
- BSRGAN-related files
- USRNet-related files

Other model families are intentionally out of scope for this round.

## Method
- Static code inspection only (no long-running training/inference)
- Cross-check between config files, training scripts, test scripts, and model modules
- Focus on correctness, reproducibility, maintainability, and runtime safety

## Detailed Findings

### P0 (Must fix first)
1. Missing import causes runtime failure in distributed gather
   - File: utils/utils_dist.py
   - Issue: pickle is used in all_gather but never imported
   - Risk: NameError at runtime

2. Device and wrapping logic is unsafe/non-optimal in model base
   - File: models/model_base.py
   - Issues:
     - Device selection uses gpu_ids is not None instead of checking CUDA availability
     - Non-dist path always wraps network with DataParallel
   - Risk: wrong device behavior on CPU-only env; unnecessary overhead on single GPU

3. Gradient clipping path can crash
   - File: models/model_plain.py
   - Issue: clip_grad_norm_ is called on self.parameters() while ModelPlain is not nn.Module
   - Risk: runtime error when clipgrad is enabled

4. SwinIR test script can easily hit model-structure mismatch
   - File: main_test_swinir.py
   - Issue: model img_size is derived from --training_patch_size and strict state_dict loading; this often mismatches checkpoint masks
   - Risk: frequent RuntimeError during checkpoint loading, user confusion

5. USRNet batch scale handling is correctness-sensitive
   - Files: data/dataset_usrnet.py, models/model_plain4.py
   - Issue: model_plain4 previously used only the first sample's sf for the whole batch
   - Risk: incorrect forward behavior if a mixed-sf batch appears

### P1 (Second wave)
1. Boolean CLI parsing robustness in train scripts
   - Files: main_train_psnr.py, main_train_gan.py

2. Endless epoch loops without explicit max-iter stop control
   - Files: main_train_psnr.py, main_train_gan.py, main_train_usrnet.py

3. Config-runtime inconsistency for optimizer weight decay in GAN model
   - File: models/model_gan.py

4. Metric protocol mismatch risk between train-time validation and standalone testing
   - Files: main_train_*.py, main_test_*.py (especially USRNet and SwinIR)

5. Dataset pairing by sorted order only
   - File: data/dataset_sr.py

### P2 (Cleanup and modernization)
1. Heavy and side-effect imports in utils_image
2. Duplicate utility functions (checkpoint discovery)
3. Dependency specification incompleteness and no version pinning
4. Legacy backup files and naming consistency
5. Downloader robustness (timeout/retry/streaming)

## Phase Plan

### Phase 1 (Now): correctness and safety hardening (P0)
- [x] Fix missing pickle import in utils_dist
- [x] Harden model_base device selection and wrapping logic
- [x] Fix clip_grad target in model_plain
- [x] Harden SwinIR checkpoint loading and img_size mismatch handling in main_test_swinir
- [x] Make ModelPlain4 robust to mixed per-sample scale factors
- [x] Run short smoke checks only (no long-running jobs)

### Phase 2
- CLI parsing consistency and finite-run controls
- Logging and metric protocol alignment
- Config semantics cleanup (single-GPU friendliness, explicit behavior)

### Phase 3
- Utility deduplication, dependency improvements, docs refresh

### Phase 4
- Runtime-side-effect cleanup and tooling hardening
- Downloader robustness under unstable network
- Dependency declaration completeness

## Time-Safety Policy for Validation
- No multi-hour training or full-dataset long inference
- Only short checks with explicit timeout
- Prefer import-level and 1-2 sample smoke tests

## Phase 1 Execution Notes (This Round)
Code files updated:
- utils/utils_dist.py
- models/model_base.py
- models/model_plain.py
- models/model_plain4.py
- main_test_swinir.py

Short validation results:
- py_compile passed for all modified files
- static diagnostics reported no Python errors in modified files
- SwinIR one-image smoke test succeeded in the kair environment with intentional mismatch input:
   - command used --training_patch_size=128 with a checkpoint trained at img_size=48
   - script auto-detected and overrode to img_size=48
   - output image generated successfully

Validation constraints encountered:
- default base python lacked cv2
- some utility-import paths in the kair environment may trigger optional plotting-stack ABI issues unrelated to this phase patch set

## Phase 2/3/4 Execution Notes (This Round)
Code files updated:
- main_train_psnr.py
- main_train_gan.py
- main_train_usrnet.py
- models/model_plain.py
- models/model_gan.py
- utils/utils_option.py
- utils/utils_model.py
- utils/utils_image.py
- main_download_pretrained_models.py
- requirement.txt

Implemented changes summary:
- Phase 2:
   - unified argument parsing in train scripts (single parse call; robust bool parsing for `--dist`)
   - added finite-run stopping based on `train.total_iter` (fallback to `train.max_iter` or 1000000)
   - added explicit validation network control via `train.val_which_model` (`G`/`E`) with clear logging
   - enabled `model.test(use_ema=...)` in `ModelPlain` and `ModelGAN` for train-time metric consistency
   - fixed GAN optimizer semantics to actually honor `G_optimizer_wd`/`D_optimizer_wd`

- Phase 3:
   - deduplicated checkpoint discovery by routing `utils_model.find_last_checkpoint` to canonical `utils_option.find_last_checkpoint`
   - added missing default for `train.D_optimizer_wd` and explicit default for `train.val_which_model`

- Phase 4:
   - removed import-time plotting side effects from `utils_image` (lazy matplotlib import)
   - hardened pretrained model downloader with timeout/retry/streaming and safer model list parsing
   - expanded dependency declaration with missing core packages (`torch`, `matplotlib`, `scipy`)

Short validation results:
- `py_compile` passed for all modified Python files
- `get_errors` reported no diagnostics in modified files
- CLI smoke checks passed:
   - `main_train_psnr.py --help`
   - `main_train_gan.py --help`
   - `main_train_usrnet.py --help`
- Downloader smoke check passed (`--models "foo"` graceful message, no crash)
- `ModelPlain.test` and `ModelGAN.test` signatures confirm optional `use_ema` argument
- `utils_image` import smoke test passed in `kair` environment
