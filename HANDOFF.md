# 给下一位 AI 的交接说明

更新日期：2026-09-20。以下描述已完成的本机工作，不代表主要实验已经复现成功。

## 目标与当前状态

用户希望在本机用 Codex 复现报告 https://anonymous-report-421.github.io/public-website/?lang=en&view=1 的主要实验。
当前阶段已完成本地执行入口、依赖安装、资源下载、组件验证、独立认证，以及**各一个完整的 Direct 和混合策略回合（含真实 GPT 决策）**。
**尚未运行正式的 100 回合评测**； smoke 回合为单案例验证，不能当作实验结果。
用户已确认单案例验证足以证明链路通畅，不再继续补跑其他案例或启动 100 回合正式实验。

仓库路径：`/home/wangyu/code/GPT-as-Policy`，实际解析到 `/mnt/data1/wy/code/GPT-as-Policy`。
这是报告的公开评测源码，不需要从零实现策略。主要实验是推理评测，不是训练模型。

## 仓库结构与实验口径

- `hybrid_rollout/robodojo/`：仿真 RPC、π₀.₅ 服务、GPT 控制器、动作契约、评分和归档。
- `hybrid_rollout/robolab/`：另一套 RoboLab 集成，本轮未配置。
- `hybrid_rollout/report_site/`、`report_web/`：报告源码和静态站点。
- `public_results/evaluation_cases.json`：已发布的配对案例与历史结果，不是本次新实验结果。
- `runtime/`：本机环境、外部源码、模型、资产、脚本与检查记录；被 Git 忽略，换机器不会随源码自动带走。

复现范围：10 个 RoboDojo 任务 × 5 个固定案例 × 2 种方法 = 100 回合。
`gpt_only` 为 GPT 直接生成末端执行器动作；`pi05_plus_gpt` 为 π₀.₅ 提议、GPT 审核和修正。
模型固定 `gpt-6-astra`，推理强度 `xhigh`，不能静默换模型。
50 个案例按已发布身份从冻结 panel60 中选择；三个泛化任务为 2 个标准 + 3 个随机案例，不能简单取每任务前五项。
历史参考：混合策略成功 24/50、均分 62.60；Direct 成功 13/50、48 个有效分数均分 37.8125。
这些只是公开数据重算，不能作为本机复现成果。RoboLab 和官方非配对 leaderboard 比较不在本次入口范围内。

## 已实现但尚未提交的源码改动

新增：
- `hybrid_rollout/robodojo/local_experiment.py`：`plan / published / doctor / login / run / summarize`；独立认证、账户锁、串行执行、身份校验及原生结果汇总。
  2026-09-19 新增 `prune_codex_trust_sections()`：Codex CLI 每次运行会向 profile 的 `config.toml` 自动追加 `[projects."…"] trust_level` 段，
  导致同 work-dir 重启时 `initialize()` 的模板一致性保护报错；该函数仅当差异完全为此类自动段时恢复模板，其他差异仍拒绝。
- `hybrid_rollout/robodojo/test_local_experiment.py`：12 项本地入口测试（含上述清理函数的通过/拒绝两用例）。
- `hybrid_rollout/robodojo/LOCAL_REPRODUCTION.md`：通用中文操作指南。

修改：
- `README.md`：本地入口说明。
- `hybrid_rollout/robodojo/run_local.sh`：源码根目录自动推导，允许任意绝对结果路径。
- `hybrid_rollout/robodojo/robodojo_server/runtime.sh`：新增系统图形库模式，保留原有 bundled 模式。
- `hybrid_rollout/robodojo/runtime_manifest.py`：记录并正确识别图形库模式。
- `HANDOFF.md`：本文件。

工作树有未提交改动；不要 reset/清理掉它们，尚未创建提交或 PR。

## 本机环境与依赖

Ubuntu 22.04；两张 NVIDIA RTX 6000 Ada（各约 48 GiB）；驱动 570.211.01；CUDA 12.8。
`runtime/run-experiment.sh` 固定 GPU 0 跑 Isaac Sim、GPU 1 跑 π₀.₅。

仿真环境：`runtime/sim-venv`，Python 3.11.15、Isaac Sim 5.1.0.0、torch 2.7.0+cu128、NumPy 1.26.0。
它复用已有基础环境的包，并把新增依赖装在本项目 overlay 中，基础环境必须保留：
`/home/wangyu/code/BEHAVIOR-Challenge/repro/miniforge3/envs/behavior-repro`。
**启动仿真使用 `runtime/sim-python` 包装器**，它预加载基础环境的 `libstdc++.so.6`，解决系统 CXXABI 版本不足的问题。
IsaacLab 可选手部遥操作/Pink 依赖与 NumPy 1.26 不兼容，本次没有安装这些不参与评测的 extras；不要直接全量升级仿真依赖。

策略环境：`runtime/openpi-venv`，按 vendored OpenPI 的 `uv.lock` 安装，Python 3.11.15、JAX 0.5.3、torch 2.10.0、NumPy 1.26.4。
OpenPI 源码在 `runtime/src/RoboDojo/XPolicyLab/policy/Pi_05/openpi`。
缓存位于 `runtime/local_experiment/openpi_cache`，tokenizer 已下载。
Codex CLI 版本 0.155.0（`~/.local/bin/codex`，注意 PATH 中默认不可见，需显式 `--codex` 或加 PATH）。

**系统级修复（2026-09-19，需 root，重启后需重设）**：
用户 inotify watch 配额 65536 被 Qoder 远程工作器 node 进程占满，导致 Isaac Sim omniclient 线程抛
`std::system_error` 后 abort。已执行 `sudo sysctl fs.inotify.max_user_watches=1048576` 临时生效；
未写入 `/etc/sysctl.d/`（可选持久化）。如重启后仿真再次崩溃并见 `errno=28`，先检查此项。

固定上游版本：
- RoboDojo：`ee67a1468510da7624a089164402359f2afc72c8`
- XPolicyLab（内含 OpenPI）：`432f82b1758c5b1202e42a3dfe014546dbc50871`
- IsaacLab：`afca7b09d60d8beb9c1cb28b43066499940b969b`
- CuRobo：`d17b54ce32cba095c0b000c4c58777075d11de0e`

## 资源与网络

官方 HF 数据集 `RoboDojo-Benchmark/RoboDojo`，revision `91f76c28d93dd20c5fa46ce6a5a1d96a4f384acd`。
已下载 7,373 个文件，共 53,628,141,657 bytes；文件大小全部匹配。冻结源码、60 个布局和所需辅助轨迹通过哈希检查。
没有下载优化器 train_state，也没有下载无关评测布局。
下载目录 `runtime/downloads/robodojo`；源码下 Assets 和 `runtime/checkpoints` 已配置符号链接。
检查点：`runtime/checkpoints/RoboDojo-sim-arx_x5-joint-0/59999`。
配置名：`pi05_base_aloha_full_sim_arx-x5_seed_0`。
检查点内容身份 SHA256：`d15fb8bd1d29cb30b69f01b71c66596cb0293c1a8a94111b343c1580dd3e3e5b`，与报告记录一致；混合 smoke 回合的 policy_identity.json 记录同值前缀，已交叉验证。

代理：实际 Mihomo 监听 `127.0.0.1:7899`（7890 拒绝）。HF 走现有代理；大型 PyPI wheels 直连。

## 认证（已完成）

用户选择复制方案：将日常 `~/.codex/auth.json` 复制到各 work-dir 的
`private/auth_profiles/codex_a/codex_home/auth.json`（chmod 600，非符号链接，通过 `validate_credential`）。
smoke 已证明账户可调用 `gpt-6-astra`。已知代价：实验与日常 CLI 共享同一对 token，
若 OpenAI 轮换 refresh token 可能互踢；实验期间建议不用日常 Codex CLI。
每个新 work-dir 需要重新复制（或 `login` 子命令设备码授权）。
不要提交或展示 auth.json 内容；不要改日常 Codex 配置。

## 已验证边界（截至 2026-09-19）

组件级（此前已通过）：Isaac Sim GPU 启动、原生场景 reset、三路相机、14 维状态、IK 控制步、
π₀.₅ GPU 推理输出有效 50×14 动作、Codex app-server 握手、离线测试。

回合级（2026-09-19 新通过，`build_tower__standard__g0__l0`）：
- Direct（`runtime/repro_smoke`）：完整 1050 步，`complete=true`、原生 `success=false, score=0.1`，
  `artifact_manifest.status=verified`，全部 14 项检查通过（含 paired identity、EEF 动作契约、视频契约）。
- 混合（`runtime/repro_smoke_mix`）：完整 1050 步，`complete=true`、原生 `success=false, score=0.1`，
  `status=verified`，policy_identity 检查点哈希与报告记录一致。
- 该案例公开历史值两方法均为 `success=false, score=0.1`，与本机结果一致（单案例吻合，非整体复现结论）。
- 耗时：Direct 约 103 分钟、混合约 62 分钟（均满步数）；推算 100 回合串行约 140 小时。

2026-09-20 曾两次尝试补跑 `fold_clothes__standard__g0__l0`（Direct + Hybrid），均未成功：
- 第一次：约 140 步后被外部 SIGTERM 终止（`exit 143`）。
- 第二次/第三次：Codex CLI 无法连接 ChatGPT 后端（`wss://chatgpt.com/backend-api/codex/responses` 持续 `Connection reset by peer`），
  Direct 回目在第 0 步即超时失败。问题在同环境可复现，疑似代理/认证/网络策略导致。
- 用户确认单案例 `build_tower` 已足够说明问题，不再继续补跑。
失败现场保留：`runtime/repro_smoke2/results/gpt_only__fold_clothes__standard__g0__l0*`（含 killed、network-fail 后缀，仅作证据，勿并入正式结果）。

**以上仍不是主要实验复现成功。**没有新的成功率/均分，未与 24/50、13/50 等历史值做整体对比。

证据入口：
- `runtime/logs/smoke-direct-detached.log`、`runtime/logs/smoke-hybrid-detached.log`
- `runtime/repro_smoke/`、`runtime/repro_smoke_mix/`（plan.json、results/、summarize 输出）
- 旧证据：`runtime/setup-state.json`、`runtime/logs/doctor.log`、`runtime/checks/*`

## 下一步：正式 100 回合（当前未启动）

用户已确认单案例 `build_tower` 验证足以证明链路，**不继续启动正式 100 回合评测**。
若后续改变主意，可参考以下命令（新 work-dir `runtime/repro_main`，先复制 auth.json，doctor 通过后启动）：

```bash
# 1) 初始化 profile 并复制凭据（同上文路径，chmod 600）
# 2) 预检（两种方法全量参数，同 smoke_mix 的 doctor 参数但 --method both）
# 3) 用 systemd-run 等脱离用户会话的方式启动，避免被终端/工作器清理杀掉：
systemd-run --user --unit=qoder-repro-main.service --service-type=simple \
  --working-directory="$PWD" --property=RemainAfterExit=yes \
  --property=StandardOutput=append:"$PWD/runtime/logs/repro-main-detached.log" \
  --property=StandardError=append:"$PWD/runtime/logs/repro-main-detached.log" \
  --setenv=PATH="$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  python3 -m hybrid_rollout.robodojo.local_experiment run \
    --work-dir "$PWD/runtime/repro_main" \
    --source "$PWD/runtime/src/RoboDojo" \
    --sim-python "$PWD/runtime/sim-python" \
    --openpi-source "$PWD/runtime/src/RoboDojo/XPolicyLab/policy/Pi_05/openpi" \
    --openpi-python "$PWD/runtime/openpi-venv/bin/python" \
    --checkpoint "$PWD/runtime/checkpoints/RoboDojo-sim-arx_x5-joint-0/59999" \
    --sim-gpu 0 --policy-gpu 1 --codex "$HOME/.local/bin/codex"
# 4) 汇总：
python3 -m hybrid_rollout.robodojo.local_experiment summarize --work-dir "$PWD/runtime/repro_main"
```

注意事项：
- **约 140 小时（近 6 天）串行运行，且持续消耗 Codex 额度**；用户关心费用，启动前必须确认。
- 长跑无断点续跑：中途失败/中断后，同 work-dir 因已有归档或 plan 不匹配而拒绝重入；
  现状只能换新 work-dir 从头跑。若用户同意，可先给 runner 增加保守的"跳过已验证完成回合、
  拒绝半途归档"的续跑能力（不改评测语义），再启动长跑；未经同意不要改。
- 保持 `MAX_DECISIONS=0`；失败保留现场，不重试不换案例；汇总只接受身份匹配且归档验证通过的原生结果。
- 本机包装器固定参数后置，不能通过追加 `--work-dir` 覆盖；直接调模块并显式传参。
- 仿真必须用 `runtime/sim-python`；GPU 检查需在有设备权限的终端运行。
- 图形默认 system 模式；容器库才加 `--graphics-mode bundled`。
- 若再次遇到 Codex `Connection reset by peer`/`error sending request`，先检查代理/认证，
  必要时重新 device login 或改用 OpenAI API Key 认证。

后续 AI 应如实区分"环境与链路已验证"与"主要实验已复现"，不要把前者写成后者。
