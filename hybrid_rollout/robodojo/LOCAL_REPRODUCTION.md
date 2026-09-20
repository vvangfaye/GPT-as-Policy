# 本地复现主要 RoboDojo 实验

此入口复用发布版控制器、GPT 工具契约、π₀.₅ 服务和原生评分器。
完整实验是 **50 个固定案例 × 两种方法 = 100 回合**，不是训练模型。
`gpt_only` 是 GPT 直接输出 EEF 动作；`pi05_plus_gpt` 是 π₀.₅ 提议、GPT 审核与修正。
固定模型为 `gpt-6-astra`，推理强度为 `xhigh`，不会自动换模型。
RoboLab 补充实验和官方 leaderboard 的非配对参考值不属于这个入口的执行范围。

## 先检查计划和公开参考值（不需要 GPU 或登录）

在仓库根目录执行：

```sh
python3 -m hybrid_rollout.robodojo.local_experiment plan
python3 -m hybrid_rollout.robodojo.local_experiment published
python3 -m unittest hybrid_rollout.robodojo.test_local_experiment -v
```

案例来自 `public_results/evaluation_cases.json`，并与发布版冻结 panel 校验种子和身份。
三个泛化任务选用 2 个标准场景和 3 个随机场景；不会误取 panel 的前五项。
`published` 仅重新计算已发布数据，输出带有 `published_reference_not_new_experiments` 标签。
预期：混合策略 24/50 成功、平均分 62.60；直接策略 13/50 成功、48 个有效分数平均 37.8125。

## 外部环境

需要自行准备发布版 README / `SOURCE.json` 指定的 RoboDojo 源码和 Assets、Isaac Sim 5.1 环境。
混合策略还需要 OpenPI/JAX 环境和 RoboDojo π₀.₅ checkpoint（包括 params 和 norm_stats.json）。
本仓库不包含仿真资源或权重。冻结 panel 会校验源码 commit、源码内容及场景哈希；不匹配会停止。
Isaac Sim Python 需要项目运行依赖；Python 3.10 还需 `tomli`。
`doctor` 是预检，不替代真实渲染、模型权限或 checkpoint 加载验证。

下面路径为占位值，替换为实际安装路径。可将 bash 参数数组保存到自己的脚本：

```bash
COMMON=(
  --work-dir "$PWD/runtime/repro_main"
  --source /absolute/path/to/RoboDojo
  --sim-python /absolute/path/to/isaacsim-env/bin/python
  --openpi-source /absolute/path/to/openpi
  --openpi-python /absolute/path/to/openpi-env/bin/python
  --checkpoint /absolute/path/to/checkpoint/59999
  --sim-gpu 0 --policy-gpu 1
)
python3 -m hybrid_rollout.robodojo.local_experiment doctor "${COMMON[@]}"
```

工作站默认用系统 NVIDIA/Vulkan 库。需要发布版容器图形库覆盖时，加 `--graphics-mode bundled`。
默认两个进程都使用 GPU 0；上述示例显式分配到两张卡。GPU 检查应在有设备访问权限的终端中运行。

## 本机 Codex

自动查找 PATH 中的 `codex`，也可用 `--codex /absolute/path/to/codex`。
使用内置 OpenAI provider，不使用发布版的占位 gateway。
实验登录保存在 `--work-dir/private/auth_profiles/codex_a/codex_home`，复用原有账号锁防止并发刷新。
不复制日常 Codex 的 OAuth token，不改日常配置；使用同一账号完成一次独立设备登录：

```bash
python3 -m hybrid_rollout.robodojo.local_experiment login "${COMMON[@]}"
```

登录需要本人完成终端显示的设备授权；不需要将密钥粘贴到代码里。
这是 [Codex 官方支持的认证方式](https://developers.openai.com/codex/auth)。
账号是否能调用指定模型需通过实际 smoke run 验证。

## 先跑一个案例，再跑完整计划

先用独立目录跑一个完整 Direct 回合，不需要 OpenPI 或 checkpoint：

```bash
python3 -m hybrid_rollout.robodojo.local_experiment login --work-dir "$PWD/runtime/repro_smoke"
python3 -m hybrid_rollout.robodojo.local_experiment run \
  --work-dir "$PWD/runtime/repro_smoke" \
  --source /absolute/path/to/RoboDojo \
  --sim-python /absolute/path/to/isaacsim-env/bin/python \
  --method gpt_only --case-id build_tower__standard__g0__l0
```

完整执行（会使用 GPU 和 Codex 额度，可能耗时较长）：

```bash
python3 -m hybrid_rollout.robodojo.local_experiment run "${COMMON[@]}"
python3 -m hybrid_rollout.robodojo.local_experiment summarize --work-dir "$PWD/runtime/repro_main"
```

也可以 `--method gpt_only` 或 `--method pi05_plus_gpt`，以及 `--task build_tower` 选择子集。
保持 `MAX_DECISIONS=0`，沿用原生任务的步数上限，不用人为决策数提前截断。
每次只运行一个回合。服务或归档失败时停止，保留现场，不重试抽取更好的结果。
已有同名回合或不同计划时拒绝覆盖；重新实验用新的 work-dir 并重新登录。
本入口暂不支持中断续跑，失败前已完成的结果可用 summarize 查看。

结果存于 `work-dir/results/<method>__<case_id>/`，含日志、轨迹、视频、身份和校验清单。
只有原生完成、身份匹配且归档验证通过的结果才参与汇总。
缺失结果不填零、不当作普通失败，成功率会显示 null；分数均值同时显示有效样本数。
汇总为本地原生结果；不自动重用报告中的人工超时裁决。

## 当前验证边界

本地工具的离线测试可在无仿真环境下运行。完整 GPU/模型实验必须在外部依赖准备好后验证；
生成计划或重算公开分数不表示已完成实验复现。
