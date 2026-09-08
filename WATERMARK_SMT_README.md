# 把"选水印"变成一个 SMT 问题

> 受 SMT solver 的启发,把"给定用户条件 → 输出最优水印组合"形式化为一个**带约束的优化问题(SMT + Optimize)**,用 Z3 求解。工具:`scripts/defense/watermark_smt_optimizer.py`。

## 思路

用户不再手动按 level 选水印,而是**声明自己的条件**;求解器自动给出满足全部条件的**最优 fragment 组合**;若不存在 → **UNSAT**(恰好对应"这些攻击防不住 = L4")。

## SMT 编码

**决策变量(Boolean,= SAT 层)**:是否启用每个 fragment 与解码端前端
`use_VINE, use_TrustMark, use_VideoSeal, use_resync, use_nested-ring`

**理论约束(线性算术 theory,= presentation 里的 Linear Arithmetic)**:
- **画质**:`PSNR = 38.5 − 1.5·(帧数) − 1.0·nested ≥ 用户 min_psnr`
- **速度**:`time = Σ(嵌入+解码耗时) + resync + nested-search ≤ 用户 max_ms`
- **鲁棒性**:对每个域内攻击 `a`,`OR_f ( use_f ∧ 帧f在阈值下能防a )`
  - "帧 f 能防 a" 用**实测 bit-acc**(frag_suite n=20)+ 前端增益编码:
    - resync:对 TrustMark/VideoSeal 的 `rot9/rot30/crop75` 恢复到 ~0.96
    - nested-ring:对 VINE 的 `crop90/crop75` 恢复到 ~0.96(尺度搜索到 crop50)
- **结构**:`nested ⟹ use_VINE`;至少 1 帧

**目标(Optimize / νZ,字典序)**:`min 帧数 → max PSNR → min 耗时`
→ "最优 = 满足条件的**最小、最高画质、最快**组合"

**SAT** → 输出最优组合;**UNSAT** → 无组合能满足(诚实的"防不住")

## 用法

```bash
python scripts/defense/watermark_smt_optimizer.py \
    --min_psnr 34 --max_ms 3000 \
    --attacks jpeg25 crop75 rot30 regen --min_ba 0.90
# → VINE + TrustMark + resync (VINE 管 jpeg25/regen,TM+resync 管 crop75/rot30)
```
无参数则跑 7 个内置示例。`min_ba`:0.90 = 密码身份验证,0.63 = presence 检测。

## 示例结果(求解器自动得出)

| 用户条件 | 求解器输出 |
|---|---|
| 证件照:仅信号,身份 0.90,高画质 | **VINE 单帧**(37dB)— TM 的 jpeg25=0.85 达不到 0.90 |
| 社交:信号+几何,身份 0.90 | **VINE + TrustMark + resync**(35.5dB) |
| 网络/AI:+再生,身份 0.90 | **UNSAT** — rinse 0.80 < 0.90(帧容量天花板) |
| 网络/AI:+再生,presence 0.63 | **VINE + VideoSeal + resync**(2 帧) |
| 快而小:200ms,presence | **TrustMark 单帧**(100ms)— VINE 太慢被排除 |
| 对抗:+CtrlRegen+ s0.9 | **UNSAT** — 防不住(= L4 天花板) |
| UnMarker + CtrlRegen+ s0.5,presence | **VINE 单帧** — 两者都防住 |

## 求解器揭示的洞察(不是人工规则,是约束推出来的)

1. **jpeg25 要身份验证 → 强制上 VINE**(TrustMark 0.85 不够 0.90)。
2. **rinse 在身份 0.90 下 → UNSAT**(VINE 独扛 0.80 是帧容量上限);降到 presence 0.63 → 可解。
3. **CtrlRegen+ s0.9 → UNSAT**(= L4 无解),而 **UnMarker → 可解**(VINE 0.99)。
4. **速度预算排除慢的 VINE** → 逼出像素帧。
5. **resync 让几何"免费"** → 不必为旋转单加一帧。

## 与 SMT presentation 的对应

| presentation 概念 | 本问题 |
|---|---|
| SAT(布尔可满足) | fragment 选择的布尔组合 |
| Theory(线性算术) | PSNR / 速度 的数值约束 |
| Theory solver | 实测 bit-acc + 前端增益的鲁棒性判定 |
| Optimization(νZ) | 求满足约束的最小/最优组合 |
| **UNSAT** | **"这些攻击在此预算下防不住"**(L4) |

## 数据来源与边界
- bit-acc:`frag_suite.log`(TrustMark-B / VINE-R,n=20);L4 复合体 `ctrlregen1000_*.json`(n=1000)、`unmarker100_*.json`(n=100)。VideoSeal 的信号/crop 值部分为估计(其实测轴是旋转 rot9=0.83),标注在 `BA` 表注释。
- PSNR/速度为**标定的成本模型**(锚点:1帧~37dB、2帧~35.5、3帧~34;VINE≈2.2s、像素帧≈0.1–0.2s),非逐配置实测 —— 想更精确可把每配置的实测 PSNR/耗时填进模型。
- 鲁棒性模型 = **best-path / 任一帧存活**(部署的双检测器);融合头在此之上只兜稀释,不改可行性判定。
