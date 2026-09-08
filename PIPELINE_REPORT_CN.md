# 复合密码水印管线报告

> 采用配置:**VINE-R + TrustMark-B 双 fragment**,共享一个 shortened-BCH(100,37,t=10) 码字,per-image 密码学白化,软-LLR 融合,双检测(zero-bit ∨ crypto-ID),几何 resync。
> 代码入口:[composite_external_eval.py](scripts/defense/composite_external_eval.py)。本报告所有参数均经源码核实(2026-06-22)。

---

## 0. 总览

整条管线分三层:**① Encode(嵌入) → ② Channel(攻击/传输) → ③ Decode(检测+识别)**。

![pipeline](results/defense/pipeline_flowchart.png)

设计要点:
- **一个码字、两个互补载体**。VINE-R 把码字嵌进 SD-Turbo 生成潜空间(抗 regen),TrustMark-B 嵌进像素残差(抗几何 crop/rotate)。两者互补、互不干扰。
- **密码学贯穿全程**:每张图、每个 fragment 用独立的置换 σ + 符号掩码 M 把码字"白化"打散;检测端用它做软对齐、resync 的零误接受闸门、以及 2⁻³⁷ 的身份验证。
- **盲检测 + 强身份**:zero-bit 回答"有没有水印",crypto-ID 回答"是谁",二者取或。

---

## 1. ① Encode(嵌入)

代码:[composite_external_eval.py:71-110](scripts/defense/composite_external_eval.py#L71-L110)

### 1.1 码字生成
1. **身份 → 载荷**:`image_id` 经 SHA-256 取前 **37 bit** 作为数据位([payload.py:154](src/payload.py#L154))。
2. **纠错编码**:`ShortenedBCH(100,37,t=10)`([shortened_bch.py](src/shortened_bch.py))。它把 BCH(127,64,t=10) 的数据位 37–63 固定为 0,只传 data[0:37] + 完整 63 位 ECC = **100 位码字 cw**,可纠 **10 个比特翻转**(软判决下有效半径更大)。

### 1.2 密码学白化(每图每 fragment 独立)
对每个 fragment(method ∈ {"vine","trustmark"})派生一对常量([vine_crypto_wrapper.py:32-44](src/vine_crypto_wrapper.py#L32)):

```
salt = "multi_method/{method}/{image_id}"
σ (perm) = Fisher–Yates 洗牌，种子 = HMAC-SHA512(master_key, salt+"/perm") 的 uint64 流
M (mask) = ±1 数组，          种子 = HMAC-SHA512(master_key, salt+"/mask") 的奇偶
```

然后把码字映射到该 fragment 的目标位([vine_crypto_wrapper.py:47](src/vine_crypto_wrapper.py#L47)):

```
target_r = ( 1 − M[r] · (1 − 2·cw[ inv_perm[r] ]) ) / 2          # ∈ {0,1}
```

**作用**:同一码字在两个 fragment 上被打散成两套互不相关的伪随机目标 → **没有系统性弱 bit**(实测 per-position 错误相关 ≈ 0),错误按 fragment 分布而非按 bit。

### 1.3 嵌入与强度
- **VINE-R**:`VINE-R-Enc`(SD-Turbo 潜空间)+ `CustomConvNeXt` 解码器,嵌入分辨率 256;`embed_with_target(target_v)` 产生残差([vine_crypto_wrapper.py:114](src/vine_crypto_wrapper.py#L114))。
- **TrustMark-B**:Adobe TrustMark `model_type="B"`, `use_ECC=False`;`embed_with_target(target_t)` 产生像素残差([trustmark_fragment.py:51](src/trustmark_fragment.py#L51))。
- **强度缩放(默认 α=0.70)**:每个 fragment 的残差按 `out = cover + α·(wm − cover)` 缩放,下一个 fragment 在缩放结果上继续嵌([composite_external_eval.py:49](scripts/defense/composite_external_eval.py#L49))。顺序:VINE 先、TrustMark 后。

> **频带与互补性(实测,见 §4)**:两个 fragment **都是低频**,TrustMark 甚至比 VINE 更低频。它们的互补**不是频带高低**,而是"嵌入域(潜空间 vs 像素加性)+ TrustMark 的几何增广训练"。两者**互不干扰**,叠加只多花 ~1.3 dB 画质。

---

## 2. ② Channel(攻击 / 传输)

这层是对手 + 信道,不是我们的代码。把码字看成过一个有噪信道:
- **regen / rinse**(扩散去噪重生成)主要抹掉 **TrustMark**(像素残差不在扩散先验上),VINE 存活;
- **crop / rotate**(几何)主要打死 **VINE**(潜空间模板对几何敏感),TrustMark 存活。

→ 单类攻击各打掉一个互补 fragment,融合后码字错误通常仍在 t=10 之内。完整攻击集(RAVEN/WAVES):JPEG、blur、noise、brightness、contrast、erasing、resize、crop、rotate、regen、rinse、VAE-b/c、UnMarker、overwrite。

---

## 3. ③ Decode(检测 + 识别)

代码:[composite_external_eval.py:95-139](scripts/defense/composite_external_eval.py#L95) + [soft_fusion.py](src/soft_fusion.py) + [soft_bch.py](src/soft_bch.py)

### 3.1(可选)几何 resync
若基础解码失败,做旋转搜索([raven_resync_regen_eval.py:49-54](scripts/defense/raven_resync_regen_eval.py#L49)):
- 角度搜索 `arange(-30, 30, 3°)` 共 **21 个候选**;
- **每个候选都要 crypto-verify 通过才接受**(精确 37-bit 匹配)。单候选 FPR ≈ 2⁻³⁷,21 个并集 ≈ 2⁻³¹ ≈ 0 → **零误接受**。
- 可救**可逆几何**(旋转/平移/缩放);**救不了 crop**(载体被永久裁掉)。

### 3.2 软输出 → 对齐码字 LLR
- **VINE**:sigmoid 概率 p → `LLR = log(p/(1−p))`([soft_fusion.py:34](src/soft_fusion.py#L34));
- **TrustMark**:原生 logit 直接透传([soft_fusion.py:45](src/soft_fusion.py#L45));
- **软逆密码对齐**:`LLR_codeword[j] = M[σ[j]] · LLR_target[σ[j]]`([soft_fusion.py:86](src/soft_fusion.py#L86)),硬判时等价于 `undo_crypto`。

### 3.3 融合(最大比合并 MRC)
`fuse_llrs`:`L_fused[j] = Σ_m w_m · L_{m,j}`,**当前默认等权 w=1**([soft_fusion.py:136](src/soft_fusion.py#L136))。
> 正在训练一个**上下文门控的学习型融合头**替代等权 MRC(见 §6)。

### 3.4 双检测
| 路径 | 判据 | 回答 |
|---|---|---|
| **zero-bit** | 融合 bit-acc ≥ τ,`τ=(binom.ppf(0.99,100,0.5)+1)/100 = 0.63` | 有没有水印 |
| **crypto-ID** | Chase-II 软 BCH 译码(p=8,256 个翻转模式)→ **精确 37-bit 身份匹配**,FPR ≈ 2⁻³⁷([soft_bch.py:123](src/soft_bch.py#L123)) | 是哪张图 |

**最终判定**:`composite_or = zero-bit ∨ crypto-ID`(若 TrustMark 不作 fused fragment 而走 OR 路径,再 ∨ TrustMark bit-acc≥0.75)。

---

## 4. 密码学的作用(三处)

1. **嵌入端 — 白化**:per-image σ/M 把码字打散到每个 fragment → 无系统性弱 bit,错误按 fragment 而非 bit。
2. **检测端 — 软对齐**:`align_llr_to_codeword` 是 `apply_crypto` 的软逆,让两路 LLR 回到同一逻辑帧再融合。
3. **身份与零误接受**:crypto-ID 的 37-bit 精确匹配给出 2⁻³⁷ 的 FPR;resync 用它当闸门,搜 21 个角度也不误接受。

> 另一安全收益(实测,§4):**反向覆盖攻击**要把水印打到最低需要知道码字才能算反码,而码字被 per-image 白化 → 不知 key 算不出反码 → 密码白化挡住了最优覆盖攻击。

---

## 5. 实测特性小结(本会话)

| 主题 | 关键结论 | 图 |
|---|---|---|
| **频带** | VINE 与 TrustMark **都低频**;TrustMark 97% 能量在 0–0.25 带、更低频。互补靠域+增广不靠频带。 | [tm_vs_vine_freq.png](results/defense/tm_vs_vine_freq.png) |
| **互不干扰** | 复合后两路 bit-acc 各自仍 ≈ solo(掉 0),串扰≈瞎猜;唯一代价 ~1.3 dB 画质。 | — |
| **覆盖律** | 同种水印叠加=后写覆盖,首标保留 = 1 − 汉明距离;**反码覆盖把检出打到 ~0(非 0.5)**。 | [overwrite_hamming.png](results/defense/overwrite_hamming.png) · [complement_residual.png](results/defense/complement_residual.png) |
| **强度–质量** | 默认 α=1 过嵌入;**α=0.70 白送 +2.7 dB(检出全 1.0)**;α=0.50 +5.5 dB(regen 检出 1.0→0.88)。 | [strength_tradeoff.png](results/defense/strength_tradeoff.png) |

### 鲁棒性总览(RAVEN-aligned,SD-2.1 图,复合+resync)
- brightness / contrast / noise / JPEG / erasing:**1.00**(全强度);VAE-b/c:**1.00**;regen / rinse2x:**0.98**;
- resizedcrop:**1.00**(到 rel0.75 / 面积 0.625);仅 scale≤0.5(裁掉≥50% 面积)崩;
- blurring:**1.00**(到半径 10),半径≥15 崩;
- rotation:基线崩(9° 0.14 / 22.5° 0.00)**+resync = 1.00**。
- **唯一真实缺口** = 极端 crop(scale≤0.5,不可逆)。已知对手:UnMarker(用其捆绑的 0.9 crop 杀 VINE + 谱攻击杀 TrustMark)。

---

## 6. 进行中:学习型融合头(方案 A)

**动机**:当前 `fuse_llrs` 是**等权** MRC,但两个 fragment 的可靠度随攻击剧烈变化(regen 信 VINE、crop 信 TrustMark)。等权在一路被打死、且其 LLR 过自信时会被毒化。

**方案**:冻结编码器,训练一个**上下文门控的标定 MRC 头**(置换不变、逐位共享):
```
g = 全局可靠度/一致性上下文(每路统计 + 符号一致率)
w = softplus(gate(g))                                  # 攻击自适应权重
f[j] = w_v · cal_v(LLR_v[j]) + w_t · cal_t(LLR_t[j])    # cal_* = 学到的标定器
```
输出 f 仍是码字 LLR → 照走 Chase-BCH 精确匹配 → **crypto-ID 的 2⁻³⁷ FPR 由构造保留**。

**流程**:① 生成 (对齐 LLR_v, LLR_t, 码字) 数据(embed@0.7 + 攻击增广,180 图×9 攻击)→ ② headroom 探针 → ③ 训练头 → ④ 留出集对比 + FPR 检查。

**结果(test n=495,2026-06-22)—— 有价值的负结果:**

| | 等权 MRC | oracle_w(上界) | **学习头** | FPR |
|---|---|---|---|---|
| 总体检出 | 0.970 | 0.976 | **0.974** | 0.000 |
| regen | 0.95 | 0.98 | 0.96 | — |
| rinse | 0.80 | 0.80 | 0.82 | — |

- **融合不是瓶颈**:oracle 上界仅 0.976,**等权 MRC 已接近最优**。原因:一路被打死时其 LLR≈0(低置信),等权已自动降权 → "自信错误毒化"不显著。
- 学习头拿到大部分(小)空间(0.970→0.974)且 **crypto-ID 零 FPR 由构造保留**(实测未水印图检出 0.000),但绝对收益仅 +0.4 点 → **不值得部署**。
- **真实缺口 = rinse(两路同时被打)→ 0.80,oracle 也才 0.84**:当两个 fragment 都掉到阈值下,任何融合都救不了 → 需更强 fragment 或第三层,不是融合问题。
- 附带:朴素 std 标定(calib)反而更差(0.939),放大死 fragment 噪声。
- 脚本:[fusion_head_gendata.py](scripts/defense/fusion_head_gendata.py) · [fusion_head_probe.py](scripts/defense/fusion_head_probe.py) · [train_fusion_head.py](scripts/defense/train_fusion_head.py)。

**结论:保留等权 MRC + α=0.70 强度调优(更高 ROI);提升目标应对准 rinse/双重打击的 fragment 强度,而非融合层。**

---

## 7. 复现命令

```bash
PY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python
# 嵌入(默认 α=0.70,VINE+TrustMark 双 fragment)
CUDA_VISIBLE_DEVICES=0 $PY scripts/defense/composite_external_eval.py --mode embed \
  --fragments vine trustmark --image_dir <imgs> --image_glob '*.png' \
  --n_images 100 --embed_dir <out>
# 解码(攻击后)
CUDA_VISIBLE_DEVICES=0 $PY scripts/defense/composite_external_eval.py --mode decode \
  --fragments vine trustmark --embed_dir <out> --attacked_dir <attacked> --attack_name <name>
```
