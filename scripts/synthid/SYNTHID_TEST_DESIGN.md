# SynthID 鲁棒性测试 pipeline — 设计文档 + 可执行模板

> 目标:在与我们复合水印**相同的攻击套件**下,评测 Google **SynthID-Image** 的鲁棒性,并(可选)在同一批图上与我们的方案做头对头。
>
> 核心约束(务必先读):**SynthID 不能后处理嵌入、也没有开源检测器**。
> - 嵌入:只能在 **Gemini 2.5 Flash Image / Imagen 生成或编辑**时自动写入(输出必带 SynthID)。
> - 检测:无干净公开批量 API。三条路径:① SynthID Detector 门户 / Gemini App(手动,权威但不可批量),② **Gemini API 问答式检测**(可脚本化,但非官方、需自校准),③ Vertex AI 企业版验证(权威+可批量,但访问受限)。

---

## 0. 关键思路:用 Gemini "最小编辑"把 COCO 变成带 SynthID 的近-COCO 图

把一张 COCO 图喂给 Gemini 2.5 Flash Image,用一个**近乎恒等的编辑指令**(如 "Return this image unchanged")。输出会:
- 走一遍生成管线 → **写入 SynthID**(Google 政策:所有生成/编辑输出必带);
- 像素接近原 COCO(内容几乎不变)。

于是得到"**几乎就是原 COCO 图 + SynthID**",这是最接近"给自己的图打 SynthID"的可行做法。之后按我们现成的攻击套件攻击 + 检测。

---

## 1. Pipeline(5 步)

| 步 | 动作 | 脚本 | 成本 |
|---|---|---|---|
| **1. 生成** | COCO → Gemini 最小编辑 → 带 SynthID 的近-COCO 图 | `synthid_gen.py` | ~$0.02–0.04/图 |
| **2. 检测器自校准(关键)** | 在 clean-SynthID(应 TPR≈1)和非-SynthID 原 COCO(应 FPR≈0)上先测**检测器本身**的准确率 | `synthid_detect.py --calibrate` | 检测费 |
| **3. 攻击** | 复用我们的攻击套件(jpeg/blur/noise/crop/rotate/regen…)本地施加 | `synthid_eval.py`(内含攻击) | 免费 |
| **4. 检测** | 每张攻击图逐一检测 → 每攻击的检出率 | `synthid_eval.py` | ~$0.0005/图 |
| **5. 分析** | 检出-vs-攻击、Q@P;(可选)叠加我们的水印做同像素头对头 | `synthid_eval.py` 输出 JSON | — |

**头对头(可选)**:在步骤 1 的 SynthID 图上,再用我们的 `composite_external_eval.py --mode embed` 叠加复合水印 → 同一批像素同时带两种水印 → 攻击后两个检测器分别判 → 直接可比。

---

## 2. 成本表(已核实定价,2025)

- Gemini 2.5 Flash Image 生成/编辑:**$0.039/图**(输出 1290 token × $30/1M);输入图 ~560 token ≈ **+$0.0011/图**;**Batch API 5 折 → ~$0.0195/图**。
- 检测(Gemini Flash 问答):图 ~250–560 token + 提示 ≈ **~$0.0005/图**。
- GCP 新账户 **$300 试用金** + 通常免费 100–200 张图额度 → 小规模可能 **$0**。

| 规模 | A(攻击点) | 生成(batch) | 检测(N×(A+1)) | **合计** |
|---|---|---|---|---|
| **MVP** N=64 | 18 | 64×$0.02 ≈ $1.3 | 64×19×$0.0005 ≈ $0.6 | **≈ $2**(或 $0,试用金覆盖) |
| 中 N=256 | 20 | ≈ $5 | 256×21×$0.0005 ≈ $2.7 | **≈ $8** |
| 论文级 N=1000 | 20 | ≈ $20 | ≈ $10.5 | **≈ $30** |

**成本公式**:`总价 ≈ N × 生成价 + N × (A+1) × 检测价`(+ 头对头则我们的嵌入/解码本地免费)。

---

## 3. 三个必须盯住的点(比钱更重要)

1. **检测器是真瓶颈,不是钱**。Gemini API 问答式检测是**非官方、黑盒 yes/no、可能不稳/拒答**。所以**步骤 2 的自校准是硬门槛**:若在 clean-SynthID 上 TPR 不接近 1、在原 COCO 上 FPR 不接近 0,则该检测器不可用于定量评测——此时改用官方门户(小 N 手动)或申请 Vertex AI 企业版验证。
2. **是重渲染,非逐像素原图**:内容匹配,像素是 Gemini 的。鲁棒性评测够用;严格逐像素对比做不到。
3. **口径**:SynthID 只测 presence 的检出 P + FPR,没有我们的 2⁻³⁷ 身份维度。

---

## 4. 环境准备

```bash
pip install google-genai pillow    # 官方 SDK
export GEMINI_API_KEY=...           # 从 Google AI Studio 获取;或用 Vertex: 见 synthid_detect.py 注释
# 攻击套件复用本仓库 scripts/wbench/attacks.py(regen 需 diffusers+SD,可选)
```

## 5. 文件清单

| 文件 | 作用 |
|---|---|
| `synthid_gen.py` | COCO → Gemini 最小编辑 → 带 SynthID 图(+ 基线自检) |
| `synthid_detect.py` | 封装检测(Gemini 问答 / Vertex 验证桩);`--calibrate` 自校准 TPR/FPR |
| `synthid_eval.py` | 施加攻击套件 + 逐图检测 + 汇总每攻击检出率 → JSON |
| `run_synthid_mvp.sh` | 一键 MVP:生成 64 → 校准 → 攻击评测 |

> 所有脚本为**模板**:凭据处标 `# TODO`,填入 API key 即可跑。先跑 MVP(N=8–16)验证三步通,再放大。
