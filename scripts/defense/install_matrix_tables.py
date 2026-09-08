"""Wrap the tabulars emitted by make_matrix_table.py --tex into the paper's tab/main.tex and
tab/main_quality.tex, with captions that state exactly what was measured.
Usage: python install_matrix_tables.py <rows.tex> <paper_dir>"""
import sys, re, os

rows, pdir = sys.argv[1], sys.argv[2]
txt = open(rows).read()

def solver_cfg_sentence():
    """The configuration the solver returned for the matrix request, read from solve_table_request's
    output so the caption never drifts from the row it describes."""
    import json
    p = "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k/solver_table_config.json"
    try:
        sol = json.load(open(p))["solver"]
    except Exception:
        return "the configuration recorded in solver\_table\_config.json"
    names = {"VINE": "VINE", "TrustMark": "TrustMark", "VideoSeal": "VideoSeal"}
    parts = [f"{names.get(f, f)} at ${sol['s'][f]:.2f}$" for f in sol["order"]]
    fe = [k for k, v in sol.get("fe", {}).items() if v]
    fe_txt = {"angle": "the decode-side angle probe", "resync": "the SyncSeal resynchronizer",
              "scale": "the nested-ring scale search", "tile": "the tiled TrustMark grid"}
    txt_ = " and ".join(parts) if len(parts) <= 2 else ", ".join(parts[:-1]) + " and " + parts[-1]
    if fe: txt_ += " with " + " and ".join(fe_txt.get(k, k) for k in fe)
    return txt_

SOLVER_CFG = solver_cfg_sentence()
tabs = re.findall(r"\\begin\{tabular\}.*?\\end\{tabular\}", txt, re.S)
assert len(tabs) == 2, f"expected 2 tabulars, got {len(tabs)}"
main_tab, qual_tab = tabs

MAIN = r"""\begin{table}[h]
\centering
\caption{\textbf{Main result: robustness at a fixed operating point (Mode M1).} Bit accuracy
($\uparrow$; $0.5$ = payload destroyed) after each attack. Rows are post-hoc watermarks decoded under a
common protocol: each method carries its own native payload (\emph{bits}) and is declared detected at
its own $1\%$-FPR binomial threshold, so the false-positive rate is matched across rows while payload
length is not. Columns follow the RAVEN attack suite (\S\ref{sec:attack-baselines}); every number is
measured by us; none is transcribed from prior work. $N{=}200$ images drawn across the five-source
pool (\S\ref{sec:datasets}); $N{=}30$ for Edit and I2V, $N{=}20$ for CtrlRegen$+$, which runs in a
separate environment. \textbf{Ours} is the composite pinned at one operating point with its
\emph{geometric front-ends disabled}, so every row here is decoded without resynchronization; the
front-ends are part of the library the solver selects over and are evaluated in
\S\ref{sec:solver-eval}. \textbf{Ours (solver)} is what the solver returns for the request ``cover the
fifteen columns of this table that the surrogate measures, at the $1\%$ false-positive budget within
$4$\,s'': @@SOLVER_CFG@@, decoded by the deployed decoder. BM3D, Edit, I2V, CtrlR, UnMk and Rin4 are
reported for the other rows but lie outside that request: the first four are not in the surrogate, UnMk is
measured only for the fixed rows, and Rin4 has no measured reliable-bit capacity, so no request may ask
for a payload there. Best per column in
\textbf{bold}; \colorbox{hl}{ours} highlighted.
Avg is over attacked columns only. Abbrev.: Brgt=brightness, Cont=contrast, Crp$k$=center crop keeping
$k\%$, Rot9=$9^\circ$ rotation, RS256=resize, HFlip=horizontal flip, CrpJ=crop$+$JPEG,
VAE-B/C=neural compression, Rin$k$=rinse-$k\times$, Edit=instruction editing, I2V=image-to-video,
CtrlR=CtrlRegen$+$.}
\label{tab:main}
\resizebox{\textwidth}{!}{%
@@TAB@@%
}
\end{table}
""".replace("@@TAB@@", main_tab).replace("@@SOLVER_CFG@@", SOLVER_CFG)

QUAL = r"""\begin{table}[h]
\centering
\caption{\textbf{Fidelity and mean detection} for the rows of Table~\ref{tab:main}, measured on the
same images. PSNR/SSIM are of the watermarked image against its cover; mean TPR@$1\%$FPR averages
over the attacked columns. The composite buys its robustness with fidelity (it is the lowest-PSNR
row here), which is why M1 alone is not a fair summary of an adaptive scheme and why
\S\ref{sec:solver-eval} reports the fidelity the solver actually delivers per request. Best per column
in \textbf{bold}.}
\label{tab:main-quality}
@@TAB@@
\end{table}
""".replace("@@TAB@@", qual_tab)

open(os.path.join(pdir, "tab/main.tex"), "w").write(MAIN)
open(os.path.join(pdir, "tab/main_quality.tex"), "w").write(QUAL)
print("installed tab/main.tex + tab/main_quality.tex")
