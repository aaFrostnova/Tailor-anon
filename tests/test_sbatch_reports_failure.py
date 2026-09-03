"""A crashed measurement must fail its job, not report COMPLETED.

This bug landed twice. `make_surrogate_ext9` died on every one of 100 images and wrote no usable
table, and both times SLURM recorded COMPLETED 0:0 -- so an `afterok` dependency was satisfied and
the canonical table would have been rebuilt from nothing. Two things were needed and only the first
was obvious: `pipefail`, so a python crash behind `| grep` propagates, AND an explicit exit, because
a script's status is that of its LAST command and every one of these ends with `echo DONE`.
"""
import glob, os, re, subprocess, tempfile, pytest

SC = "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k"
SBATCH = sorted(glob.glob(os.path.join(SC, "*.sbatch")))
PIPED = [p for p in SBATCH
         if re.search(r"(/bin/python|^\s*python) [^|\n]*\|", open(p).read(), re.M)]


@pytest.mark.skipif(not PIPED, reason="measurement scripts are not on this machine")
@pytest.mark.parametrize("path", PIPED, ids=lambda p: os.path.basename(p))
def test_a_crashed_python_fails_the_job(path):
    src = open(path).read()
    assert "pipefail" in src, f"{os.path.basename(path)}: a python crash behind a pipe is swallowed"
    assert "PIPESTATUS" in src and re.search(r"^exit \$\{rc", src, re.M), \
        (f"{os.path.basename(path)}: the trailing echo becomes the job's status, so the python "
         f"exit code never reaches SLURM")


def test_the_pattern_actually_propagates():
    """Guard the mechanism itself, not just its presence in the text."""
    script = ("set -u -o pipefail\n"
              "python3 -c 'raise SystemExit(3)' 2>&1 | grep --line-buffered -vE 'Warning'\n"
              "rc=${PIPESTATUS[0]}\n"
              "echo DONE\n"
              "exit ${rc:-0}\n")
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
        fh.write(script); p = fh.name
    try:
        assert subprocess.run(["bash", p], capture_output=True).returncode == 3
    finally:
        os.unlink(p)
