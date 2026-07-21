#!/bin/bash
# One-time environment fixes for running the Olmo pipeline on a fresh RunPod pod
# (2x A100, runpod-torch template). Run once after `python -m venv .venv` and
# `pip install vllm transformers peft huggingface_hub matplotlib numpy ninja`.
#
# Two fixes, both needed for Olmo-3(.1)-32B-Think under vllm 0.25.1 +
# transformers 5.14 (the versions pip resolves as of 2026-07):
set -e
cd /root/subliminal-persona

# 1) ninja must be on the system PATH — vllm's torch.compile spawns a subprocess
#    that calls `ninja` directly (the venv bin dir isn't on the child's PATH).
ln -sf /root/subliminal-persona/.venv/bin/ninja /usr/local/bin/ninja
ninja --version

# 2) vllm 0.25.1's olmo2.py reads a FLAT config.rope_parameters, but
#    transformers 5.14 emits a PER-LAYER-TYPE nested dict
#    ({'sliding_attention': {...}, 'full_attention': {...}}) for Olmo3's
#    yarn-on-full-attention rope. Teach vllm to read both layouts.
.venv/bin/python - <<'PY'
import pathlib
f = pathlib.Path(".venv/lib/python3.10/site-packages/vllm/model_executor/models/olmo2.py")
s = f.read_text()
old = """        if sliding_window is None:
            rope_parameters = self.config.rope_parameters
        else:
            rope_theta = self.config.rope_parameters["rope_theta"]
            rope_parameters = {"rope_type": "default", "rope_theta": rope_theta}"""
new = """        _rp = self.config.rope_parameters
        if isinstance(_rp, dict) and ("full_attention" in _rp or "sliding_attention" in _rp):
            # transformers>=5.x: per-layer-type rope_parameters (Olmo3 yarn-on-full)
            rope_parameters = _rp["full_attention" if sliding_window is None else "sliding_attention"]
        elif sliding_window is None:
            rope_parameters = _rp
        else:
            rope_parameters = {"rope_type": "default", "rope_theta": _rp["rope_theta"]}"""
if new in s:
    print("olmo2.py already patched")
elif old in s:
    f.write_text(s.replace(old, new)); print("patched olmo2.py rope handling")
else:
    raise SystemExit("olmo2.py anchor not found — vllm version changed, re-derive the patch")
PY
echo "POD SETUP DONE"
