FasterMoE: Train MoE Models Faster
===

This repository is the open-source codebase of the PPoPP'22 paper, _FasterMoE:
Modeling and Optimizing Training of Large-Scale Dynamic Pre-Trained Models_. It
is a prototype to verify the ideas in the paper. Based on
[FastMoE](https://github.com/laekov/fastmoe), the hard-coding and ad-hoc
modifications when we were working on the paper are preserved as they were in
this repository.  We have already released a clean and elegant version that is
merged into FastMoE's `v1.0.0` release.

If you want to try this prototype, refer to  [FastMoE's
README](FastMoE-README.md) for installation guide. 

Dynamic shadowing is enabled by environment variable `FMOE_ENABLE_DYNREP=1`,
and the related code can be found in `fmoe/transformer.py`.

The smart schedule is enabled by environment variable `FMOE_ENABLE_FUSE=1`.

The topology-aware gate is in `fmoe/gates/hir_gate.py`. You may use it as a
customized gate in _FastMoE_.

Additionally, the _Artifical Evaluation_ package is located in
<https://zenodo.org/record/5728493#.YaBlGyURVBU>, which contains a copy of this
repo, as well as scripts to reproduce the experiments in the paper.
