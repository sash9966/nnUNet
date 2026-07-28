"""
Why does ResEnc-M collapse on the combined dataset (102) but the standard UNet trains fine,
and ResEnc-M is fine on SmartHealth-only (100)?  -> it's the ResEnc PLANS, not the data.

nnUNet's ResEnc planner sizes patch/batch/target-spacing to a GPU-memory budget FROM the
dataset fingerprint. If the combined dataset's fingerprint differs (esp. spacing), the planner
can pick a very different patch/batch -> unstable training (small batch + residual encoder +
poly-LR = the rise-then-collapse curve). The standard planner is more conservative.

This dumps, side by side:
  - per-cohort nifti SPACING (zooms) + affine  (fingerprint input; diag_cohorts only checked shape)
  - key plan fields (patch_size, batch_size, target spacing, features/stages) for:
       Dataset102 ResEnc-M   vs   Dataset100 ResEnc-M   vs   Dataset102 standard nnUNetPlans
  - dataset.json normalization scheme per dataset

Usage:  python diag_plans.py
"""
import os, glob, json, re
import numpy as np
import nibabel as nib

RAW  = '/home/sastocke/nnUNet/nnUNet_raw'
PREP = '/home/sastocke/nnUNet/nnUNet_preprocessed'


def cohort(n):
    n = n.lower()
    return 'DirVsAvg' if ('dirvsavg' in n or 'dirvsaverages' in n) else 'SmartHealth'


def spacing_report(dataset_id):
    D = glob.glob(f'{RAW}/Dataset{dataset_id:03d}_*')
    if not D:
        print(f'  (no raw Dataset{dataset_id:03d})'); return
    IMG = f'{D[0]}/imagesTr'
    cases = sorted({re.sub(r'_\d{4}\.nii\.gz$', '', os.path.basename(f)) for f in glob.glob(f'{IMG}/*.nii.gz')})
    groups = {}
    for c in cases:
        groups.setdefault(cohort(c), []).append(c)
    print(f'  Dataset{dataset_id:03d}: {os.path.basename(D[0])}')
    for coh, cs in groups.items():
        zs = set()
        for c in cs[:60]:
            h = nib.load(f'{IMG}/{c}_0000.nii.gz').header
            zs.add(tuple(np.round(h.get_zooms(), 4)))
        print(f'    {coh:11s}: zooms seen = {sorted(zs)}')


def load_plans(dataset_id, plans_name):
    p = glob.glob(f'{PREP}/Dataset{dataset_id:03d}_*/{plans_name}.json')
    return json.load(open(p[0])) if p else None


def plan_summary(tag, plans):
    if plans is None:
        print(f'  [{tag}] (plans not found)'); return
    cfg = plans.get('configurations', {}).get('2d', {})
    arch = cfg.get('architecture', {}).get('arch_kwargs', {})
    print(f'  [{tag}]')
    print(f'      patch_size    : {cfg.get("patch_size")}')
    print(f'      batch_size    : {cfg.get("batch_size")}')
    print(f'      spacing       : {cfg.get("spacing")}')
    print(f'      median shape  : {cfg.get("median_image_size_in_voxels")}')
    print(f'      normalization : {cfg.get("normalization_schemes")}')
    print(f'      n_stages      : {arch.get("n_stages")}')
    print(f'      features/stage: {arch.get("features_per_stage")}')


def dataset_json_norm(dataset_id):
    p = glob.glob(f'{RAW}/Dataset{dataset_id:03d}_*/dataset.json')
    if not p:
        print(f'  (no dataset.json for {dataset_id:03d})'); return
    dj = json.load(open(p[0]))
    print(f'  Dataset{dataset_id:03d} channel_names: {dj.get("channel_names")}  labels: {dj.get("labels")}')


print('=' * 70)
print('1) PER-COHORT SPACING (fingerprint input -- differing spacing shifts the plans)')
spacing_report(100)
spacing_report(102)

print('\n' + '=' * 70)
print('2) PLANS side by side (patch/batch/spacing drive training stability)')
plan_summary('102 ResEnc-M', load_plans(102, 'nnUNetResEncUNetMPlans'))
plan_summary('100 ResEnc-M (good, 0.89)', load_plans(100, 'nnUNetResEncUNetMPlans'))
plan_summary('102 standard nnUNetPlans (trained fine)', load_plans(102, 'nnUNetPlans'))

print('\n' + '=' * 70)
print('3) dataset.json normalization (should be noNorm x4, same across datasets)')
dataset_json_norm(100)
dataset_json_norm(102)

print('\n' + '=' * 70)
print('READ: if 102 ResEnc-M has a much SMALLER batch_size or LARGER patch_size than 100 '
      'ResEnc-M / the 102 standard plans, that is the instability source. Also check whether '
      '102 spacing/median-shape differs (driven by the per-cohort zooms above).')
