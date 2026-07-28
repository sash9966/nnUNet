"""
Cohort-consistency diagnostic for a MERGED nnUNet dataset (SmartHealth + DirVsAverages).

Why: adding DirVsAverages to the LV dataset (Dataset102) made training deteriorate --
pseudo-dice rises then collapses while train/val LOSS look fine. With validation kept pure
SmartHealth (see specific_split_combined.py), that signature means the DirVsAverages TRAINING
data is inconsistent with SmartHealth, so the shared weights drift and SmartHealth val Dice drops.

This project uses nnUNet **noNorm** (normalization is baked into the prep script), so the #1
suspect is a per-channel intensity-scale mismatch between the two cohorts. Also checks label
values, foreground fraction, and shapes.

Usage:  python diag_cohorts.py [DATASET_ID]     (default 102)
"""
import os, glob, re, sys
import numpy as np
import nibabel as nib

RAW = '/home/sastocke/nnUNet/nnUNet_raw'
dataset_id = int(sys.argv[1]) if len(sys.argv) > 1 else 102


def cohort(name):
    n = name.lower()
    return 'DirVsAvg' if ('dirvsavg' in n or 'dirvsaverages' in n) else 'SmartHealth'


def main():
    matches = glob.glob(f'{RAW}/Dataset{dataset_id:03d}_*')
    assert matches, f'no {RAW}/Dataset{dataset_id:03d}_* found'
    D = matches[0]
    IMG, LAB = f'{D}/imagesTr', f'{D}/labelsTr'
    print(f'dataset: {os.path.basename(D)}')

    n_ch = len(glob.glob(f'{IMG}/{os.path.basename(glob.glob(f"{LAB}/*.nii.gz")[0])[:-7]}_*.nii.gz'))
    cases = sorted({re.sub(r'_\d{4}\.nii\.gz$', '', os.path.basename(f)) for f in glob.glob(f'{IMG}/*.nii.gz')})
    groups = {'SmartHealth': [], 'DirVsAvg': []}
    for c in cases:
        groups[cohort(c)].append(c)
    print('cases per cohort:', {k: len(v) for k, v in groups.items()}, f'| channels: {n_ch}\n')

    per_cohort_stats = {}
    for coh, cs in groups.items():
        if not cs:
            print(f'===== {coh}: (none) =====\n'); continue
        print(f'===== {coh}  ({len(cs)} cases) =====')
        chan_means = {}
        for ch in range(n_ch):
            mins, maxs, means = [], [], []
            for c in cs[:80]:
                p = f'{IMG}/{c}_{ch:04d}.nii.gz'
                if not os.path.exists(p):
                    continue
                d = np.asanyarray(nib.load(p).dataobj).astype(float)
                mins.append(d.min()); maxs.append(d.max()); means.append(d.mean())
            if means:
                chan_means[ch] = (np.mean(mins), np.mean(maxs), np.mean(means))
                print(f'  ch{ch}: min~{np.mean(mins):+8.3f}  max~{np.mean(maxs):+8.3f}  mean~{np.mean(means):+8.3f}')
        lv, fg, shapes = set(), [], set()
        for c in cs[:80]:
            m = np.asanyarray(nib.load(f'{LAB}/{c}.nii.gz').dataobj)
            lv.update(np.unique(m).tolist()); fg.append((m > 0).mean()); shapes.add(m.shape)
        print(f'  label values : {sorted(lv)}')
        print(f'  foreground % : mean {np.mean(fg)*100:.2f}  (min {np.min(fg)*100:.2f}, max {np.max(fg)*100:.2f})')
        print(f'  mask shapes  : {shapes}\n')
        per_cohort_stats[coh] = {'chan': chan_means, 'labels': sorted(lv), 'fg': np.mean(fg), 'shapes': shapes}

    # ---- verdict ----
    if len(per_cohort_stats) == 2:
        sh, dv = per_cohort_stats['SmartHealth'], per_cohort_stats['DirVsAvg']
        print('=' * 60)
        flags = []
        for ch in sh['chan']:
            if ch in dv['chan']:
                s_mean, d_mean = sh['chan'][ch][2], dv['chan'][ch][2]
                s_max, d_max = sh['chan'][ch][1], dv['chan'][ch][1]
                # flag if scales differ by >2x
                if max(abs(s_mean), abs(d_mean)) > 1e-6 and (
                        abs(s_mean - d_mean) / (abs(s_mean) + 1e-6) > 0.5 or
                        max(s_max, d_max) / (min(s_max, d_max) + 1e-6) > 2):
                    flags.append(f'ch{ch}: SmartHealth mean/max {s_mean:.2f}/{s_max:.2f} vs DirVsAvg {d_mean:.2f}/{d_max:.2f}')
        if flags:
            print('NORMALIZATION MISMATCH (noNorm -> this breaks training):')
            for f in flags:
                print('   ' + f)
            print('  -> re-prep DirVsAverages with the SAME per-channel normalization as SmartHealth.')
        if sh['labels'] != dv['labels']:
            print(f'LABEL MISMATCH: SmartHealth {sh["labels"]} vs DirVsAvg {dv["labels"]}  -> remap DirVsAvg labels.')
        if sh['shapes'] != dv['shapes'] or len(sh['shapes']) > 1 or len(dv['shapes']) > 1:
            print(f'SHAPE MISMATCH: SmartHealth {sh["shapes"]} vs DirVsAvg {dv["shapes"]}.')
        if abs(sh['fg'] - dv['fg']) / (sh['fg'] + 1e-6) > 0.5:
            print(f'FOREGROUND-FRACTION MISMATCH: SmartHealth {sh["fg"]*100:.2f}% vs DirVsAvg {dv["fg"]*100:.2f}% '
                  '(different crop/FoV or empty masks).')
        if not flags and sh['labels'] == dv['labels'] and sh['shapes'] == dv['shapes']:
            print('No obvious cohort mismatch in intensity/labels/shape -- look at mask ALIGNMENT '
                  '(overlay a few DirVsAvg image+mask pairs) and spacing in dataset.json/plans.')
        print('=' * 60)


if __name__ == '__main__':
    main()
