"""
Overlay image+mask pairs for a merged nnUNet dataset, to check (image, mask) ALIGNMENT.

diag_cohorts.py showed SmartHealth and DirVsAverages are statistically identical (same channel
scales, labels, shapes, foreground %). Yet Dataset102 training collapses. A flip / transpose /
offset between the DirVsAverages images and their masks would be INVISIBLE to those stats (same
intensities, same label, same blob size -- just in the wrong place) but would wreck training.

This renders a montage: rows of [ch0 image | mask | overlay] for a sample of DirVsAvg cases and
a few SmartHealth cases for reference. The LV ring must sit ON the myocardium in the overlay.

Also prints a quantitative flip/transpose proxy: mean ch0 intensity inside the mask vs inside the
mask flipped up-down / left-right / transposed. If a FLIPPED mask matches the image better than the
given mask, the masks are misaligned.

Usage:  python diag_overlay.py [DATASET_ID]     (default 102)
Writes: <dataset>/diag_overlay_alignment.png
"""
import os, glob, re, sys
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RAW = '/home/sastocke/nnUNet/nnUNet_raw'
dataset_id = int(sys.argv[1]) if len(sys.argv) > 1 else 102


def cohort(name):
    n = name.lower()
    return 'DirVsAvg' if ('dirvsavg' in n or 'dirvsaverages' in n) else 'SmartHealth'


def alignment_score(img, mask):
    """Mean ch0 intensity inside the ring, for the mask and 3 misaligned variants.
    The myocardium ring is mid-intensity; a well-aligned mask should NOT score like a
    random shift. If a flipped/transposed mask scores clearly differently, alignment is off."""
    fg = mask > 0
    if fg.sum() == 0:
        return {}
    base = img[fg].mean()
    return {
        'as-is':      base,
        'flip_ud':    img[np.flipud(fg)].mean(),
        'flip_lr':    img[np.fliplr(fg)].mean(),
        'transpose':  img[fg.T].mean() if fg.shape[0] == fg.shape[1] else float('nan'),
    }


def main():
    D = glob.glob(f'{RAW}/Dataset{dataset_id:03d}_*')[0]
    IMG, LAB = f'{D}/imagesTr', f'{D}/labelsTr'
    cases = sorted({re.sub(r'_\d{4}\.nii\.gz$', '', os.path.basename(f)) for f in glob.glob(f'{IMG}/*.nii.gz')})
    groups = {'SmartHealth': [], 'DirVsAvg': []}
    for c in cases:
        groups[cohort(c)].append(c)

    pick = groups['DirVsAvg'][::max(1, len(groups['DirVsAvg']) // 8)][:8] + \
           groups['SmartHealth'][::max(1, len(groups['SmartHealth']) // 4)][:4]

    fig, ax = plt.subplots(len(pick), 3, figsize=(9, 3 * len(pick)))
    print('per-case ch0 intensity inside mask vs flipped variants (as-is should be the "intended" one):')
    for r, c in enumerate(pick):
        img = np.asanyarray(nib.load(f'{IMG}/{c}_0000.nii.gz').dataobj).squeeze().astype(float)
        m = np.asanyarray(nib.load(f'{LAB}/{c}.nii.gz').dataobj).squeeze()
        sc = alignment_score(img, m)
        print(f'  [{cohort(c):11s}] {c[:48]:48s} '
              + '  '.join(f'{k}={v:.3f}' for k, v in sc.items()))
        ax[r, 0].imshow(img, cmap='gray'); ax[r, 0].set_ylabel(cohort(c), fontsize=8)
        ax[r, 1].imshow(m, cmap='gray')
        ax[r, 2].imshow(img, cmap='gray')
        ax[r, 2].imshow(np.ma.masked_where(m == 0, m), cmap='autumn', alpha=0.5)
        for c2 in range(3):
            ax[r, c2].set_xticks([]); ax[r, c2].set_yticks([])
        ax[r, 0].set_title(c.replace('DirVsAvgHannum_', 'DVA_').replace('Hannum_Volunteer_', 'V')[:34], fontsize=7)
    ax[0, 0].set_title('ch0 image', fontsize=9)
    out = f'{D}/diag_overlay_alignment.png'
    plt.tight_layout(); plt.savefig(out, dpi=110); print(f'\nwrote {out}')
    print('CHECK: in the "overlay" (3rd) column, the red ring must sit on the myocardium for BOTH '
          'cohorts. If DirVsAvg rings are off the heart / flipped, that is the bug.')


if __name__ == '__main__':
    main()
