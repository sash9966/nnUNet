"""
Parameterized cDTI segmentation prep (LV or IP), built DIRECTLY from the ORIGINAL data folders.

Single-cohort OR combined: it loops over the selected cohorts' raw folders and writes every case
into ONE output dataset, so all cases -- SmartHealth and DirVsAverages -- pass through the SAME
normalization code path. This is the fix for the fused-dataset normalization mismatch: nothing is
copied from an already-built dataset, so the two halves cannot disagree.

All 4 contrasts are used (avg/MD/eigenvector/FA). Each cohort keeps its own slice count via
dynamic discovery (DirVsAverages ~3 slices/DiVO, SmartHealth ~9+).

Set TASK, COHORTS, OUTPUT_ID/NAME below, then:  python prep_segmentation.py
Produce, e.g.:
    ('SmartHealth',)                 LV->100  IP->105
    ('SmartHealth','DirVsAverages')  LV->102  IP->107   (combined, from source)
"""
import os
import glob
import re
import json
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt

# ============================ config ============================
TASK    = 'LV'                                  # 'LV' or 'IP'
COHORTS = ['SmartHealth', 'DirVsAverages']      # 1 cohort = single dataset; both = combined
OUTPUT_ID   = 102
OUTPUT_NAME = 'HannumSmartHealthandDirVsAvgs'

PAPER = '/Users/saschastocker/Documents/Stanford/DanEnnis20242025/Paper2025Automatic'
COHORTS_SRC = {
    'SmartHealth':   dict(pwd=f'{PAPER}/Smart_Health',  root='Hannum'),
    'DirVsAverages': dict(pwd=f'{PAPER}/DirVsAverages',  root='DirVsAveragesHannum'),
}

OUT_BASE = COHORTS_SRC[COHORTS[0]]['pwd']       # datasets staged under the first cohort's folder
datasetname = f'Dataset{OUTPUT_ID:03d}_{OUTPUT_NAME}'
output_image_folder = f'{OUT_BASE}/{datasetname}/imagesTr'
output_mask_folder  = f'{OUT_BASE}/{datasetname}/labelsTr'
inspection_folder   = f'{OUT_BASE}/inspection{datasetname}'
for d in (output_image_folder, output_mask_folder, inspection_folder):
    os.makedirs(d, exist_ok=True)


# ============ normalization -- copied verbatim from NewFALVonly.py (single source of truth) ============
def normalize_image(image):
    image_min = np.min(image)
    image_max = np.max(image)
    return (image - image_min) / (image_max - image_min)

def normalise_MD(image):
    image_min = 0
    image_max = 4
    return (image - image_min) / (image_max - image_min)

def normalise_eigenvector(image):
    # X and Y are components of a UNIT eigenvector, so their sum is capped at sqrt(2) (~1.414),
    # not 2. Divide by that fixed ceiling instead of min-max (stable; every image reaches sqrt(2)).
    return image / np.sqrt(2)


def process_mask_slices(mask_data, task):
    """3-channel GT (LV, IP1, IP2) -> single-label mask. LV: {LV:1}. IP: {IP1:1, IP2:2}."""
    combined = np.zeros((256, 256), dtype=np.uint8)
    if mask_data.shape[0] == 256 and mask_data.shape[-1] == 3:
        mask_data = np.transpose(mask_data, (2, 0, 1))
    if task == 'LV':
        combined[mask_data[0, :, :] == 1] = 1
    else:  # IP  (same label map as NewFAIPOnly.py: IP1=1, IP2=2)
        combined[mask_data[1, :, :] == 1] = 1
        combined[mask_data[2, :, :] == 1] = 2
    return combined


def save_inspection_plots(image_data, mask_data, base):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(image_data[:, :, 0], cmap='gray'); ax[0].set_title('Average Diffusion'); ax[0].axis('off')
    ax[1].imshow(mask_data, cmap='gray'); ax[1].set_title('Mask'); ax[1].axis('off')
    ax[2].imshow(image_data[:, :, 0], cmap='gray')
    ax[2].imshow(np.ma.masked_where(mask_data == 0, mask_data), cmap='autumn', alpha=0.5)
    ax[2].set_title('Overlay'); ax[2].axis('off')
    plt.tight_layout(); plt.savefig(os.path.join(inspection_folder, f'{base}_inspection.png')); plt.close(fig)


# ============================ build (loop cohorts -> one dataset) ============================
count = 0
for cohort in COHORTS:
    src = COHORTS_SRC[cohort]
    root_folder, root_path = src['root'], os.path.join(src['pwd'], src['root'])
    print(f'\n=== cohort {cohort}: {root_path} ===')
    if not os.path.isdir(root_path):
        print(f'  !! missing source folder: {root_path}'); continue

    for volunteer_folder in sorted(os.listdir(root_path)):
        if not volunteer_folder.startswith('Volunteer'):
            continue
        dc = os.path.join(root_path, volunteer_folder, 'Distortion_Corrected')
        if not os.path.isdir(dc):
            continue
        for divo_folder in sorted(os.listdir(dc)):
            if not (divo_folder.startswith('DiVO') or divo_folder.startswith('MDDW')):
                continue
            divo_path = os.path.join(dc, divo_folder)
            excel_path = os.path.join(divo_path, 'Detailed_Information.xlsx')
            if not os.path.exists(excel_path):
                print(f'  missing quality info: {divo_folder}'); continue
            quality = pd.read_excel(excel_path)

            mask_folder  = os.path.join(divo_path, '06_Segmentation_Masks_CI')
            image_folder = os.path.join(divo_path, '05_Segmentation_Images_CI')
            slice_files = glob.glob(os.path.join(mask_folder, 'Cropped_Segmentation_Slice_*.nii'))
            slices = sorted(int(re.search(r'Slice_(\d+)\.nii$', f).group(1)) for f in slice_files)
            print(f'  {volunteer_folder}/{divo_folder}: {len(slices)} slices -> {slices}')

            for i in slices:
                row = quality[quality['Slice Number'] == i]
                if row.empty or row.iloc[0]['Image Quality'] != 'Good Image Quality':
                    print(f'    skip slice {i} (bad quality)'); continue

                mask_file = os.path.join(mask_folder, f'Cropped_Segmentation_Slice_{i:03d}.nii')
                avg_f = os.path.join(image_folder, f'Cropped_Average_Diffusion_Weighted_Image_Slice_{i:03d}.nii')
                md_f  = os.path.join(image_folder, f'Cropped_Mean_Diffusivty_Image_Slice_{i:03d}.nii')
                eig_f = os.path.join(image_folder, f'Cropped_Primary_Eigenvector_Image_Slice_{i:03d}.nii')
                fa_f  = os.path.join(image_folder, f'Cropped_Fractional_Anisotropy_Image_Slice_{i:03d}.nii')
                if not all(os.path.exists(f) for f in (mask_file, avg_f, md_f, eig_f, fa_f)):
                    print(f'    missing required files for slice {i}'); continue

                mask_img = nib.load(mask_file)
                combined_mask = process_mask_slices(mask_img.get_fdata(), TASK)

                avg_img, md_img, eig_img, fa_img = (nib.load(avg_f), nib.load(md_f), nib.load(eig_f), nib.load(fa_f))
                avg = normalize_image(avg_img.get_fdata())
                md  = normalise_MD(md_img.get_fdata())
                eig = eig_img.get_fdata()
                fa  = fa_img.get_fdata()                                      # already [0,1]
                eig_c = normalise_eigenvector(eig[:, :, 0] + eig[:, :, 1])

                cid = f'{root_folder}_{volunteer_folder}_{divo_folder}_slice_{i:03d}'
                nib.save(nib.Nifti1Image(avg,   avg_img.affine), os.path.join(output_image_folder, f'{cid}_0000.nii.gz'))
                nib.save(nib.Nifti1Image(md,    md_img.affine),  os.path.join(output_image_folder, f'{cid}_0001.nii.gz'))
                nib.save(nib.Nifti1Image(eig_c, eig_img.affine), os.path.join(output_image_folder, f'{cid}_0002.nii.gz'))
                nib.save(nib.Nifti1Image(fa,    fa_img.affine),  os.path.join(output_image_folder, f'{cid}_0003.nii.gz'))
                nib.save(nib.Nifti1Image(combined_mask, mask_img.affine), os.path.join(output_mask_folder, f'{cid}.nii.gz'))
                save_inspection_plots(np.stack([avg, md, eig[:, :, 1]], axis=-1), combined_mask, cid)
                count += 1

print(f'\nwrote {count} cases -> {output_image_folder}')

# ============================ dataset.json (noNorm x4; labels per task) ============================
labels = {'background': 0, 'LV': 1} if TASK == 'LV' else {'background': 0, 'IP1': 1, 'IP2': 2}
dataset_json = {
    'channel_names': {str(c): 'noNorm' for c in range(4)},   # noNorm -> our prep IS the normalization
    'labels': labels,
    'numTraining': count,
    'file_ending': '.nii.gz',
}
with open(f'{OUT_BASE}/{datasetname}/dataset.json', 'w') as f:
    json.dump(dataset_json, f, indent=4)
print(f'wrote dataset.json  (task={TASK}, labels={labels}, numTraining={count})')
print(f'dataset staged at: {OUT_BASE}/{datasetname}  -> copy to nnUNet_raw/ on the server')
