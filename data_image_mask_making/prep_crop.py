"""
Parameterized cDTI CROP prep, built DIRECTLY from the ORIGINAL data folders.

Single-cohort OR combined: loops over the selected cohorts' raw folders and writes every case into
ONE output dataset. The crop model only needs the DWI, so this is a 1-channel dataset:
    _0000 = average diffusion-weighted image (min-max)   +   label = the square crop box.

Set COHORTS, OUTPUT_ID/NAME below, then:  python prep_crop.py
Produce, e.g.:
    ('SmartHealth',)                 -> 110
    ('SmartHealth','DirVsAverages')  -> 111   (combined, from source)
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
COHORTS = ['SmartHealth', 'DirVsAverages']
OUTPUT_ID   = 111
OUTPUT_NAME = 'HannumSmartHealthandDirVsAvgCrop'

PAPER = '/Users/saschastocker/Documents/Stanford/DanEnnis20242025/Paper2025Automatic'
COHORTS_SRC = {
    'SmartHealth':   dict(pwd=f'{PAPER}/Smart_Health',  root='Hannum'),
    'DirVsAverages': dict(pwd=f'{PAPER}/DirVsAverages',  root='DirVsAveragesHannum'),
}

OUT_BASE = COHORTS_SRC[COHORTS[0]]['pwd']
datasetname = f'Dataset{OUTPUT_ID:03d}_{OUTPUT_NAME}'
output_image_folder = f'{OUT_BASE}/{datasetname}/imagesTr'
output_mask_folder  = f'{OUT_BASE}/{datasetname}/labelsTr'
inspection_folder   = f'{OUT_BASE}/inspection{datasetname}'
for d in (output_image_folder, output_mask_folder, inspection_folder):
    os.makedirs(d, exist_ok=True)


# ---- normalization: DWI min-max (verbatim from NewCropPrep.py) ----
def normalize_image(image):
    image_min = np.min(image)
    image_max = np.max(image)
    return (image - image_min) / (image_max - image_min)


def save_inspection_plots(image_data, mask_data, base):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(image_data, cmap='gray'); ax[0].set_title('Average Diffusion'); ax[0].axis('off')
    ax[1].imshow(mask_data, cmap='gray'); ax[1].set_title('Crop box'); ax[1].axis('off')
    ax[2].imshow(image_data, cmap='gray')
    ax[2].imshow(np.ma.masked_where(mask_data != 1, mask_data), cmap='Reds', alpha=0.5)
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

            mask_folder  = os.path.join(divo_path, '02_Crop_Masks')
            image_folder = os.path.join(divo_path, '03_Segmentation_Images')
            slice_files = glob.glob(os.path.join(mask_folder, 'Square_Crop_Mask_Slice_*.nii'))
            slices = sorted(int(re.search(r'Slice_(\d+)\.nii$', f).group(1)) for f in slice_files)
            print(f'  {volunteer_folder}/{divo_folder}: {len(slices)} slices -> {slices}')

            for i in slices:
                row = quality[quality['Slice Number'] == i]
                if row.empty or row.iloc[0]['Image Quality'] != 'Good Image Quality':
                    print(f'    skip slice {i} (bad quality)'); continue

                mask_file  = os.path.join(mask_folder, f'Square_Crop_Mask_Slice_{i:03d}.nii')
                image_file = os.path.join(image_folder, f'Average_Diffusion_Weighted_Image_Slice_{i:03d}.nii')
                if not (os.path.exists(mask_file) and os.path.exists(image_file)):
                    print(f'    missing files for slice {i}'); continue

                mask_img = nib.load(mask_file); mask_data = mask_img.get_fdata()
                image_img = nib.load(image_file); image_data = normalize_image(image_img.get_fdata())

                cid = f'{root_folder}_{volunteer_folder}_{divo_folder}_slice_{i:03d}'
                nib.save(nib.Nifti1Image(image_data, image_img.affine), os.path.join(output_image_folder, f'{cid}_0000.nii.gz'))
                nib.save(nib.Nifti1Image(mask_data,  mask_img.affine),  os.path.join(output_mask_folder, f'{cid}.nii.gz'))
                save_inspection_plots(image_data, mask_data, cid)
                count += 1

print(f'\nwrote {count} cases -> {output_image_folder}')

# ---- dataset.json: 1 channel noNorm, crop-box label ----
dataset_json = {
    'channel_names': {'0': 'noNorm'},
    'labels': {'background': 0, 'crop': 1},
    'numTraining': count,
    'file_ending': '.nii.gz',
}
with open(f'{OUT_BASE}/{datasetname}/dataset.json', 'w') as f:
    json.dump(dataset_json, f, indent=4)
print(f'wrote dataset.json (crop, numTraining={count})')
print(f'dataset staged at: {OUT_BASE}/{datasetname}  -> copy to nnUNet_raw/ on the server')
