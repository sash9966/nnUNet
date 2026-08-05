"""
Deployable cardiac-DTI whole-heart segmentation pipeline (drop-in for cardpy).

Fully automatic: you point it at a folder of contrasts, it returns a segmentation in the
ORIGINAL image spacing. No manual cropping, no resampling by hand.

-------------------------------------------------------------------------------
1. YOUR FOLDERS
-------------------------------------------------------------------------------
Put every case's co-registered contrasts in one input folder, named `<case>_000X.nii.gz`:

    input/
      Patient01_0000.nii.gz   # DWI  (average diffusion-weighted image)   REQUIRED
      Patient01_0001.nii.gz   # MD   (mean diffusivity)                   baseline 2-contrast
      Patient02_0000.nii.gz
      Patient02_0001.nii.gz
      ...

-------------------------------------------------------------------------------
2. HOW TO SET IT UP  (RULES)
-------------------------------------------------------------------------------
- Channel suffixes are FIXED and 0-indexed:  _0000 = DWI,  _0001 = MD  (add _0002.. only if
  your nnUNet model was trained with more channels; keep the SAME order as training).
- The DWI (`_0000`) is always required -- the crop and insertion-point detectors run on it.
- All contrasts for one case MUST share the same shape/geometry (co-registered).
- The base name before `_000X` is the case id; keep it identical across a case's channels.
- Normalization is applied HERE to match training (nnUNet runs noNorm), one entry per channel:
      CHANNEL_NORM = ['minmax', 'md4']   # DWI min-max, MD /4   <- the DWI+MD baseline

-------------------------------------------------------------------------------
3. WEIGHTS
-------------------------------------------------------------------------------
Set the three weight locations below (local paths). If you host them, set the *_URL fields
and call `download_weights()` once -- it fetches them into ./deployable_weights/.
    - CROP_YOLO           : YOLO crop model (best.pt)
    - IP_YOLO             : YOLO insertion-point model (best.pt), single-contrast avg
    - NNUNET_LV_MODEL_DIR : the nnUNet results folder for the LV model
                            (contains plans.json, dataset.json, fold_0 ...)

-------------------------------------------------------------------------------
4. RUN
-------------------------------------------------------------------------------
    from deployable import CardiacPipeline
    pipe = CardiacPipeline()                 # uses the config below
    pipe.predict_folder('input', 'output')   # -> output/<case>.nii.gz

or from the shell:
    python deployable.py --input input --output output

-------------------------------------------------------------------------------
OUTPUT
-------------------------------------------------------------------------------
    output/<case>.nii.gz  in the ORIGINAL spacing:  1 = LV myocardium,
    2 = anterior insertion point, 3 = inferior insertion point.
"""
import os
import glob
import argparse
import numpy as np
import nibabel as nib

# ------------------------------------------------------------------ config ----
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deployable_weights')

CROP_YOLO           = os.path.join(WEIGHTS_DIR, 'crop.pt')            # YOLO crop best.pt
IP_YOLO             = os.path.join(WEIGHTS_DIR, 'ip.pt')              # YOLO insertion-point best.pt
NNUNET_LV_MODEL_DIR = os.path.join(WEIGHTS_DIR, 'lv_model')           # nnUNet results folder
NNUNET_FOLDS        = (0, 1, 2, 3, 4)                                 # 5-fold ensemble
NNUNET_CHECKPOINT   = 'checkpoint_final.pth'

# one normalization per input channel, IN ORDER (must match how the model was trained)
CHANNEL_NORM = ['minmax', 'md4']     # baseline: DWI (min-max), MD (/4)

# optional download URLs (leave None until you host the weights)
CROP_YOLO_URL = None
IP_YOLO_URL   = None
LV_MODEL_URL  = None                 # a .zip of the nnUNet model folder

TARGET_SIZE = 256
IP_CONF     = 0.01                   # low: the single-class IP detector takes the top-2 points


# ---------------------------------------------------------- normalization ----
def _minmax(x):
    lo, hi = np.min(x), np.max(x)
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

NORMALIZERS = {
    'minmax': _minmax,                    # DWI, FA-like
    'md4':    lambda x: x / 4.0,          # MD physical ceiling
    'sqrt2':  lambda x: x / np.sqrt(2),   # combined eigenvector (X+Y), if pre-summed
    'raw':    lambda x: x,                # already in [0,1] (e.g. FA)
}


def _square_crop_indices(mask, target_size=TARGET_SIZE):
    coords = np.argwhere(mask > 0)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0) + 1
    side = max(x_max - x_min, y_max - y_min)
    cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
    x_min = max(0, cx - side // 2); x_max = x_min + side
    y_min = max(0, cy - side // 2); y_max = y_min + side
    return x_min, x_max, y_min, y_max


def download_weights():
    """Fetch weights from the *_URL fields into WEIGHTS_DIR (call once after hosting them)."""
    import urllib.request
    import zipfile
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    for url, dest in [(CROP_YOLO_URL, CROP_YOLO), (IP_YOLO_URL, IP_YOLO)]:
        if url and not os.path.exists(dest):
            print(f'downloading {url} -> {dest}')
            urllib.request.urlretrieve(url, dest)
    if LV_MODEL_URL and not os.path.isdir(NNUNET_LV_MODEL_DIR):
        zp = NNUNET_LV_MODEL_DIR + '.zip'
        print(f'downloading {LV_MODEL_URL} -> {zp}')
        urllib.request.urlretrieve(LV_MODEL_URL, zp)
        with zipfile.ZipFile(zp) as z:
            z.extractall(NNUNET_LV_MODEL_DIR)
    print('weights ready in', WEIGHTS_DIR)


class CardiacPipeline:
    def __init__(self, crop_yolo=CROP_YOLO, ip_yolo=IP_YOLO, lv_model_dir=NNUNET_LV_MODEL_DIR,
                 folds=NNUNET_FOLDS, channel_norm=CHANNEL_NORM, checkpoint=NNUNET_CHECKPOINT):
        self.crop_yolo, self.ip_yolo = crop_yolo, ip_yolo
        self.lv_model_dir, self.folds, self.checkpoint = lv_model_dir, folds, checkpoint
        self.channel_norm = channel_norm
        for w in (crop_yolo, ip_yolo):
            if not os.path.exists(w):
                print(f'!! weight not found: {w}  (set the path or call download_weights())')
        if not os.path.isdir(lv_model_dir):
            print(f'!! nnUNet LV model dir not found: {lv_model_dir}')

    # ---- stage 1: crop on DWI, square-crop + resize every channel to 256 ----
    def _crop_and_resize(self, case, in_dir, work_dir, crop_dir):
        import cv2
        import yolo_pipeline as yp
        chans = sorted(glob.glob(os.path.join(in_dir, f'{case}_[0-9][0-9][0-9][0-9].nii.gz')))
        dwi_img = nib.load(os.path.join(in_dir, f'{case}_0000.nii.gz'))
        dwi = dwi_img.get_fdata().squeeze()

        box = yp.detect_crop_box(self._crop_model, dwi)
        if box is None:
            return False
        crop_mask = yp.crop_box_to_mask(box, dwi.shape)
        # save crop mask WITH the original affine -> needed to un-crop back to original spacing
        nib.save(nib.Nifti1Image(crop_mask.astype(np.uint8), dwi_img.affine),
                 os.path.join(crop_dir, f'{case}.nii.gz'))

        x0, x1, y0, y1 = _square_crop_indices(crop_mask)
        for ch_path in chans:
            ch = int(ch_path[-11:-7])                       # _000X index
            arr = nib.load(ch_path).get_fdata().squeeze()
            crop = cv2.resize(arr[x0:x1, y0:y1], (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
            norm = self.channel_norm[ch] if ch < len(self.channel_norm) else 'minmax'
            crop = NORMALIZERS[norm](crop)
            nib.save(nib.Nifti1Image(crop, np.eye(4)),
                     os.path.join(work_dir, f'{case}_{ch:04d}.nii.gz'))
        return True

    # ---- stage 2: nnUNet LV segmentation on the 256 crops (Python API) ----
    def _segment_lv(self, work_dir, lv_dir):
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        predictor = nnUNetPredictor(
            tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
            perform_everything_on_device=torch.cuda.is_available(),
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            verbose=False, verbose_preprocessing=False, allow_tqdm=True)
        predictor.initialize_from_trained_model_folder(
            self.lv_model_dir, use_folds=self.folds, checkpoint_name=self.checkpoint)
        predictor.predict_from_files(work_dir, lv_dir, save_probabilities=False, overwrite=True,
                                     num_processes_preprocessing=2, num_processes_segmentation_export=2)

    # ---- stage 3: YOLO insertion points on the 256 DWI ----
    def _detect_ips(self, case, work_dir):
        import yolo_pipeline as yp
        dwi0 = os.path.join(work_dir, f'{case}_0000.nii.gz')
        pts = yp.detect_rvips_from_path(self._ip_model, dwi0, channels=(0, 0, 0),
                                        single_class=True, flip_assignment=False, conf_thresh=IP_CONF)
        return yp.rvips_to_mask(pts, (TARGET_SIZE, TARGET_SIZE), anterior_label=2, inferior_label=3)

    def predict_folder(self, in_dir, out_dir, work_dir=None):
        """Run the full pipeline for every case in in_dir. Writes out_dir/<case>.nii.gz."""
        import yolo_pipeline as yp
        from eval_util import upscale_segmentation_to_original
        work_dir = work_dir or os.path.join(out_dir, '_work')
        crop_dir = os.path.join(work_dir, 'crop_masks')
        s256_dir = os.path.join(work_dir, 'cropped256')
        lv_dir   = os.path.join(work_dir, 'lv256')
        for d in (out_dir, crop_dir, s256_dir, lv_dir):
            os.makedirs(d, exist_ok=True)

        self._crop_model = yp.load_model(self.crop_yolo)
        self._ip_model   = yp.load_model(self.ip_yolo)

        cases = sorted({os.path.basename(f)[:-12] for f in glob.glob(os.path.join(in_dir, '*_0000.nii.gz'))})
        print(f'{len(cases)} cases in {in_dir}')

        kept = []
        for case in cases:
            if self._crop_and_resize(case, in_dir, s256_dir, crop_dir):
                kept.append(case)
            else:
                print(f'  no heart crop detected -> skipped {case}')
        print(f'cropped+resized {len(kept)} cases -> nnUNet LV segmentation ...')

        self._segment_lv(s256_dir, lv_dir)          # LV on all cases at once

        for case in kept:
            lv_p = os.path.join(lv_dir, f'{case}.nii.gz')
            if not os.path.exists(lv_p):
                print(f'  no LV output for {case}'); continue
            lv = nib.load(lv_p).get_fdata().astype(np.int32)
            ip = self._detect_ips(case, s256_dir)
            combined = lv.copy(); combined[lv == 1] = 1
            combined[ip == 2] = 2; combined[ip == 3] = 3     # insertion points

            crop_mask_p = os.path.join(crop_dir, f'{case}.nii.gz')
            full = upscale_segmentation_to_original(combined, crop_mask_p, target_size=TARGET_SIZE)
            ref  = nib.load(os.path.join(in_dir, f'{case}_0000.nii.gz'))
            nib.save(nib.Nifti1Image(full.astype(np.uint8), ref.affine, ref.header),
                     os.path.join(out_dir, f'{case}.nii.gz'))
            print(f'  wrote {case}.nii.gz  (original spacing)')
        print('done ->', out_dir)


def main():
    ap = argparse.ArgumentParser(description='Deployable cardiac-DTI whole-heart segmentation.')
    ap.add_argument('--input', required=True, help='folder with <case>_000X.nii.gz contrasts')
    ap.add_argument('--output', required=True, help='folder for <case>.nii.gz segmentations')
    ap.add_argument('--crop_yolo', default=CROP_YOLO)
    ap.add_argument('--ip_yolo', default=IP_YOLO)
    ap.add_argument('--lv_model_dir', default=NNUNET_LV_MODEL_DIR)
    args = ap.parse_args()
    CardiacPipeline(crop_yolo=args.crop_yolo, ip_yolo=args.ip_yolo,
                    lv_model_dir=args.lv_model_dir).predict_folder(args.input, args.output)


if __name__ == '__main__':
    main()
