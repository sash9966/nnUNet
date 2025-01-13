import os
import nibabel as nib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Root directory and subfolders
pwd = '/Users/saschastocker/Documents/Stanford/work2024/WholeHeartCropISMRM/Summer_of_Segmentation_2024/Directions_vs_Averages_Data'
root_folders = ['Le']
datasetname = 'Dataset022_AvgDiff'
output_mask_folder = f'{pwd}/{datasetname}/labelsTr'
output_image_folder = f'{pwd}/{datasetname}/imagesTr'
inspection_folder = f'{pwd}/inspection{datasetname}'

# Ensure output folders existw
os.makedirs(output_mask_folder, exist_ok=True)
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(inspection_folder, exist_ok=True)

# Function to save images for inspection
def save_inspection_plots(image_data, mask_data, filename_base):
    """Saves inspection plots of image, mask, and overlay using matplotlib."""
    
    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Just the image
    axes[0].imshow(image_data, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Plot 2: Just the mask
    axes[1].imshow(mask_data, cmap='gray')
    axes[1].set_title('Mask Only')
    axes[1].axis('off')

    # Plot 3: Image with mask overlay
    axes[2].imshow(image_data, cmap='gray')
    mask_overlay = np.ma.masked_where(mask_data != 1, mask_data)
    axes[2].imshow(mask_overlay, cmap='Reds', alpha=0.5)
    axes[2].set_title('Image with Mask Overlay')
    axes[2].axis('off')

    # Save the figure
    output_file = os.path.join(inspection_folder, f'{filename_base}_inspection.png')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close(fig)
    print(f'Saved inspection plot: {output_file}')


for root_folder in root_folders:
    print(f'root folder: {root_folder}')
    root_path = os.path.join(pwd, root_folder)

    # Loop through volunteer folders
    for volunteer_folder in os.listdir(root_path):
        if volunteer_folder.startswith('Volunteer'):
            volunteer_path = os.path.join(root_path, volunteer_folder)
            distortion_corrected_folder = os.path.join(volunteer_path, 'Distortion_Corrected')

            # Loop through DiVO and MDDW folders
            for divo_folder in os.listdir(distortion_corrected_folder):
                if divo_folder.startswith('DiVO') or divo_folder.startswith('MDDW'):
                    divo_path = os.path.join(distortion_corrected_folder, divo_folder)

                    # Load the quality control information from the Excel file in the DiVO/MDDW folder
                    excel_path = os.path.join(divo_path, 'Detailed_Information.xlsx')
                    if os.path.exists(excel_path):
                        quality_data = pd.read_excel(excel_path)

                        # Check if the "Image Quality" column has "Good Image" for all rows (3 slices)
                        if quality_data['Image Quality'].str.contains('Good Image').sum() == 3:
                            # Proceed with the images only if all slices are of good quality
                            mask_folder = os.path.join(divo_path, '02_Crop_Masks')
                            image_folder = os.path.join(divo_path, '03_Segmentation_Images')

                            # Loop over the three mask and image files (mask_001.nii to mask_003.nii)
                            for i in range(1, 4):  # mask_001.nii to mask_003.nii and image_001.nii to image_003.nii

                                # Select mask and image files for each iteration
                                mask_file = os.path.join(mask_folder, f'Square_Crop_Mask_Slice_00{i}.nii')
                                image_file = os.path.join(image_folder, f'Average_Diffusion_Weighted_Image_Slice_00{i}.nii')

                                if os.path.exists(mask_file) and os.path.exists(image_file):
                                    # Load the NIfTI mask file and extract the 0th slice
                                    mask_img = nib.load(mask_file)
                                    mask_data = mask_img.get_fdata()

                                    # Load the NIfTI image file (no slicing needed)
                                    image_img = nib.load(image_file)
                                    image_data = image_img.get_fdata()

                                    common_name_id = f'{root_folder}_{volunteer_folder}_{divo_folder}_slice_00{i}'

                                    mask_output_filename = os.path.join(output_mask_folder,
                                                                        f'{common_name_id}.nii.gz')
                                    image_output_filename = os.path.join(output_image_folder,
                                                                        f'{common_name_id}_0000.nii.gz')

                                    # Save the mask and image
                                    nib.save(nib.Nifti1Image(mask_data, mask_img.affine), mask_output_filename)
                                    nib.save(nib.Nifti1Image(image_data, image_img.affine), image_output_filename)

                                    print(f'Saved mask slice {i}: {mask_output_filename}')
                                    print(f'Saved image slice {i}: {image_output_filename}')

                                    # Save inspection images (original, mask, overlay)
                                    save_inspection_plots(image_data, mask_data, common_name_id)
                                else:
                                    if not os.path.exists(mask_file):
                                        print(f'Failed to find mask file: {mask_file}')
                                    if not os.path.exists(image_file):
                                        print(f'Failed to find image file: {image_file}')
                        else:
                            # If any slice is not "Good Image", skip the processing and print the folder name
                            print(f'Skipping images in {divo_folder} due to bad quality')
                    else:
                        print(f'Missing quality information in {divo_folder}')
