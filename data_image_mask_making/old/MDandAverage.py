import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Root directory and subfolders
pwd = '/Users/saschastocker/Documents/Stanford/work2024/WholeHeartCropISMRM/Summer_of_Segmentation_2024/Directions_vs_Averages_Data'
root_folders = ['Le']
datasetname = 'Dataset005_MDandAverageNormalised'
output_mask_folder = f'{pwd}/{datasetname}/labelsTr'
output_image_folder = f'{pwd}/{datasetname}/imagesTr'
inspection_folder = f'{pwd}/inspection{datasetname}'

# Ensure output folders exist
os.makedirs(output_mask_folder, exist_ok=True)
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(inspection_folder, exist_ok=True)

# Function to normalize images to [0, 1] range
def normalize_image(image):
    image_min = np.min(image)
    image_max = np.max(image)
    return (image - image_min) / (image_max - image_min)

def normalise_MD(image):
    image_min = 0
    image_max = 4
    return (image - image_min) / (image_max - image_min)

# Function to save images for inspection
def save_inspection_plots(image_data, mask_data, filename_base):
    """Saves inspection plots of image, mask, and overlay using matplotlib."""
    
    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Just the image (first channel)
    axes[0].imshow(image_data[:, :, 0], cmap='gray')  # Show first channel (average)
    axes[0].set_title('Average Diffusion Image')
    axes[0].axis('off')

    # Plot 2: Just the mask
    axes[1].imshow(mask_data, cmap='gray')
    axes[1].set_title('Mask Only')
    axes[1].axis('off')

    # Plot 3: Image with mask overlay
    axes[2].imshow(image_data[:, :, 0], cmap='gray')  # Show first channel (average)
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
                            mask_folder = os.path.join(divo_path, '06_Segmentation_Masks_CI')
                            image_folder = os.path.join(divo_path, '05_Segmentation_Images_CI')

                            # Loop over the three mask and image files (mask_001.nii to mask_003.nii)
                            for i in range(1, 4):  # mask_001.nii to mask_003.nii and image_001.nii to image_003.nii

                                # Select mask and image files for each iteration
                                mask_file = os.path.join(mask_folder, f'Cropped_Segmentation_Slice_00{i}.nii')
                                avg_image_file = os.path.join(image_folder, f'Cropped_Average_Diffusion_Weighted_Image_Slice_00{i}.nii')
                                mean_diff_file = os.path.join(image_folder, f'Cropped_Mean_Diffusivty_Image_Slice_00{i}.nii')
                                

                                if os.path.exists(mask_file) and os.path.exists(avg_image_file) and os.path.exists(mean_diff_file):
                                    # Load the NIfTI mask file and extract the 0th slice
                                    mask_img = nib.load(mask_file)
                                    mask_data = mask_img.get_fdata()
                                    mask_slice = mask_data[:, :, 0]  # Extract the 0th slice from each mask file

                                    # Load the three NIfTI image files (average, mean diffusivity, eigenvector)
                                    avg_img = nib.load(avg_image_file)
                                    mean_diff_img = nib.load(mean_diff_file)


                                    # Get the data for all three images
                                    avg_image_data = avg_img.get_fdata()  # (256, 256)
                                    mean_diff_data = mean_diff_img.get_fdata()  # (256, 256)


                                    # Normalize each image to [0, 1] range to prevent "washed out" effect
                                    avg_image_data = normalize_image(avg_image_data)
                                    mean_diff_data = normalise_MD(mean_diff_data)


                                    # Save each channel separately (modality files with 0000, 0001, 0002 suffixes)
                                    common_name_id = f'{root_folder}_{volunteer_folder}_{divo_folder}_slice_00{i}'

                                    # Save Average Diffusion Image as _0000
                                    nib.save(nib.Nifti1Image(avg_image_data, avg_img.affine), 
                                             os.path.join(output_image_folder, f'{common_name_id}_0000.nii.gz'))

                                    # Save Mean Diffusivity Image as _0001
                                    nib.save(nib.Nifti1Image(mean_diff_data, mean_diff_img.affine), 
                                             os.path.join(output_image_folder, f'{common_name_id}_0001.nii.gz'))

                                    # Save the mask
                                    nib.save(nib.Nifti1Image(mask_slice, mask_img.affine), 
                                             os.path.join(output_mask_folder, f'{common_name_id}.nii.gz'))

                                    print(f'Saved mask slice {i}: {common_name_id}.nii.gz')
                                    print(f'Saved image modalities: {common_name_id}_0000.nii.gz, _0001.nii.gz, ')

                                    # Save inspection images (original, mask, overlay)
                                    save_inspection_plots(np.stack([avg_image_data, mean_diff_data], axis=-1), 
                                                          mask_slice, common_name_id)
                                else:
                                    print(f'Failed to find required files for slice {i}')
                        else:
                            # If any slice is not "Good Image", skip the processing and print the folder name
                            print(f'Skipping images in {divo_folder} due to bad quality')
                    else:
                        print(f'Missing quality information in {divo_folder}')
