import nibabel as nib
import os
import cv2
import numpy as np

# Define folders
image_folder = '/Users/saschastocker/Documents/Stanford/work2024/WholeHeartCropISMRM/Summer_of_Segmentation_2024/Directions_vs_Averages_Data/all_cropped_images'
mask_folder = '/Users/saschastocker/Documents/Stanford/work2024/WholeHeartCropISMRM/Summer_of_Segmentation_2024/Directions_vs_Averages_Data/all_cropped_masks'

# Get list of all image files
image_files = [f for f in os.listdir(image_folder) if f.endswith('.nii')]

# Define function to plot a single image and mask using opencv
def show_image_and_mask(image_file, mask_file):
    # Load the image and mask using nibabel
    image_nii = nib.load(image_file)
    mask_nii = nib.load(mask_file)

    # Get the data from the image and mask, rescale to 8-bit for display
    image_data = image_nii.get_fdata().astype(np.float32)
    mask_data = mask_nii.get_fdata().astype(np.float32)

    # Normalize both image and mask for better visualization
    image_data = cv2.normalize(image_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mask_data = cv2.normalize(mask_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Convert to 3-channel grayscale to overlay the mask in red
    image_color = cv2.cvtColor(image_data, cv2.COLOR_GRAY2BGR)
    
    # Overlay the mask where mask values are not 0, set red channel (BGR -> R)
    image_overlay = image_color.copy()
    image_overlay[mask_data > 0] = [0, 0, 255]  # Red overlay on mask

    # Concatenate the three images horizontally for display
    concatenated_image = np.hstack((image_color, cv2.cvtColor(mask_data, cv2.COLOR_GRAY2BGR), image_overlay))

    # Display the concatenated image with filename as title
    window_title = f'Original, Mask, and Overlay: {os.path.basename(image_file)}'
    cv2.imshow(window_title, concatenated_image)

# Loop through the image files and wait for user input to go to the next one
for image_file in image_files:
    # Construct full paths for image and mask
    image_path = os.path.join(image_folder, image_file)
    mask_file = image_file.replace('_image.nii', '_mask.nii')
    mask_path = os.path.join(mask_folder, mask_file)

    # Check if the mask exists
    if os.path.exists(mask_path):
        # Show the current image and mask
        show_image_and_mask(image_path, mask_path)

        # Wait for the user to press any key to show the next plot
        print("Press any key to see the next image, or press ESC to exit.")
        key = cv2.waitKey(0)  # 0 means wait indefinitely
        if key == 27:  # ESC key
            break
    else:
        print(f'No matching mask found for {image_file}')

# Clean up windows after the loop ends
cv2.destroyAllWindows()
