from time import time
import numpy as np
from numba import njit
import os
import glymur
from PIL import Image
import matplotlib.pyplot as plt
import multiprocessing as mp
import h5py
from tqdm import tqdm
import yaml

plt.rcParams.update({
    "font.size": 13,
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}",
})

def load_hip_ct_scan_to_np(scan_path_folder):
    """
    Loads HIP CT image slices from a specified folder or HDF5 file into a NumPy array.
    Args:
        scan_path_folder (str): Path to the folder containing HIP CT image files or an HDF5 file.
    Returns:
        np.ndarray: A NumPy array containing all loaded HIP CT image slices, stacked along the first axis.
    Notes:
        - Supports ".jp2" (JPEG2000), ".tif"/".tiff" (TIFF), and ".h5"/".hdf5" (HDF5) files.
        - If a folder is provided, loads all JP2 or TIFF files (sorted by filename).
        - If an HDF5 file is provided, loads the first dataset found.
        - Prints the name of the loaded dataset (the folder or file name).
        - Requires `glymur`, `numpy`, `PIL`, `h5py`, and `tqdm` libraries.
    """

    if os.path.isfile(scan_path_folder) and scan_path_folder.lower().endswith(('.h5', '.hdf5')):
        # Load from HDF5 file
        with h5py.File(scan_path_folder, 'r') as f:
            # Try to find the first dataset
            def find_dataset(g):
                for key in g:
                    item = g[key]
                    if isinstance(item, h5py.Dataset):
                        return item
                    elif isinstance(item, h5py.Group):
                        ds = find_dataset(item)
                        if ds is not None:
                            return ds
                return None
            dataset = find_dataset(f)
            if dataset is None:
                raise ValueError("No dataset found in HDF5 file.")
            arr = np.array(dataset)
        print(f'Loaded Dataset {os.path.basename(scan_path_folder)} (HDF5)')
        return arr

    elif os.path.isdir(scan_path_folder):
        files = sorted(os.listdir(scan_path_folder))
        # Check for JP2 files
        jp2_files = [f for f in files if f.lower().endswith('.jp2')]
        tif_files = [f for f in files if f.lower().endswith(('.tif', '.tiff'))]
        if jp2_files:
            jp2_slices = []
            print(f"Loading {len(jp2_files)} JP2 slices...")
            for file in tqdm(jp2_files, desc="JP2 Loading", unit="slice"):
                jp2 = glymur.Jp2k(os.path.join(scan_path_folder, file))
                image_array = np.array(jp2[:])
                jp2_slices.append(image_array)
            jp2_slices = np.array(jp2_slices)
            print(f'Loaded Dataset {os.path.basename(os.path.normpath(scan_path_folder))} (JP2)')
            return jp2_slices
        elif tif_files:
            tif_slices = []
            print(f"Loading {len(tif_files)} TIFF slices...")
            for file in tqdm(tif_files, desc="TIFF Loading", unit="slice"):
                tif = Image.open(os.path.join(scan_path_folder, file))
                image_array = np.array(tif)
                tif_slices.append(image_array)
            tif_slices = np.array(tif_slices)
            print(f'Loaded Dataset {os.path.basename(os.path.normpath(scan_path_folder))} (TIFF)')
            return tif_slices
        else:
            raise ValueError("No supported image files (.jp2, .tif, .tiff) found in the folder.")
    else:
        raise ValueError("scan_path_folder must be a directory containing JP2/TIFF files or an HDF5 file.")
    
def np_to_h5(array, path_dicom, path_h5):
    """
    Saves a NumPy array to an HDF5 file using the DICOM file's basename as the dataset name.
    If a dataset with the same name already exists in the HDF5 file, the function skips saving.
    Parameters:
        array (np.ndarray): The NumPy array to be saved.
        path_dicom (str): Path to the DICOM file; its basename is used as the dataset name.
        path_h5 (str): Path to the HDF5 file where the array will be saved.
    Returns:
        None
    Side Effects:
        Writes the NumPy array to the specified HDF5 file. Prints status messages indicating whether the dataset was saved or skipped.
    """
    dataset_name = os.path.basename(path_dicom) 
    
    # Open the HDF5 file in append mode
    with h5py.File(path_h5, 'a') as h5_file:
        # Check if the dataset already exists
        if dataset_name in h5_file:
            print(f"Dataset {dataset_name} already exists in {path_h5}, skipping.")
        else:
            # Save the NumPy array to the HDF5 file
            h5_file.create_dataset(dataset_name, data=array)
            print(f"Saved {dataset_name} to {path_h5}")



@njit
def surrounded_check(mask, merge_size, diag_size, cross_size, iterations):
    """
    Performs a series of morphological checks on a mask array over multiple iterations.

    This function applies three types of checks—merge, diagonal, and cross—on the input mask,
    using the corresponding sizes provided for each operation. Each check is applied in sequence
    for each iteration, and only if the respective size for that iteration is non-zero.

    Args:
        mask (ndarray): The input mask array to be processed.
        merge_size (array-like): Sequence of sizes for the merge check in each iteration.
        diag_size (array-like): Sequence of sizes for the diagonal check in each iteration.
        cross_size (array-like): Sequence of sizes for the cross check in each iteration.
        iterations (int): Number of iterations to perform the checks.

    Returns:
        ndarray: The processed mask array after all iterations and checks.
    """
    for i in range(iterations):
        if merge_size[i] != 0:
            mask = surrounded_merge_check(mask, merge_size[i])
        if diag_size[i] != 0:
            mask = surrounded_diagonal_check(mask, diag_size[i])
        if cross_size[i] != 0:
            mask = surrounded_cross_check(mask, cross_size[i])

    return mask


@njit
def surrounded_cross_check(mask, size):
    """
    Checks each zero-valued pixel in the input mask to determine if it is surrounded by non-zero values within a specified cross-shaped neighborhood, and sets it to one if so.

    Args:
        mask (np.ndarray): 2D binary array representing the mask to be processed.
        size (int): The size of the cross-shaped neighborhood to check around each zero-valued pixel.

    Returns:
        np.ndarray: The modified mask with surrounded zero-valued pixels set to one.
    """
    N_y, N_x = mask.shape

    for i in range(N_y):
        for j in range(N_x):
            if mask[i, j] == 0:
                surrounded = True
                x_low = max(0, j-size)
                x_high = min(N_x, j+size)
                y_low = max(0, i-size)
                y_high = min(N_y, i+size)
                if np.sum(mask[y_low:i, j]) == 0 or np.sum(mask[i:y_high, j]) == 0 or np.sum(mask[i, x_low:j]) == 0 or np.sum(mask[i, j:x_high]) == 0:
                    surrounded = False
                if surrounded:
                    mask[i, j] = 1

    return mask


@njit
def surrounded_diagonal_check(mask, size):
    """
    Checks each zero-valued pixel in a 2D mask to determine if it is "surrounded" along its four diagonal directions within a specified distance, and sets it to 1 if so.

    For each zero pixel, the function examines the diagonals in the four quadrants (top-left, top-right, bottom-left, bottom-right) up to a given size. If all diagonals in these quadrants contain only nonzero values (i.e., are "surrounded"), the pixel is set to 1.

    Parameters:
        mask (np.ndarray): 2D numpy array representing the mask, with shape (N_y, N_x).
        size (int): The maximum distance to check along each diagonal direction.

    Returns:
        np.ndarray: The modified mask with surrounded zero pixels set to 1.
    """
    N_y, N_x = mask.shape

    for i in range(N_y):
        for j in range(N_x):
            if mask[i, j] == 0:
                surrounded = True
                # ensure we don't go out of bounds
                x_low_size = min(j, size)
                x_high_size = min(N_x-j-1, size)
                y_low_size = min(i, size)
                y_high_size = min(N_y-i-1, size)

                # ensure quadrants are quadratic
                top_left_size = min(x_low_size, y_low_size)
                bottom_right_size = min(x_high_size, y_high_size)
                top_right_size = min(x_high_size, y_low_size)
                bottom_left_size = min(x_low_size, y_high_size)

                # get the quadrants
                top_left = mask[i-top_left_size:i, j-top_left_size:j]
                bottom_right = mask[i+1:i+1+bottom_right_size, j+1:j+1+bottom_right_size]
                top_right = mask[i-top_right_size:i, j+1:j+1+top_right_size]
                bottom_left = mask[i+1:i+1+bottom_left_size, j-bottom_left_size:j]

                # get the diagonals
                diag1 = np.array([top_left[k, k] for k in range(top_left_size)][::-1], dtype=np.int32)
                diag2 = np.array([bottom_right[k, k] for k in range(bottom_right_size)], dtype=np.int32)
                diag3 = np.array([top_right[top_right_size-k-1, k] for k in range(top_right_size)], dtype=np.int32)
                diag4 = np.array([bottom_left[k, bottom_left_size-k-1] for k in range(bottom_left_size)], dtype=np.int32)
                if np.sum(diag1) == 0 or np.sum(diag2) == 0 or np.sum(diag3) == 0 or np.sum(diag4) == 0:
                    surrounded = False
                if surrounded:
                    mask[i, j] = 1

    return mask


@njit
def surrounded_merge_check(mask, size):
    """
    Checks each zero-valued pixel in a 2D binary mask to determine if it is completely surrounded by ones
    within a specified neighborhood size, and if so, sets it to one.

    The function examines both cross (vertical and horizontal) and diagonal neighbors within the given
    `size` radius. If a zero-valued pixel is surrounded by ones in all directions (cross and diagonals),
    it is considered "surrounded" and is set to one in the mask.

    Parameters
    ----------
    mask : np.ndarray
        A 2D numpy array representing the binary mask (values should be 0 or 1).
    size : int
        The radius of the neighborhood to check for surrounding ones.

    Returns
    -------
    np.ndarray
        The modified mask with surrounded zeros set to one.
    """
    N_y, N_x = mask.shape

    for i in range(N_y):
        for j in range(N_x):
            if mask[i, j] == 0:
                surrounded = True

                # cross check
                x_low = max(0, j-size)
                x_high = min(N_x, j+size)
                y_low = max(0, i-size)
                y_high = min(N_y, i+size)
                if np.sum(mask[y_low:i, j]) == 0 or np.sum(mask[i:y_high, j]) == 0 or np.sum(mask[i, x_low:j]) == 0 or np.sum(mask[i, j:x_high]) == 0:
                    surrounded = False

                # ensure we don't go out of bounds
                x_low_size = min(j, size)
                x_high_size = min(N_x-j-1, size)
                y_low_size = min(i, size)
                y_high_size = min(N_y-i-1, size)

                # ensure quadrants are quadratic
                top_left_size = min(x_low_size, y_low_size)
                bottom_right_size = min(x_high_size, y_high_size)
                top_right_size = min(x_high_size, y_low_size)
                bottom_left_size = min(x_low_size, y_high_size)

                # get the quadrants
                top_left = mask[i-top_left_size:i, j-top_left_size:j]
                bottom_right = mask[i+1:i+1+bottom_right_size, j+1:j+1+bottom_right_size]
                top_right = mask[i-top_right_size:i, j+1:j+1+top_right_size]
                bottom_left = mask[i+1:i+1+bottom_left_size, j-bottom_left_size:j]

                # get the diagonals (and the neighbors for stability)
                diag1 = np.array([top_left[k, k] for k in range(top_left_size)][::-1], dtype=np.int32)
                if top_left_size > 1:
                    diag1_up = np.array([top_left[k, k+1] for k in range(top_left_size-1)][::-1], dtype=np.int32)
                    diag1_down = np.array([top_left[k+1, k] for k in range(top_left_size-1)][::-1], dtype=np.int32)
                    diag1 = np.concatenate((diag1, diag1_up, diag1_down))
                diag2 = np.array([bottom_right[k, k] for k in range(bottom_right_size)], dtype=np.int32)
                if bottom_right_size > 1:
                    diag2_up = np.array([bottom_right[k, k+1] for k in range(bottom_right_size-1)], dtype=np.int32)
                    diag2_down = np.array([bottom_right[k+1, k] for k in range(bottom_right_size-1)], dtype=np.int32)
                    diag2 = np.concatenate((diag2, diag2_up, diag2_down))
                diag3 = np.array([top_right[top_right_size-k-1, k] for k in range(top_right_size)], dtype=np.int32)
                if top_right_size > 1:
                    diag3_up = np.array([top_right[top_right_size-k-1, k+1] for k in range(top_right_size-1)], dtype=np.int32)
                    diag3_down = np.array([top_right[top_right_size-k-2, k] for k in range(top_right_size-1)], dtype=np.int32)
                    diag3 = np.concatenate((diag3, diag3_up, diag3_down))
                diag4 = np.array([bottom_left[k, bottom_left_size-k-1] for k in range(bottom_left_size)], dtype=np.int32)
                if bottom_left_size > 1:
                    diag4_up = np.array([bottom_left[k, bottom_left_size-k-2] for k in range(bottom_left_size-1)], dtype=np.int32)
                    diag4_down = np.array([bottom_left[k+1, bottom_left_size-k-1] for k in range(bottom_left_size-1)], dtype=np.int32)
                    diag4 = np.concatenate((diag4, diag4_up, diag4_down))

                if np.sum(diag1) == 0 or np.sum(diag2) == 0 or np.sum(diag3) == 0 or np.sum(diag4) == 0:
                    surrounded = False
                if surrounded:
                    mask[i, j] = 1

    return mask


@njit
def neighborhood_check(mask, size, threshold, iterations):
    """
    Performs an iterative neighborhood check on a binary mask, setting pixels to 0 if the average value
    in their local neighborhood falls below a specified threshold.

    Parameters:
        mask (np.ndarray): 2D binary numpy array representing the mask to be processed.
        size (int): The half-size of the square neighborhood to consider around each pixel.
        threshold (float): The minimum average value required in the neighborhood to keep a pixel set to 1.
        iterations (int): Number of times to repeat the neighborhood check process.

    Returns:
        np.ndarray: The modified mask after applying the neighborhood check for the specified number of iterations.
    """
    N_y, N_x = mask.shape

    for _ in range(iterations):
        for i in range(N_y):
            for j in range(N_x):
                if mask[i, j] == 1:
                    x_low = max(0, j-size)
                    x_high = min(N_x, j+size)
                    y_low = max(0, i-size)
                    y_high = min(N_y, i+size)
                    neighborhood = mask[y_low:y_high, x_low:x_high]
                    avg_val = np.mean(neighborhood)
                    if avg_val < threshold:
                        mask[i, j] = 0

    return mask


def ring_check(mask, radius, width, offset, return_ring_mask=False):
    """
    Removes a ring-shaped region from a 2D mask array by setting its values to zero.

    Parameters:
        mask (np.ndarray): 2D array representing the mask to be modified.
        radius (float): The radius of the ring (distance from the center).
        width (float): The width (thickness) of the ring.
        offset (tuple of float): (x, y) offset to shift the center of the ring.
        return_ring_mask (bool, optional): If True, also returns the boolean mask of the ring region. Default is False.

    Returns:
        np.ndarray: The modified mask with the ring region set to zero.
        tuple (optional): If return_ring_mask is True, returns a tuple (modified_mask, ring_mask), where ring_mask is a boolean array indicating the ring region.
    """
    N_y, N_x = mask.shape
    x_vals = np.arange(N_x) - (N_x-1)/2 - offset[0]
    y_vals = np.arange(N_y) - (N_y-1)/2 + offset[1]
    x_vals, y_vals = np.meshgrid(x_vals, y_vals)
    dist_center = np.sqrt(np.square(x_vals) + np.square(y_vals))
    ring_mask = (dist_center > radius - width/2) & (dist_center < radius + width/2)

    mask[ring_mask] = 0

    if return_ring_mask:
        return mask, ring_mask
    return mask


@njit
def fill_black_boxes(mask):
    """
    Fills isolated black (0-valued) pixels in a binary mask that are surrounded by sufficiently large white (1-valued) regions, 
    based on configurable distance and neighborhood criteria.
    For each black pixel, the function:
        - Measures the distance to the nearest white pixel in all four cardinal directions (up, down, left, right).
        - Checks if the pixel is sufficiently far from the nearest white pixels (minimum and maximum distance thresholds).
        - Verifies that the region around the pixel is mostly black, and that the border regions are mostly white.
        - If all criteria are met, the black pixel is converted to white in the output mask.
    Parameters
    ----------
    mask : np.ndarray
        2D numpy array of shape (N_y, N_x), representing a binary mask with values 0 (black) and 1 (white).
    Returns
    -------
    output_mask : np.ndarray
        2D numpy array of the same shape as `mask`, with selected black pixels filled to white according to the criteria.
    """
    output_mask = mask.copy()
    N_y, N_x = mask.shape
    for i in range(N_y):
        for j in range(N_x):
            if mask[i, j] == 0:
                # get distances to next white pixel in each direction (up, down, left, right)
                dists = np.zeros(4)
                in_bounds = True
                for k in range(1, N_y):
                    if i-k < 0:
                        in_bounds = False
                        break
                    if mask[i-k, j] == 1:
                        dists[0] = k
                        break
                if in_bounds:
                    for k in range(1, N_y):
                        if i+k >= N_y:
                            in_bounds = False
                            break
                        if mask[i+k, j] == 1:
                            dists[1] = k
                            break
                if in_bounds:
                    for k in range(1, N_x):
                        if j-k < 0:
                            in_bounds = False
                            break
                        if mask[i, j-k] == 1:
                            dists[2] = k
                            break
                if in_bounds:
                    for k in range(1, N_x):
                        if j+k >= N_x:
                            in_bounds = False
                            break
                        if mask[i, j+k] == 1:
                            dists[3] = k
                            break
                if in_bounds and np.min(dists) > 5 and np.max(dists) > 25:
                    in_black_box = True

                    # infield with at least 2px margin to each side has to be almost full black
                    infield_top_dist = min(int(0.9*dists[0]), dists[0]-3)
                    infield_bottom_dist = min(int(0.9*dists[1]), dists[1]-3)
                    infield_left_dist = min(int(0.9*dists[2]), dists[2]-3)
                    infield_right_dist = min(int(0.9*dists[3]), dists[3]-3)
                    infield = mask[i-infield_top_dist:i+infield_bottom_dist+1, j-infield_left_dist:j+infield_right_dist+1]
                    if np.mean(infield) > 0.01:
                        in_black_box = False

                    # check if border region (+2px away from center and + 1px in direction of the center) is basically 75% white (3 white lines + black line)
                    outer_border_top = max(i-dists[0]-2, 0)
                    outer_border_bottom = min(i+dists[1]+2, N_y)
                    outer_border_left = max(j-dists[2]-2, 0)
                    outer_border_right = min(j+dists[3]+2, N_x)

                    border1 = mask[outer_border_top:i-dists[0]+2, j-infield_left_dist:j+infield_right_dist+1]
                    border2 = mask[i+dists[1]-1:outer_border_bottom+1, j-infield_left_dist:j+infield_right_dist+1]
                    border3 = mask[i-infield_top_dist:i+infield_bottom_dist+1, outer_border_left:j-dists[2]+2]
                    border4 = mask[i-infield_top_dist:i+infield_bottom_dist+1, j+dists[3]-1:outer_border_right+1]
                    
                    ratio_threshold = 0.7
                    n_pass_ratio = int(np.mean(border1) > ratio_threshold) + int(np.mean(border2) > ratio_threshold) + int(np.mean(border3) > ratio_threshold) + int(np.mean(border4) > ratio_threshold)
                    if n_pass_ratio < 2:
                        in_black_box = False

                    if in_black_box:
                        output_mask[i, j] = 1

    return output_mask


def segement_slice(scan_slice, threshold=100, ring_radius=500, ring_width=10, ring_offset=(0, 0), surrounded_merge_size=100, surrounded_diag_size=10, surrounded_cross_size=10, surrounded_iterations=1, neighborhood_size=15, neighborhood_threshold=0.01, neighborhood_iterations=1):
    """
    Segments a single scan slice using a series of image processing steps.

    Parameters:
        scan_slice (np.ndarray): The input 2D image slice to be segmented.
        threshold (int, optional): Intensity threshold for initial mask creation. Defaults to 100.
        ring_radius (int, optional): Radius for the ring check operation. Defaults to 500.
        ring_width (int, optional): Width of the ring for the ring check. Defaults to 10.
        ring_offset (tuple, optional): Offset for the ring center (x, y). Defaults to (0, 0).
        surrounded_merge_size (int, optional): Merge size parameter for surrounded check. Defaults to 100.
        surrounded_diag_size (int, optional): Diagonal size for surrounded check. Defaults to 10.
        surrounded_cross_size (int, optional): Cross size for surrounded check. Defaults to 10.
        surrounded_iterations (int, optional): Number of iterations for surrounded check. Defaults to 1.
        neighborhood_size (int, optional): Size of the neighborhood for neighborhood check. Defaults to 15.
        neighborhood_threshold (float, optional): Threshold for neighborhood check. Defaults to 0.01.
        neighborhood_iterations (int, optional): Number of iterations for neighborhood check. Defaults to 1.

    Returns:
        np.ndarray: Boolean mask of the segmented slice after all processing steps.
    """
    mask = scan_slice > threshold
    mask = ring_check(mask, ring_radius, ring_width, ring_offset)
    mask = surrounded_check(mask, surrounded_merge_size, surrounded_diag_size, surrounded_cross_size, surrounded_iterations)
    mask = neighborhood_check(mask, neighborhood_size, neighborhood_threshold, neighborhood_iterations)
    mask = fill_black_boxes(mask)
    mask = surrounded_check(mask, surrounded_merge_size, surrounded_diag_size, surrounded_cross_size, 1)
    return mask


class HiPCTSegmenter:
    def __init__(self, threshold=100, ring_radius=500, ring_width=10, ring_offset=(0, 0)):
        self.threshold = threshold
        self.ring_radius = ring_radius
        self.ring_width = ring_width
        self.ring_offset = ring_offset

        self.surrounded_merge_size = 100
        self.surrounded_diag_size = 10
        self.surrounded_cross_size = 10
        self.surrounded_iterations = 3
        self.neighborhood_size = 15
        self.neighborhood_threshold = 0.01
        self.neighborhood_iterations = 1

    def tune_threshold(self, threshold, scan_slice, do_plots=True, enhance_contrast=False):
        """
        Apply a threshold to a 2D scan slice to generate a binary mask, with optional visualization.

        Parameters
        ----------
        threshold : float
            The threshold value to apply to the scan slice. Pixels greater than this value are set to True in the mask.
        scan_slice : np.ndarray
            2D numpy array representing the scan slice to be thresholded.
        do_plots : bool, optional (default=True)
            If True, display visualizations including the original slice, threshold mask, and histogram of pixel values.
        enhance_contrast : bool, optional (default=False)
            If True, clip the scan slice values to the 1st and 99th percentiles to enhance contrast in the visualizations.

        Returns
        -------
        mask : np.ndarray
            Boolean mask of the same shape as `scan_slice`, where True indicates pixels above the threshold.
        """
        self.threshold = threshold
        mask = scan_slice > threshold

        if do_plots:
            min_val = np.quantile(scan_slice, 0.01)
            max_val = np.quantile(scan_slice, 0.99)
            bins = np.linspace(min_val, max_val, 50)
            if enhance_contrast:
                scan_slice[scan_slice < min_val] = min_val
                scan_slice[scan_slice > max_val] = max_val

            _, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(scan_slice, cmap='gray')
            ax[0].set_title('Original slice')
            ax[0].axis('off')
            ax[1].imshow(mask, cmap='gray')
            ax[1].axis('off')
            ax[1].set_title('Threshold mask')
            ax[2].hist(scan_slice.flatten(), bins=bins, color='c', alpha=0.7)
            ax[2].axvline(threshold, color='r', linestyle='dashed', linewidth=2)
            ax[2].set_title('Histogram of pixel values')
            ax[2].set_xlim(min_val, max_val)

        return mask
    
    def tune_ring(self, orig_mask, radius, width=10, offset=(0, 0), do_plots=True):
        """
        Tunes and visualizes a ring-shaped mask on a given binary mask.
        Parameters
        ----------
        orig_mask : np.ndarray
            The original binary mask (2D array) to which the ring will be applied.
        radius : float
            The radius of the ring to be tuned/applied.
        width : int, optional
            The width of the ring (default is 10).
        offset : tuple of int, optional
            The (x, y) offset to shift the center of the ring (default is (0, 0)).
        do_plots : bool, optional
            If True, displays diagnostic plots showing the original mask, the ring mask, and the mask with the ring applied (default is True).
        Returns
        -------
        ring_mask : np.ndarray
            The mask with the ring applied.
        """
        self.ring_radius = radius
        self.ring_width = width
        self.ring_offset = offset

        ring_mask = orig_mask.copy()
        ring_mask, ring = ring_check(ring_mask, radius, width, offset, return_ring_mask=True)

        if do_plots:
            N_y, N_x = orig_mask.shape
            x_vals = np.arange(N_x) - (N_x-1)/2 - offset[0]
            y_vals = np.arange(N_y) - (N_y-1)/2 + offset[1]
            x_vals, y_vals = np.meshgrid(x_vals, y_vals)
            dist_center = np.sqrt(np.square(x_vals) + np.square(y_vals))

            show_ring = np.stack([orig_mask, orig_mask, orig_mask], axis=-1) * 255
            show_ring[ring] = [255, 0, 0]

            _, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(orig_mask, cmap='gray')
            ax[0].set_title('Original mask')
            ax[0].axis('off')
            ax[1].imshow(show_ring, cmap='gray')
            ax[1].axis('off')
            ax[1].set_title('Ring mask')
            ax[2].imshow(ring_mask, cmap='gray')
            ax[2].axis('off')
            ax[2].set_title('Ring mask applied')
            

            # draw a diagonal red line in first plot and write the radius for certain steps
            x = np.arange(0, N_x)
            #ax[0].plot(x, x, 'r--')
            '''for i in range(0, N_x, 100):
                cur_dist = np.round(dist_center[i, i])
                ax[0].text(i, i, cur_dist, color='red', rotation=45)'''

        return ring_mask

    def tune_neighborhood_cut(self, orig_mask, size=15, threshold=0.01, iterations=1, do_plots=True):
        """
        Refines a binary mask by applying a neighborhood-based filtering operation.

        This method updates the object's neighborhood parameters and applies the
        `neighborhood_check` function to the input mask. Optionally, it displays
        side-by-side plots of the original and processed masks.

        Args:
            orig_mask (np.ndarray): The original binary mask to be refined.
            size (int, optional): The size of the neighborhood window. Defaults to 15.
            threshold (float, optional): The threshold for neighborhood filtering. Defaults to 0.01.
            iterations (int, optional): Number of times to apply the neighborhood filter. Defaults to 1.
            do_plots (bool, optional): Whether to display plots of the original and processed masks. Defaults to True.

        Returns:
            np.ndarray: The refined mask after neighborhood filtering.
        """
        self.neighborhood_size = size
        self.neighborhood_threshold = threshold
        self.neighborhood_iterations = iterations

        neighborhood_mask = orig_mask.copy()
        neighborhood_mask = neighborhood_check(neighborhood_mask, size, threshold, iterations)

        if do_plots:
            _, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(orig_mask, cmap='gray')
            ax[0].set_title('Original mask')
            ax[0].axis('off')
            ax[1].imshow(neighborhood_mask, cmap='gray')
            ax[1].axis('off')
            ax[1].set_title('Neighborhood mask')

        return neighborhood_mask
    
    def tune_surrounded_cut(self, orig_mask, merge_size=100, diag_size=10, cross_size=10, iterations=1, do_plots=True):
        """
        Tunes the parameters for the surrounded cut segmentation process over a specified number of iterations.
        This method applies a sequence of morphological operations (merge, diagonal, and cross checks) to an input mask,
        allowing for iterative refinement of the segmentation. The parameters for each operation can be specified as either
        scalars (applied uniformly across all iterations) or as lists (specifying values per iteration). Optionally, the
        method can plot the original and intermediate masks for visual inspection.
        Args:
            orig_mask (ndarray): The original binary mask to be processed.
            merge_size (int or list of int, optional): Size(s) for the merge operation per iteration. Default is 100.
            diag_size (int or list of int, optional): Size(s) for the diagonal operation per iteration. Default is 10.
            cross_size (int or list of int, optional): Size(s) for the cross operation per iteration. Default is 10.
            iterations (int, optional): Number of iterations to perform. Default is 1.
            do_plots (bool, optional): Whether to display plots of the original and intermediate masks. Default is True.
        Returns:
            ndarray: The final mask after all iterations of surrounded cut operations.
        """
        if type(merge_size) is int:
            merge_size = [merge_size] * iterations
        if type(diag_size) is int:
            diag_size = [diag_size] * iterations
        if type(cross_size) is int:
            cross_size = [cross_size] * iterations

        self.surrounded_merge_size = merge_size
        self.surrounded_diag_size = diag_size
        self.surrounded_cross_size = cross_size
        self.surrounded_iterations = iterations

        surrounded_mask = orig_mask.copy()
        surrounded_masks = []

        for i in range(iterations):
            if merge_size[i] != 0:
                surrounded_mask = surrounded_merge_check(surrounded_mask, merge_size[i])
            if diag_size[i] != 0:
                surrounded_mask = surrounded_diagonal_check(surrounded_mask, diag_size[i])
            if cross_size[i] != 0:
                surrounded_mask = surrounded_cross_check(surrounded_mask, cross_size[i])
            surrounded_masks.append(surrounded_mask.copy())

        if do_plots:
            _, ax = plt.subplots(1, iterations+1, figsize=(5*(iterations+1), 5))
            ax[0].imshow(orig_mask, cmap='gray')
            ax[0].set_title('Original mask')
            ax[0].axis('off')
            for i in range(iterations):
                ax[i+1].imshow(surrounded_masks[i], cmap='gray')
                ax[i+1].set_title(f'Surrounded mask (iteration {i+1})')
                ax[i+1].axis('off')
    

        return surrounded_mask
    
    def fill_boxes(self, mask, do_plots=True):
        """
        Fills black boxes in the given mask and optionally plots the input and filled masks.
        This method applies the `fill_black_boxes` function to the input mask to fill black (empty) regions,
        then performs a "surrounded check" to further process the mask using specified merge and neighborhood sizes.
        Optionally, it displays side-by-side plots of the original and filled masks.
        Args:
            mask (np.ndarray): The input binary mask to process.
            do_plots (bool, optional): If True, displays plots of the input and filled masks. Defaults to True.
        Returns:
            np.ndarray: The processed mask with black boxes filled and surrounded regions merged.
        """
        output_mask = fill_black_boxes(mask) 
        output_mask = surrounded_check(output_mask, self.surrounded_merge_size, self.surrounded_diag_size, self.surrounded_cross_size, 1)

        if do_plots:
            _, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(mask, cmap='gray')
            ax[0].set_title('Input mask')
            ax[0].axis('off')
            ax[1].imshow(output_mask, cmap='gray')
            ax[1].set_title('Filled mask')
            ax[1].axis('off')
    

        return output_mask
    
    def segment_slices(self, scan_slices, n_cpus=4):
        """
        Segments a list of scan slices in parallel using multiple CPUs.

        Args:
            scan_slices (list): List of scan slice data to be segmented.
            n_cpus (int, optional): Number of CPUs to use for parallel processing. Defaults to 4.

        Returns:
            list: List of segmentation masks corresponding to the input scan slices.

        Prints:
            The number of slices being processed and the total processing time.
        """
        N_slices = len(scan_slices)
        print(f'Processing {N_slices} slices...')

        with mp.Pool(n_cpus) as pool:
            start_time = time()
            masks = pool.starmap(segement_slice, [(scan_slices[i], self.threshold, self.ring_radius, self.ring_width, self.ring_offset, self.surrounded_merge_size, self.surrounded_diag_size, self.surrounded_cross_size, self.surrounded_iterations, self.neighborhood_size, self.neighborhood_threshold, self.neighborhood_iterations) for i in range(N_slices)])
            print(f'Processed {N_slices} slices in {time()-start_time:.2f}s')

        return masks
    
    def plot_segmented_slices(self, array, num_slices=40, n_cpus=4, figsize=(5, 100), save_path=None):
        """
        Plots a grid of original and segmented slices from the input array.

        Args:
            array (np.ndarray): 3D array of image slices (shape: [num_slices, height, width]).
            segmenter (HiPCTSegmenter): An instance of HiPCTSegmenter for segmentation.
            num_slices (int): Number of slices to plot (default: 40).
            n_cpus (int): Number of CPUs to use for segmentation (default: 4).
            figsize (tuple): Figure size for the plot (default: (5, 100)).
            save_path (str or None): If provided, saves the figure to this path.

        Returns:
            None
        """
        n = len(array)
        k = min(num_slices, n)
        indices = np.linspace(0, n-1, k, dtype=int)
        selected_images = [array[i] for i in indices]
        masks = self.segment_slices(selected_images, n_cpus=n_cpus)

        fig, ax = plt.subplots(k, 2, figsize=figsize)
        for i in range(k):
            ax[i, 0].imshow(selected_images[i], cmap='gray')
            ax[i, 1].imshow(masks[i], cmap='gray')
            ax[i, 0].set_title(f"Slice {indices[i]}")
            ax[i, 0].axis('off')
            ax[i, 1].axis('off')
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.show()

    def export_params_to_yaml(self, filepath):
        """
        Export all current parameters of the segmenter to a YAML file.
        """
        params = {
            "threshold": self.threshold,
            "ring_radius": self.ring_radius,
            "ring_width": self.ring_width,
            "ring_offset": self.ring_offset,
            "surrounded_merge_size": self.surrounded_merge_size,
            "surrounded_diag_size": self.surrounded_diag_size,
            "surrounded_cross_size": self.surrounded_cross_size,
            "surrounded_iterations": self.surrounded_iterations,
            "neighborhood_size": self.neighborhood_size,
            "neighborhood_threshold": self.neighborhood_threshold,
            "neighborhood_iterations": self.neighborhood_iterations
        }
        with open(filepath, "w") as f:
            yaml.dump(params, f)
        print(f"Parameters exported to {filepath}")

    def load_params_from_yaml(self, filepath):
        """
        Load parameters from a YAML file and set them in the segmenter.
        """
        with open(filepath, "r") as f:
            params = yaml.load(f, Loader=yaml.UnsafeLoader)
        self.threshold = params.get("threshold", self.threshold)
        self.ring_radius = params.get("ring_radius", self.ring_radius)
        self.ring_width = params.get("ring_width", self.ring_width)
        ring_offset_val = params.get("ring_offset", self.ring_offset)
        if isinstance(ring_offset_val, list):
            self.ring_offset = tuple(ring_offset_val)
        else:
            self.ring_offset = ring_offset_val
        self.surrounded_merge_size = params.get("surrounded_merge_size", self.surrounded_merge_size)
        self.surrounded_diag_size = params.get("surrounded_diag_size", self.surrounded_diag_size)
        self.surrounded_cross_size = params.get("surrounded_cross_size", self.surrounded_cross_size)
        self.surrounded_iterations = params.get("surrounded_iterations", self.surrounded_iterations)
        self.neighborhood_size = params.get("neighborhood_size", self.neighborhood_size)
        self.neighborhood_threshold = params.get("neighborhood_threshold", self.neighborhood_threshold)
        self.neighborhood_iterations = params.get("neighborhood_iterations", self.neighborhood_iterations)
        print(f"Parameters loaded from {filepath}")
