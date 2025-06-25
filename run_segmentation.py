import argparse
import os
from hipct_segmenter import HiPCTSegmenter
from hipct_segmenter import load_hip_ct_scan_to_np
from hipct_segmenter import np_to_h5

def main():
    parser = argparse.ArgumentParser(description="Run segmentation with config file.")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input HiP-CT scan, can be a directory of JP2/tiff or a single h5 file.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the segmentation results.')
    parser.add_argument('--ncpus', type=int, default=4, help='Number of CPUs to use for segmentation (default: 4).')
    
    args = parser.parse_args()

    segmentator = HiPCTSegmenter()
    segmentator.load_params_from_yaml(args.config)

    array = load_hip_ct_scan_to_np(args.input)

    masks = segmentator.segment_slices(array, n_cpus=args.ncpus)

    np_to_h5(masks, path_dicom=args.input, path_h5=args.output)



if __name__ == "__main__":
    main()