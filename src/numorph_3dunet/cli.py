#!/usr/bin/env python
"""
Command-line interface for NuMorph 3DUnet.
"""

# load libraries
import argparse
import os

from numorph_3dunet.nuclei.read_params import read_params_from_csv
from numorph_3dunet.nuclei.generate_chunks import generate_chunks




def cl_parser():
    """
    command-line argument parser for NuMorph 3DUnet.
    """

    parser = argparse.ArgumentParser(description="NuMorph 3DUnet Command-Line Interface")

    # parameters from csv file 
    parser.add_argument('-p', default='', metavar='p', type=str, nargs='?', 
                        help='Path to csv file with parameters.')
    #params when using without param file 
    parser.add_argument('-i', required=True, metavar='i', type=str, nargs='?',
                        help='Input image directory')
    parser.add_argument('-o', required=True, metavar='i', type=str, nargs='?',
                        help='Output image directory')
    parser.add_argument('--n_channels', required=False ,metavar='n_channels', type=int, nargs='?',
                        help='Number of channels')
    parser.add_argument('--sample_id', required=False ,metavar='save_name', type=str, nargs='?',
                        help='sample_id')
    parser.add_argument('--model_file', required=False ,metavar='model', type=str, nargs='?',
                        help='Model file')
    parser.add_argument('-gpu', default=0, metavar='g', type=int, nargs='?',
                        help='GPU tag')
    parser.add_argument('--chunk_overlap', default=[16, 16, 8] ,metavar='chunk_overlap', type=int, nargs=3,
                        help='Overlap between chunks in voxels. Default is 16, 16, 8')
    parser.add_argument('--pred_threshold', default=0.5 ,metavar='pred_threshold', type=float, nargs='?',
                        help='Prediction threshold. Default is 0.5')
    parser.add_argument('--int_threshold', default=200 , metavar='int_threshold', type=float, nargs='?',
                        help='Minimum intensity of positive cells. Default is 200')
    parser.add_argument('--normalize_intensity', default=True, metavar='normalize_intensity', type=bool, nargs='?',
                        help='Whether to normalize intensities using min/max. Default is true')
    parser.add_argument('--resample_chunks', default=False, metavar='resample_chunks', type=bool, nargs='?',
                        help='Whether to resample image to match trained image resolution. Note: increases computation time. Default is false')
    parser.add_argument('--use_mask', default=False, metavar='use_mask', type=bool, nargs='?',
                        help='Use mask')
    parser.add_argument('--mask_file', default='', metavar='mask_file', type=str, nargs='?',
                        help='Mask file')
    parser.add_argument('--acquired_img_resolution', default=[0.75, 0.75, 4], metavar='acquired_img_resolution', type=float, nargs=3,
                        help='Resolution of acquired images')
    parser.add_argument('--tree_radius', default=2 ,metavar='tree_radius', type=float, nargs='?',
                        help='Pixel radius for removing centroids near each other')
    parser.add_argument('--measure_coloc', default=False, metavar='measure_coloc', type=bool, nargs='?',
                        help='Measure intensity of co-localized channels. Default is false')
    parser.add_argument('--resample_resolution', default=25, metavar='resample_resolution', type=float, nargs=3,
                        help='Resolution of resampled images')
    parser.add_argument('--trained_img_resolution', default=[0.75, 0.75, 2.5], metavar='trained_img_resolution', type=float, nargs=3,
                        help='Resolution of images the model was trained on')
    parser.add_argument('--chunk_size', default=[112, 112, 32], metavar='chunk_size', type=int, nargs=3,
                        help='Chunk size in voxels. Default is 112, 112, 32')


    return parser


def run():
    """
    Entry point for the prediction command.
    This function is called when running `numorph_3dunet.predict`
    """
    parser = cl_parser()
    args = parser.parse_args()

    # if params are provided in a csv file, read them and overwrite the corresponding args
    if args.p and os.path.exists(args.p):
        try: 
                params_dict = read_params_from_csv(args.p)
                for key, value in params_dict.items():
                    if hasattr(args, key):
                        setattr(args, key, value)
        except Exception as e:
            print(f"Error reading parameters from CSV: {e}")
            return

    # Set GPU environment variable
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(int(args.gpu))
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    

    generate_chunks(input_img_directory=args.i,
                    output_img_directory=args.o,
                    n_channels=args.n_channels,
                    sample_id=args.sample_id,
                    model_file=args.model_file,
                    chunk_overlap=args.chunk_overlap,
                    pred_threshold=args.pred_threshold,
                    int_threshold=args.int_threshold,
                    normalize_intensity=args.normalize_intensity,
                    resample_chunks=args.resample_chunks,
                    use_mask=args.use_mask,
                    mask_file=args.mask_file,
                    acquired_img_resolution=args.acquired_img_resolution,
                    tree_radius=args.tree_radius,
                    measure_coloc=args.measure_coloc,
                    resample_resolution=args.resample_resolution,
                    trained_img_resolution=args.trained_img_resolution,
                    chunk_size=args.chunk_size)

if __name__ == "__main__":
    run()
