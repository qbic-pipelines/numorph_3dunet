import os
import cv2
import numpy as np
import cc3d
import argparse
import mat73
import csv
import pandas as pd 

from datetime import datetime
from scipy.io import loadmat
from scipy import ndimage
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from operator import contains

from numorph_3dunet.unet3d.training import load_old_model
from numorph_3dunet.nuclei.img_utils import calculate_rescaling_intensities, measure_colocalization, remove_touching_df
from numorph_3dunet.nuclei.read_params import read_params_from_csv

# Move the main functionality into a function that can be imported
def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description='Predict cell nuclei')

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
                        help='Measure intensity of co-localizaed channels. Default is false')
    parser.add_argument('--resample_resolution', default=25, metavar='resample_resolution', type=float, nargs=3,
                        help='Resolution of resampled images')
    parser.add_argument('--trained_img_resolution', default=[0.75, 0.75, 2.5], metavar='trained_img_resolution', type=float, nargs=3,
                        help='Resolution of images the model was trained on')
    parser.add_argument('--chunk_size', default=[112, 112, 32], metavar='chunk_size', type=int, nargs=3,
                        help='Chunk size in voxels. Default is 112, 112, 32')


    args = parser.parse_args()
    if args.p and os.path.exists(args.p):
        try: 
            params_dict = read_params_from_csv(args.p)
            for key, value in params_dict.items():
                if hasattr(args, key):
                    setattr(args, key, value)
        except Exception as e:
            print(f"Error reading parameters from CSV: {e}")
            return
    

    

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(int(args.gpu))
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    #initialize variables with args values
    input_img_directory = args.i  # Input image directory
    output_directory = args.o  # Output image directory
    img_list = []  # List of images
    
    print(args)


    ################################
    ## Extra options for running model on GPU with limited memory
    #if args.gpu:
    #    import tensorflow as tf
    #    from keras import backend as k
    #
    #    config = tf.ConfigProto()
    #
    # Don't pre-allocate memory; allocate as-needed
     #   config.gpu_options.allow_growth = True
    
    # Only allow a total of half the GPU memory to be allocated
    #    config.gpu_options.per_process_gpu_memory_fraction = 0.25
    
    # Create a session with the above options specified.
    #    k.tensorflow_backend.set_session(tf.Session(config=config))

    ################################

    # Check if output directory exists and create it if not
    if not os.path.exists(output_directory):
        print(f"Output directory {output_directory} does not exist. Creating it...")
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")

    save_name = os.path.join(output_directory, args.sample_id + ".csv")
    print('Saving results to: ', save_name)


    #####
    # Load model
    # TODO: delete hard coded file path after debugging
    #model_file = "/home/schwitalla/Documents/numorph_3dunet/src/numorph_3dunet/models/075_121_model.h5"
    model = load_old_model(args.model_file)
    print(args)
    # Load mask
    if args.use_mask:
        print('Loading mask...')
        mask = mat73.loadmat(args.mask_file)
        mask = mask['I_mask']
    else:
        mask = np.ones((1, 1, 1))

    mask_resolution = np.repeat(args.resample_resolution, 3)
    print('Mask resolution: ', mask_resolution)

    # Determine chunk sizes while considering scaling and chunk overlap
    res = [i / j for i, j in zip(args.acquired_img_resolution, args.trained_img_resolution)]
    mask_res = [i / j for i, j in zip(args.acquired_img_resolution, mask_resolution)]

    load_chunk_size = [round(i / j) for i, j in zip(args.chunk_size, res)]

    # Taking only channel 1 images (assumed to be cell nuclei)
    files = os.listdir(input_img_directory)
    #print('Files: ', files)

    # Take img_list for each channel. Nuclei is channel should be first
    if not args.measure_coloc:
        n_channels = 1

    if not img_list:
        #print('list is empty')
        for i in range(n_channels):
            #print(i)
            matching = [s for s in files if "C" + str(i + 1) in s]
            #print('Matching: ', matching)

            matching = sorted(matching)
            # Read only the number of images for each chunk
            img_list.append([input_img_directory + '/' + s for s in matching])

    total_slices = len(img_list[0])
    #print('Total slices: ', total_slices)
    # z_pad = np.ceil(total_images / load_chunk_size[2])
    # n_chunks = np.ceil((total_images + z_pad) / load_chunk_size[2]).astype(int)

    # chunk_start = np.arrange(0, n_chunks * (load_chunk_size[2] - overlap[2]), load_chunk_size[2] - overlap[2])
    # chunk_end = chunk_start[1:] + overlap[2]
    # chunk_end = np.append(chunk_end, total_images - 1)

    chunk_start = np.arange(0, total_slices, step=(args.chunk_size[2] - args.chunk_overlap[2]))
    chunk_end = chunk_start + args.chunk_size[2]
    n_chunks = len(chunk_start)

    #print('mask shape [0]: ', mask.shape[0])
    #print('mask shape [1]: ', mask.shape[1])
    # Resize mask to match the number of slices in input image directory
    mask = resize(mask, (mask.shape[0], mask.shape[1], total_slices), order=0)
    mask_res[-1] = 1

    # Calculate rescaling intensity values
    if args.normalize_intensity:
        intensity_values = calculate_rescaling_intensities(img_list[0], sampling_range=10)
        if args.int_threshold is not None:
            int_threshold = rescale_intensity(np.asarray(args.int_threshold, dtype=np.float32),
                                            in_range=intensity_values)

    # Calculate rescaling factor if resolutions are different
    if args.acquired_img_resolution != args.trained_img_resolution and args.resample_chunks:
        rescale_factor = [i / j for i, j in zip(args.acquired_img_resolution, args.trained_img_resolution)]
    else:
        rescale_factor = None

    # Read first image to get sizes
    tempI = cv2.imread(img_list[0][0], -1)
    [rows, cols] = tempI.shape

    # Begin cell counting
    total_cells = 0
    total_time = datetime.now()

    for n in range(n_chunks):
        print('Working on chunk', n + 1, 'out of', n_chunks)
        startTime = datetime.now()
        z_start = chunk_start[n]
        z_end = chunk_end[n]

        # If last chunk, add padding
        if z_end > total_slices:
            z_end = total_slices
            add_end_padding = True
        else:
            add_end_padding = False

        # Skip chunk if nothing is present
        if not mask[:, :, z_start:z_end].any():
            continue

        # Take the mask for the respective z positions
        mask_chunk1 = [cv2.resize(mask[:, :, z], (cols, rows), interpolation=0) for z in range(z_start, z_end)]
        mask_chunk1 = np.asarray(mask_chunk1, dtype=bool)
        mask_chunk = np.swapaxes(mask_chunk1, 0, 1).swapaxes(1, 2)

        # Read images
        print('Reading slices', z_start, 'through', z_end)
        images = [cv2.imread(file, -1) for file in img_list[0][z_start:z_end]]
        images = np.asarray(images, dtype=np.float32)
        images = np.swapaxes(images, 0, 1).swapaxes(1, 2)

        # If last chunk, then pad bottom to make it fit into the model
        if add_end_padding:
            print('Padding Chunk End...')
            end_pad = args.chunk_size[2] - images.shape[2]
            mask_chunk = np.pad(mask_chunk, ((0, 0), (0, 0), (0, end_pad)), 'constant')
            images = np.pad(images, ((0, 0), (0, 0), (0, end_pad)), 'mean')

        # Rescale intensity
        if args.normalize_intensity:
            print('Rescaling Intensity...')
            images_rescaled = rescale_intensity(images, in_range=intensity_values)
        else:
            images_rescaled = images

        # Rescale image size
        if rescale_factor is not None:
            print('Rescaling Size...')
            images_rescaled = ndimage.zoom(images_rescaled, rescale_factor, order=1)

        [rows, cols, slices] = images_rescaled.shape

        # Calculate y and x positions to sample image chunks
        x_positions = np.arange(0, cols, step=(args.chunk_size[0] - args.chunk_overlap[0]))
        y_positions = np.arange(0, rows, step=(args.chunk_size[1] - args.chunk_overlap[1]))
        #print('x_positions: ', x_positions)
        #print('y_positions: ', y_positions)

        n_chunks_c = len(x_positions)
        n_chunks_r = len(y_positions)
        #print('n_chunks_c: ', n_chunks_c)
        #print('n_chunks_r: ', n_chunks_r)
        

        # Append image and mask chunks to list
        img_chunks = []
        msk_chunks = []
        for i in range(n_chunks_r):
            for j in range(n_chunks_c):
                msk_chunk = mask_chunk[y_positions[i]:y_positions[i] + args.chunk_size[0],
                            x_positions[j]:x_positions[j] + args.chunk_size[1], :]
                img_chunk = images_rescaled[y_positions[i]:y_positions[i] + args.chunk_size[0],
                            x_positions[j]:x_positions[j] + args.chunk_size[1], :]

                if img_chunk.shape != tuple(load_chunk_size):
                    # If current chunk is larger than target, we need to crop instead of pad
                    if img_chunk.shape[1] > load_chunk_size[1] or img_chunk.shape[0] > load_chunk_size[0]:
                    # Crop to target size
                        img_chunk = img_chunk[:load_chunk_size[0], :load_chunk_size[1], :]
                        msk_chunk = msk_chunk[:load_chunk_size[0], :load_chunk_size[1], :]
                    else:
                    # Pad to target size
                        pad_c_right = int(load_chunk_size[1] - img_chunk.shape[1])
                        pad_r_bottom = int(load_chunk_size[0] - img_chunk.shape[0])

                        # Debug prints
                        #print(f"Debug - Chunk shapes:")
                        #print(f"Current chunk shape: {img_chunk.shape}")
                        #print(f"Target chunk shape: {load_chunk_size}")
                        #print(f"Calculated padding: right={pad_c_right}, bottom={pad_r_bottom}")

                        msk_chunk = np.pad(msk_chunk, ((0, pad_r_bottom), (0, pad_c_right), (0, 0)), 'constant')
                        img_chunk = np.pad(img_chunk, ((0, pad_r_bottom), (0, pad_c_right), (0, 0)), 'constant')

                msk_chunks.append(msk_chunk)
                img_chunks.append(img_chunk)

        print('Images prepared in: ', datetime.now() - startTime)

        # Run prediction
        # The input shape should be 5 dimensions: (m, n, x, y, z)
        # x, y, z represent the image shape, as you would expect. n is the number of
        # channels. In a standard color video image, you would have 3 channels (red,
        # green, blue). In medical imaging these channels can be separate imaging
        # modalities. m is the batch size or number of samples being passed to the
        # model for training.
        startTime = datetime.now()
        output = []
        img_shape = (1, 1) + tuple(args.chunk_size)
        empty_chunk = np.zeros(img_shape)
        empty_idx = np.zeros(len(img_chunks))
        img_reshaped = []
        msk_reshaped = []
        for idx, chunk in enumerate(img_chunks):
            #print(f"Debug - Reshaping:")
            #print(f"Current chunk size: {img_chunks[idx].shape}")
            #print(f"Target shape: {img_shape}")
            #print(f"Current chunk elements: {img_chunks[idx].size}")
            #print(f"Target elements: {np.prod(img_shape)}")
            img_reshaped.append(np.reshape(img_chunks[idx], img_shape))
            msk_reshaped.append(np.reshape(msk_chunks[idx], img_shape))
            
            if msk_reshaped[idx].any() and (img_reshaped[idx] > int_threshold).any():
                output.append(model.predict(img_reshaped[idx]) * msk_reshaped[idx])
            else:
                output.append(empty_chunk)
                empty_idx[idx] = 1
        output = [np.squeeze(chunk) for chunk in output]
        print('Object mask prediction time elapsed: ', datetime.now() - startTime)

        # Calculate connected components and determine centroid positions
        startTime = datetime.now()
        a = 0
        cen = []
        for i in range(n_chunks_r):
            for j in range(n_chunks_c):
                if empty_idx[a] != 1:
                    # Calculate final prediction mask
                    output_thresh = np.where(output[a] > args.pred_threshold, 1, 0)

                    # Label connected components
                    labels_out = cc3d.connected_components(output_thresh)
                    n_cells = np.max(labels_out)

                    # Find centroids
                    centroids = ndimage.measurements.center_of_mass(output_thresh, labels_out,
                                                                    index=np.arange(1, n_cells + 1))
                    centroids = np.asarray(centroids).round()

                    # Remove cells with low intensity
                    if int_threshold is not None:
                        img_chunk = img_chunks[a]
                        int_high = [img_chunk[tuple(centroids[c].astype(int))] > int_threshold for c in
                                    range(len(centroids))]
                        centroids = centroids[int_high]

                    if centroids.any():
                        # Remove centroids along borders
                        centroids = centroids[centroids[:, 0] > args.chunk_overlap[0] / 2]
                        centroids = centroids[centroids[:, 0] <= args.chunk_size[0] - args.chunk_overlap[0] / 2]

                        centroids = centroids[centroids[:, 1] > args.chunk_overlap[1] / 2]
                        centroids = centroids[centroids[:, 1] <= args.chunk_size[1] - args.chunk_overlap[1] / 2]

                        centroids = centroids[centroids[:, 2] > args.chunk_overlap[2] / 2]
                        centroids = centroids[centroids[:, 2] <= args.chunk_size[2] - args.chunk_overlap[2] / 2]

                        # Adjust centroid positions
                        centroids[:, 0] += y_positions[i]
                        centroids[:, 1] += x_positions[j]
                        centroids[:, 2] += chunk_start[n]

                        # Append to array
                        cen.append(centroids)
                a += 1

        if not cen:
            continue


        cent = np.concatenate(cen)
        cent = cent[cent[:, 2].argsort()]

        print('Postprocessing time elapsed: ', datetime.now() - total_time)
        print('Nuclei counted: ', cent.shape[0])

        total_cells += cent.shape[0]
        print('Total nuclei counted: ', total_cells)
        
        # Continue if no nuclei present
        if total_cells == 0:
            continue

        # Get mask structure id's
        if args.use_mask:
            structure_idx = [mask[tuple(np.floor(c * mask_res).astype(int))] for c in cent]
        else:
            structure_idx = np.ones(cent.shape[0])
        cent = np.append(cent, np.array(structure_idx)[:, None], axis=1)

        # Remove stray cells with no id
        rm_idx = cent[:, 3] == 0
        cent = cent[~rm_idx, :]
        total_cells += -sum(rm_idx)
        print('Removed ' + str(sum(rm_idx)) + ' empty nuclei')

        # Continue if no nuclei present
        if not cent.any():
            continue

        # Remove touching cells
        cent_rm = remove_touching_df(cent, radius=args.tree_radius)
        ncent_rm = cent.shape[0] - cent_rm.shape[0]
        total_cells += -ncent_rm
        print('Removed ' + str(ncent_rm) + ' touching nuclei')
        cent = cent_rm

        # Measure intensities in other channels
        if args.measure_coloc and cent.any():
            print('Measuring co-localization...')
            cent[:, 2] -= z_start
            for i in range(n_channels):
                intensities = measure_colocalization(cent, img_list[i][z_start:z_end])

                # Throwing errors
                cent = np.append(cent, intensities[:, None], axis=1)
            cent[:, 2] += z_start

        # If called by MATLAB, adjust for base 1 indexing
        #if from_matlab:
        #    cent += 1

        # Write .csv file
        if n == 0:
            np.savetxt(save_name, cent.round().astype(int), delimiter=",", fmt='%u')
        else:
            with open(save_name, "ab") as f:
                np.savetxt(f, cent.round().astype(int), delimiter=",", fmt='%u')

    # One final pass on all centroids to remove potentially touching nuclei in z



    print('Total nuclei counted: ', total_cells)
    print('Total time elapsed: ', datetime.now() - total_time)


    # Generate counts table from final CSV file
    print("\nGenerating region-based cell counts table...")
    try:
        # Check if centroids file exists and has data
        if os.path.exists(save_name) and os.path.getsize(save_name) > 0:
            # Read all centroids
            centroids_data = np.loadtxt(save_name, delimiter=",")
            
            # Handle single row case
            if centroids_data.ndim == 1:
                centroids_data = centroids_data.reshape(1, -1)
                
            # Check if annotations column exists (column 4, index 3)
            if centroids_data.shape[1] > 3:
                # Extract annotations (structure/region IDs)
                annotations = centroids_data[:, 3]
                
                # Count cells by region using numpy's unique function with return_counts
                regions, counts = np.unique(annotations, return_counts=True)
                
                # Combine into a table
                counts_table = np.column_stack((regions, counts))
                
                # Sort by region ID
                counts_table = counts_table[counts_table[:, 0].argsort()]
                
                # Save to CSV with header
                counts_file = os.path.splitext(save_name)[0] + "_counts.csv"
                header = "region_id,cell_count"
                np.savetxt(counts_file, counts_table, 
                        delimiter=",", fmt='%d', 
                        header=header, comments='')
                
                print(f"Cell counts table saved to: {counts_file}")
                print("\nRegion counts summary:")
                total_regions = len(regions)
                display_regions = min(10, total_regions)
                
                for i in range(display_regions):
                    region, count = counts_table[i]
                    print(f"  Region {int(region)}: {int(count)} cells")
                    
                if total_regions > 10:
                    remaining = sum(counts_table[10:, 1])
                    print(f"  ... and {total_regions-10} more regions with {remaining} cells")
                    
                print(f"\nTotal: {len(annotations)} cells across {total_regions} regions")
            else:
                print("No region annotations found in centroids data")
        else:
            print("No cells detected or CSV file not found")
    except Exception as e:
        print(f"Error generating counts table: {str(e)}")

# Call main() when this script is run directly
if __name__ == "__main__":
    main()