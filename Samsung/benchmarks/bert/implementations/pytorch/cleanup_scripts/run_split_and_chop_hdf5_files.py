# Copyright (C) 2021 Samsung Electronics Co. LTD

# This software is a property of Samsung Electronics.
# No part of this software, either material or conceptual may be copied or distributed, transmitted,
# transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
# electronic, mechanical, manual or otherwise, or disclosed
# to third parties without the express written permission of Samsung Electronics.

import os
import time
import glob
import argparse

import h5py
import numpy as np


# Set required params
INPUT_PATH = None
OUTPUT_PATH = None
N_OUTPUT_SHARDS = 2048  # the number of output shards for each subdir

split_subdir = [128, 256, 384, 512]
bins_tuple = [(1, 128), (129, 256), (257, 384), (385, 512)]
n_bins = len(bins_tuple)

max_pred_per_seq = 76
seq_length = 512

ofile_prefix_arr = list()
ofile_suffix = None
input_files = None


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir",
                        default=None,
                        type=str,
                        help="The input directory of raw data")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        help="The output directory of splitting data")

    parser.add_argument("--n_output_shards",
                        default=2048,
                        type=int,
                        help="The number of outputing shards for each bin-data.")

    args = parser.parse_args()
    return args


class BufferedSeqWriter(object):
    hdf5_compression_method = None

    def __init__(self, n_samples, n_shards, ofile_prefix, ofile_suffix):
        self.n_samples_per_file_nominal = (n_samples + n_shards - 1) // n_shards
        self.n_excess = n_shards * self.n_samples_per_file_nominal - n_samples
        self.ofile_handles = [h5py.File(ofile_prefix + str(x) + ofile_suffix, 'w') for x in range(n_shards)]

        self.n_samples_in_this_shard = self.n_samples_per_file_nominal
        if self.n_excess > 0:
            self.n_samples_in_this_shard -= 1
            self.n_excess -= 1

        self.o_input_ids = np.ndarray((self.n_samples_in_this_shard, seq_length))
        self.o_input_masks = np.ndarray((self.n_samples_in_this_shard, seq_length))
        self.o_segment_ids = np.ndarray((self.n_samples_in_this_shard, seq_length))
        self.o_masked_lm_positions = np.ndarray((self.n_samples_in_this_shard, max_pred_per_seq))
        self.o_masked_lm_ids = np.ndarray((self.n_samples_in_this_shard, max_pred_per_seq))
        self.o_next_sentence_labels = np.ndarray((self.n_samples_in_this_shard))

        # which output file
        self.ofile_idx = 0
        # index into an individual data element of an output file
        self.ofile_entry_idx = 0

    def write_sequence(self, seq_data, in_idx):
        input_ids, input_masks, segment_ids, masked_lm_pos, masked_lm_ids, next_lable = seq_data
        self.o_input_ids[self.ofile_entry_idx] = input_ids[in_idx]
        self.o_input_masks[self.ofile_entry_idx] = input_masks[in_idx]
        self.o_segment_ids[self.ofile_entry_idx] = segment_ids[in_idx]
        self.o_masked_lm_positions[self.ofile_entry_idx] = masked_lm_pos[in_idx]
        self.o_masked_lm_ids[self.ofile_entry_idx] = masked_lm_ids[in_idx]
        self.o_next_sentence_labels[self.ofile_entry_idx] = next_lable[in_idx]
        self.ofile_entry_idx += 1

        if self.ofile_entry_idx == self.n_samples_in_this_shard:
            self.ofile_handles[self.ofile_idx].create_dataset("input_ids", data=self.o_input_ids, dtype='i4',
                                                              compression=self.hdf5_compression_method)
            self.ofile_handles[self.ofile_idx].create_dataset("input_mask", data=self.o_input_masks, dtype='i1',
                                                              compression=self.hdf5_compression_method)
            self.ofile_handles[self.ofile_idx].create_dataset("segment_ids", data=self.o_segment_ids, dtype='i1',
                                                              compression=self.hdf5_compression_method)
            self.ofile_handles[self.ofile_idx].create_dataset("masked_lm_positions", data=self.o_masked_lm_positions,
                                                              dtype='i4', compression=self.hdf5_compression_method)
            self.ofile_handles[self.ofile_idx].create_dataset("masked_lm_ids", data=self.o_masked_lm_ids, dtype='i4',
                                                              compression=self.hdf5_compression_method)
            self.ofile_handles[self.ofile_idx].create_dataset("next_sentence_labels", data=self.o_next_sentence_labels,
                                                              dtype='i1', compression=self.hdf5_compression_method)
            self.ofile_handles[self.ofile_idx].flush()
            self.ofile_handles[self.ofile_idx].close()

            self.ofile_idx += 1
            self.ofile_entry_idx = 0

            self.n_samples_in_this_shard = self.n_samples_per_file_nominal
            if self.ofile_entry_idx < self.n_excess:
                self.n_samples_in_this_shard -= 1
                self.n_excess -= 1

            self.o_input_ids = np.ndarray((self.n_samples_in_this_shard, seq_length))
            self.o_input_masks = np.ndarray((self.n_samples_in_this_shard, seq_length))
            self.o_segment_ids = np.ndarray((self.n_samples_in_this_shard, seq_length))
            self.o_masked_lm_positions = np.ndarray((self.n_samples_in_this_shard, max_pred_per_seq))
            self.o_masked_lm_ids = np.ndarray((self.n_samples_in_this_shard, max_pred_per_seq))
            self.o_next_sentence_labels = np.ndarray((self.n_samples_in_this_shard))


def get_bin_idx(seq_len):
    return [i for i, bin_range in enumerate(bins_tuple) if bin_range[0] <= seq_len <= bin_range[1]][0]


def main():
    start = time.time()

    n_samples = 0
    n_samples_arr = [0] * n_bins

    for idx, ifile in enumerate(input_files):
        print("Scanning:", ifile, " --  Progress:", idx + 1, '/', len(input_files))
        h5_ifile = h5py.File(ifile, 'r')

        f_next_sentence_labels = h5_ifile['next_sentence_labels'][:]
        n_samples += f_next_sentence_labels.shape[0]

        f_input_masks = h5_ifile['input_mask'][:]
        for i in range(f_input_masks.shape[0]):
            seq_len = np.sum(f_input_masks[i, :])
            bin_idx = get_bin_idx(seq_len)
            n_samples_arr[bin_idx] += 1
        h5_ifile.close()

    print("n_samples_total: ", n_samples)
    for i, bin_range in enumerate(bins_tuple):
        print("n_samples in bins [{}, {}]: {}".format(bin_range[0], bin_range[1], n_samples_arr[i]))

    print("\n\nStarting to split the raw dataset...")
    writer_list = list()
    for i in range(n_bins):
        writer_list.append(BufferedSeqWriter(n_samples_arr[i], N_OUTPUT_SHARDS, ofile_prefix_arr[i], ofile_suffix))

    for idx, ifile in enumerate(input_files):
        print("Processing:", ifile, " --  Progress:", idx + 1, '/', len(input_files))
        h5_ifile = h5py.File(ifile, 'r')

        ifile_entry_idx = 0
        f_input_ids = h5_ifile['input_ids'][:]
        f_input_masks = h5_ifile['input_mask'][:]
        f_segment_ids = h5_ifile['segment_ids'][:]
        f_masked_lm_positions = h5_ifile['masked_lm_positions'][:]
        f_masked_lm_ids = h5_ifile['masked_lm_ids'][:]
        f_next_sentence_labels = h5_ifile['next_sentence_labels'][:]

        h5_ifile.close()

        # This could be vectorized but keeping it simple due to lack of time
        while ifile_entry_idx < f_input_ids.shape[0]:
            seq_len = np.sum(f_input_masks[ifile_entry_idx, :])
            bin_idx = get_bin_idx(seq_len)
            seq_data = (f_input_ids, f_input_masks, f_segment_ids, f_masked_lm_positions,
                        f_masked_lm_ids, f_next_sentence_labels)
            writer_list[bin_idx].write_sequence(seq_data, ifile_entry_idx)

            ifile_entry_idx += 1
    print("Finish splitting and sharding raw training data, consumed time: {} s".format(time.time()-start))


if __name__ == '__main__':
    args = parse_arguments()
    INPUT_PATH = args.input_dir
    OUTPUT_PATH = args.output_dir
    N_OUTPUT_SHARDS = args.n_output_shards

    assert INPUT_PATH is not None, "INPUT_PATH can not be None"
    assert OUTPUT_PATH is not None, "OUTPUT_PATH can not be None"
    assert N_OUTPUT_SHARDS > 0, "N_OUTPUT_SHARDS is invalid"

    for path in [INPUT_PATH, OUTPUT_PATH]:
        if not os.path.exists(path):
            print("Not exist: ", path)
            exit(1)

    for subdir in split_subdir:
        bin_path = os.path.join(OUTPUT_PATH, str(subdir))
        if not os.path.exists(bin_path):
            os.makedirs(os.path.join(OUTPUT_PATH, str(subdir)))

    for i in range(n_bins):
        ofile_prefix_arr.append(os.path.join(OUTPUT_PATH, str(split_subdir[i]), 'part_'))
    ofile_suffix = '_of_' + str(N_OUTPUT_SHARDS) + '.hdf5'

    # read input shards and construct output file prefix
    input_files = sorted(glob.glob(INPUT_PATH + '/part*', recursive=False))
    print('n_input_shards =', len(input_files))
    print('n_output_shards =', N_OUTPUT_SHARDS)

    main()
