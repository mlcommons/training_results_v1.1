git clone https://github.com/SAITPublic/MLPerf_Training_v1.1 samsung_training_v1.1

# Copy samsung licensed source codes which do not exist in current directory  
cp samsung_training_v1.1/run_pretraining.py .
cp samsung_training_v1.1/utils.py .
cp samsung_training_v1.1/cleanup_scripts/run_split_and_chop_hdf5_files.py cleanup_scripts/

rm -rf samsung_training_v1.1



