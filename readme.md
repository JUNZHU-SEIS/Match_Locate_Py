This is a PyTorch version GPU-Match&Locate modified from [GPU-Match&Locate1.0](https://github.com/MinLiu19/GPU-MatchLocate1.0).\
New features of this version:
1. Automatically checking active stations for each candidate-template pair
2. Continue your last experiment if your program is terminated by accidental memory limit

To run this program, two kinds of data are required.
1. Continuous waveform data preprocessed by [LOC-FLOW](https://github.com/Dal-mzhang/LOC-FLOW/blob/main/Data/waveform_download_mseed.py)
2. Catlog of relocated earthquakes by [GrowClust](https://github.com/dttrugman/GrowClust/blob/master/EXAMPLE/OUT/out.growclust_cat)

Some [`default configugrations`](./config.yml) of this program are set by [`create_config.py`](./create_config.py). Because there are too many configurations, an easy way to set/change them is as follows:
```
python GPU_MATCH_LOCATE.py --cuda=1 --mad_threshold=10 --halfwin_for_local_mad=5 --local_mad_threshold=10 --snr_threshold=1 --minimum_cc=.3 --number_high_cc_channels=5 --number_stations_threshold=4 --plot=1 --too_close_detections_to_remain=6 --max_number_stations_threshold=80
```
Once you run the command above, two folders will be created: `template` and `experiments`.
1. The `template` folder saves the template waveform in the format of hdf5 and the images of the waveform in the path (when `--plot=1`)
2. The `experiments` folder logs the results and the configurations of every experiment. Its subfolders ared named in the format of `YYYYMMDDTHHMMSS` (i.e., the local time when you run the command). In each subfolder, two sub-subfolders exist: `catalog` and `pdf` save the detections and images, respectively

If this is not your first time to run the code, I suggest you cancel the process of marking templates by commenting the 235th line in [`GPU_MATCH_LOCATE.py`](./GPU_MATCH_LOCATE.py).

Prerequisite:
* Nvidia-GPU
* PyTorch
* Obspy
* h5py

Any suggestions on the code are welcome. Please feel free to contact me via [email](mailto:jun__zhu@outlook.com) or GitHub issues.