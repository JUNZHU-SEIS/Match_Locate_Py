This is a PyTorch version GPU-Match&Locate, modified from [GPU-Match&Locate1.0](https://github.com/MinLiu19/GPU-MatchLocate1.0).\
New features:
1. Automatically checking active stations for each candidate-template pair.
2. Continue your last experiments if your program is terminated by memory limit.

To run this program, two kinds of data are required.
1. Continuous waveform data preprocessed by [LOC-FLOW](https://github.com/Dal-mzhang/LOC-FLOW/blob/main/Data/waveform_download_mseed.py)
2. Catlog of relocated earthquakes by [GrowClust](https://github.com/dttrugman/GrowClust/blob/master/EXAMPLE/OUT/out.growclust_cat)

Some [`default configugrations`](./config.yml) of this program are set by [`create_config.py`](./create_config.py).\
Because there are too many configurations, another way to set them is as follows:
```
python GPU_MATCH_LOCATE.py --cuda=1 --mad_threshold=10 --halfwin_for_local_mad=5 --local_mad_threshold=10 --snr_threshold=1 --minimum_cc=.3 --number_high_cc_channels=5 --number_stations_threshold=4 --plot=1 --too_close_detections_to_remain=6 --max_number_stations_threshold=80
```
Prerequisite:
- Nvidia-GPU
- PyTorch
- Obspy
- h5py