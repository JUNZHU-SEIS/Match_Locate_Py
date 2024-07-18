This is a PyTorch version GPU-Match&Locate, modified from (GPU-Match&Locate1.0)[https://github.com/MinLiu19/GPU-MatchLocate1.0].

To run this program, two kinds of data are required.
1. Continuous waveform data preprocessed by [LOC-FLOW](https://github.com/Dal-mzhang/LOC-FLOW/blob/main/Data/waveform_download_mseed.py)
2. Catlog of relocated earthquakes by GrowClust
The configugrations for Match&Locate can be set by (this file)[./create_config.py]

Python environment: Nvidia-GPU and PyTorch