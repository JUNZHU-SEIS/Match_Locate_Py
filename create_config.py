import yaml
dict = {}
dict['eps'] = 1e-15
dict['sampling_rate'] = 100
dict['snr_threshold'] = 1
dict['win0'],dict['win1'] = 1,3
dict['win0_save_tplt'],dict['win1_save_tplt'] = 20,80
dict['halfwin_for_local_mad'] = 5
dict['number_stations_threshold'],dict['mad_threshold'],dict['too_close_detections_to_remain'] = 5,9,6
dict['local_mad_threshold'] = 9
dict['radius'] = {'nx':1,'ny':1,'nz':1,'dx':.01,'dy':.01,'dz':1,'duplicate_nx':2,'duplicate_ny':2,'duplicate_nz':2}
dict['ctlg_path'] = '../GrowClust/OUT.0/out.growclust_cat'
dict['ctlg_nrows'] = None
dict['root'] = '../Data/waveform_sac_filtered'
dict['fvelo'] = '../pyocto/velocitymodel.txt'
dict['tplt_folder'] = 'template'
dict['tplt_log'] = 'pdf'
dict['tplt_path'] = 'tplt.hdf5'
dict['detection_folder'] = 'catalog'
dict['log'] = 'pdf'
dict['minimum_cc'] = .3
dict['number_high_cc_channels'] = 5
dict['max_number_channels'] = 100
dict['plot'] = 0
dict['continue_previous_experiment'] = 0
dict['max_number_chan_in_one_station'] = 1

with open('config.yml','w') as outfile:yaml.dump(dict,outfile,default_flow_style=False)
with open("config.yml",'r') as stream:d = yaml.safe_load(stream)
print(d)
