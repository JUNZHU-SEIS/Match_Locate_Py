import yaml
dict = {}
dict['local_mad_threshold'] = 9
dict['peakedness_cc_threshold'] = 1.9

with open('config_postprocess.yml','w') as outfile:yaml.dump(dict,outfile,default_flow_style=False)
with open("config_postprocess.yml",'r') as stream:d = yaml.safe_load(stream)
print(d)