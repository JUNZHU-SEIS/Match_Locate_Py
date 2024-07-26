import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import pandas as pd
import os,h5py,glob,yaml,argparse,torch
from datetime import datetime
from obspy import read,UTCDateTime
from obspy.taup import TauPyModel
from obspy.taup.taup_create import build_taup_model
import numpy as np
from obspy.geodetics.base import locations2degrees as loc2deg
from math import log10
from scipy.signal import detrend,peak_widths
R = 6371
deg2km = np.pi*R/180
experiment_dir = 'experiments'
debug_dir = 'debug'

def calc_relative_time(t):return 3600*t.hour+60*t.minute+t.second+t.microsecond/1e6
def plot_template(data,components,p,s,snr_p,snr_s,tag,sampling_rate,folder):
	_,ax = plt.subplots(len(components),1,sharex=True,gridspec_kw={'hspace':0})
	t = np.arange(data.shape[-1])/sampling_rate
	if p>0 and s>0:
		e = int(sampling_rate*(2*(s-p)+s))
	elif p>0:e = int(sampling_rate*(2*p))
	elif s>0:e = int(sampling_rate*(2*s))
	else:e = len(t)
	e = min(e,len(t))
	for a,x,rp,rs,comp in zip(ax,data,snr_p,snr_s,components):
		if p>0:
			a.axvline(x=p,color='b',ls='--',label='predict P')
			if p<t[e-1]:a.text(p,min(x[:e]),'SNR=%.1f'%rp,ha='right',va='bottom')
		if s>0:
			a.axvline(x=s,color='r',ls='--',label='predict S')
			if s<t[e-1]:a.text(s,min(x[:e]),'SNR=%.1f'%rs,va='bottom')
		a.plot(t[:e],x[:e],lw=.8,color='gray');a.set_ylabel(comp);a.set_yticks([])
	a.set_xlim(t[0],t[e-1])
	a.set_xlabel('Seconds since the origin time')
	ax[0].set_title(tag)
	ax[0].legend()
	plt.tight_layout()
	plt.savefig(os.path.join(folder,'%s.pdf'%str(tag)))
	plt.close()

def normalize(x):
	x -= np.mean(x,axis=1,keepdims=True)
	x = detrend(x,axis=1)
	return x

def get_snr(d,tp,ts,sampling_rate,win=3,snr=-999):
	x = d.copy()
	win = int(win*sampling_rate)
	tp = int(tp*sampling_rate)
	ts = int(ts*sampling_rate)
	snr_p,snr_s = [snr]*len(d),[snr]*len(d)
	npts = x.shape[-1]
	w = tp
	win = min(w,win,npts-w)
	if win>0:
		signal = normalize(x[:,w:w+win])
		noise = normalize(x[:,w-win:w])
		numerator = np.sum(signal**2,axis=1)
		denominator = np.sum(noise**2,axis=1)
		for i,(a,b) in enumerate(zip(numerator,denominator)):
			snr_p[i]=(10*log10(a/b) if a*b>0 else snr)
	w = ts
	win = min(w,win,npts-w)
	if win>0:
		signal = normalize(x[:,w:w+win])
		noise = normalize(x[:,w-win:w])
		numerator = np.sum(signal**2,axis=1)
		denominator = np.sum(noise**2,axis=1)
		for i,(a,b) in enumerate(zip(numerator,denominator)):
			snr_s[i]=(10*log10(a/b) if a*b>0 else snr)
	return snr_p,snr_s

def slowness(arr):
	dt_dgc = arr.ray_param*np.pi/180
	vslowness_for_source_depth_offset = -arr.ray_param/(R*np.tan(arr.takeoff_angle*np.pi/180))
	return dt_dgc,vslowness_for_source_depth_offset

def calc_travel_time_and_slowness(lat,lon,dep,stla,stlo,stel,vmodel):
	focal = max(dep+stel/1e3,0)
	gcarc = loc2deg(lat,lon,stla,stlo)
	tp = vmodel.get_travel_times(focal,distance_in_degree=gcarc,phase_list=['p','P'])
	ts = vmodel.get_travel_times(focal,distance_in_degree=gcarc,phase_list=['s','S'])
	p,p_h,p_v = -1,-1,-1
	s,s_h,s_v = -1,-1,-1
	if len(tp):
		p = tp[0].time
		p_h,p_v = slowness(tp[0])
	if len(ts):
		s = ts[0].time
		s_h,s_v = slowness(ts[0])
	dist = np.sqrt((gcarc*deg2km)**2+focal**2)
	return p,p_h,p_v,s,s_h,s_v,dist

def read_growclust(p,nrows=None):
	df = pd.read_csv(p,sep='\s+',names=['yr','mon',
		'day','hr','min','sec','evid','LAT','LON',
		'DEPTH','mag','qID','cID','nbranch',
		'qnpair','qndiffP','qndiffS','rmsP','rmsS',
		'eh','ez','et','latC','lonC','depC'])
	df['time'] = [UTCDateTime('%d%02d%02dT%02d:%02d:%06.3f'%(yr,mon,day,hr,min,sec)) for yr,mon,day,hr,min,sec in zip(
		df['yr'],df['mon'],df['day'],df['hr'],df['min'],df['sec'])]
	return df if nrows==None else df.iloc[:nrows]

def create_velocity_model(path,vmout):
	vm = pd.read_csv(path)
	f = open(vmout,'w')
	f.write('eg\n'*2)
	for dep,vp,vs in zip(vm['depth'],vm['vp'],vm['vs']):
		f.write('%f %f %f 1\n'%(dep,vp,vs))
	f.close();build_taup_model(vmout)
	return TauPyModel(model=vmout.split('.')[0])

def mark_templates(args,rows=None):
	df = read_growclust(args['ctlg_path'],nrows=args['ctlg_nrows'])
	print(df)
	tplt_path = args['tplt_path']
	root = args['root']
	vmodel = create_velocity_model(args['fvelo'],'eg.tvel')
	sampling_rate = args['sampling_rate']
	win0,win1 = args['win0_save_tplt'],args['win1_save_tplt']
	log = args['tplt_log']
	if os.path.exists(tplt_path):os.unlink(tplt_path)
	if args['plot']:
		for x in glob.glob(os.path.join(log,'*')):os.unlink(x)
	f = h5py.File(tplt_path,mode='w');grp = f.create_group('waveform')
	for i in range(len(df) if rows==None else rows):
		etry = df.iloc[i]
		t = etry['time']
		lat,lon,dep = etry['LAT'],etry['LON'],etry['DEPTH']
		print(t,lat,lon,dep,i)
		date = '%04d%02d%02d'%(t.year,t.month,t.day)
		flst = glob.glob(os.path.join(root,date,'*Z'))
		for fname in flst:
			fname = '.'.join(fname.split('.')[:-1]+['*'])
			print(fname,t)
			st = read(fname,starttime=t-win0,endtime=t+win1).sort()
			components = [tr.stats.component for tr in st]
			channels = [tr.stats.channel for tr in st]
			st_original = st.copy()
			st_original.trim(starttime=t-win0,endtime=t+win1,pad=True,fill_value=0)
			st.detrend('demean').detrend('linear').taper(.05,max_length=5).filter('bandpass',freqmin=2,freqmax=15)
			st.trim(starttime=t-win0,endtime=t+win1,pad=True,fill_value=0)
			meta = st[-1].stats.sac
			stlo,stla,stel = meta.stlo,meta.stla,meta.stel
			data = np.vstack([tr.data for tr in st_original])
			tag = '%d/%s'%(i,fname.split('/')[-1])
			p,p_h,p_v,s,s_h,s_v,dist = calc_travel_time_and_slowness(lat,lon,dep,stla,stlo,stel,vmodel)
			snr_p,snr_s = get_snr(data,p+win0,s+win0,sampling_rate)
			if args['plot']:plot_template(data,components,p+win0,s+win0,snr_p,snr_s,tag.replace('/','.'),sampling_rate,log)
			dset = grp.create_dataset(tag,(1,len(components),data.shape[-1]),dtype=np.float32)
			dset[0] = data
			dset.attrs['otime'] = str(t)
			dset.attrs['reference'] = win0
			dset.attrs['channels'] = channels
			dset.attrs['tp'] = p
			dset.attrs['p_slowness_h'] = p_h
			dset.attrs['p_slowness_v'] = p_v
			dset.attrs['snr_p'] = snr_p
			dset.attrs['ts'] = s
			dset.attrs['s_slowness_h'] = s_h
			dset.attrs['s_slowness_v'] = s_v
			dset.attrs['snr_s'] = snr_s
			dset.attrs['stlo'] = stlo
			dset.attrs['stla'] = stla
			dset.attrs['stel'] = stel
			dset.attrs['dist'] = dist
	f.close()
	return

def read_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--debug", default=0, type=int, help="Test mode")
	parser.add_argument("--eps", default=None, type=float, help="Non-zero denominator")
	parser.add_argument("--sampling_rate", default=None, type=int, help="How many grids on the x axis")
	parser.add_argument("--number_stations_threshold_for_template", default=None, type=int, help="How many stations to justify a template")
	parser.add_argument("--number_high_cc_stations_threshold", default=None, type=int, help="How many stations of high CC to justify a detection")
	parser.add_argument("--number_high_cc_channels_threshold", default=None, type=float, help="How many channels of high CC to justify a detection")
	parser.add_argument("--max_number_channels", default=None, type=int, help="At most how many channels to make a template")
	parser.add_argument("--snr_threshold", default=None, type=float, help="SNR threhold for templates")
	parser.add_argument("--minimum_cc", default=None, type=float, help="High CC threshold")
	parser.add_argument("--mad_threshold", default=None, type=float, help="MAD threhold for templates")
	parser.add_argument("--too_close_detections_to_remain", default=None, type=int, help="Threshold for close occurrences")
	parser.add_argument("--nx", default=None, type=int, help="How many grids on the x axis")
	parser.add_argument("--ny", default=None, type=int, help="How many grids on the y axis")
	parser.add_argument("--nz", default=None, type=int, help="How many grids on the z axis")
	parser.add_argument("--dx", default=None, type=float, help="Grid cell size on the x axis")
	parser.add_argument("--dy", default=None, type=float, help="Grid cell size on the y axis")
	parser.add_argument("--dz", default=None, type=float, help="Grid cell size on the z axis")
	parser.add_argument("--cuda", default=3, type=int, help="CUDA device available")
	parser.add_argument("--win0", default=None, type=int, help="Seconds before the template phase")
	parser.add_argument("--win1", default=None, type=int, help="Seconds after the template phase")
	parser.add_argument("--win0_save_tplt", default=None, type=int, help="Seconds before the origin time of the template")
	parser.add_argument("--win1_save_tplt", default=None, type=int, help="Seconds after the origin time of the template")
	parser.add_argument("--halfwin_for_local_mad", default=None, type=int, help="Seconds of the half win to calculate the local MAD")
	parser.add_argument("--plot", default=None, type=int, help="Whether plot the detection")
	parser.add_argument("--continue_previous_experiment", default=None, type=int, help="Whether continue previous experiment")
	parser.add_argument("--ctlg_nrows", default=None, type=int, help="For mini test, use a small number of templates to debug")
	parser.add_argument("--max_number_channels_in_one_station", default=None, type=int, help="For one station, how many channels you want to use")
	parser.add_argument("--ctlg_path", default=None, type=str, help="The catalog file you use")
	parser.add_argument("--root", default=None, type=str, help="Path of the continuous waveform")
	parser.add_argument("--fvelo", default=None, type=str, help="Path of the velocity model file")
	parser.add_argument("--tplt_folder", default=None, type=str, help="Parent folder of the output template file")
	parser.add_argument("--tplt_path", default=None, type=str, help="File name of the output template file")

	args = parser.parse_args()
	return args

def initialize(config,conf):
	args = {}
	args['continue_previous_experiment'] = conf.continue_previous_experiment if conf.continue_previous_experiment else config['continue_previous_experiment']
	if args['continue_previous_experiment']==0:
		now = datetime.now()
		experiment = os.path.join((experiment_dir,debug_dir)[conf.debug],'%d%02d%02dT%02d%02d%02d'%(now.year,now.month,now.day,now.hour,now.minute,now.second))
	else:
		previous_experiments = sorted(glob.glob(os.path.join(experiment_dir,'*')))
		if args['continue_previous_experiment']<0:
			try:experiment = previous_experiments[args['continue_previous_experiment']]
			except:exit('The experiment you chose does NOT exist. Please select again.')
		elif args['continue_previous_experiment']==1:
			message = "Please select which experiment below you'd like to continue by enter index:\n"
			lines = ''.join(['%s %d\n'%(x,i) for i,x in enumerate(previous_experiments)])
			idx = input(message+lines)
			try:experiment = previous_experiments[int(idx)]
			except:exit('The experiment you chose does NOT exist. Please select again.')
		else:exit('The number is NOT valid. Please choose another number between %d and 1.'%(-len(previous_experiments)))
	args['experiment'] = experiment
	args['plot'] = conf.plot if conf.plot else config['plot']
	args['eps'] = conf.eps if conf.eps else config['eps']
	args['sampling_rate'] = conf.sampling_rate if conf.sampling_rate else config['sampling_rate']
	args['snr_threshold'] = conf.snr_threshold if conf.snr_threshold else config['snr_threshold']
	args['minimum_cc'] = conf.minimum_cc if conf.minimum_cc else config['minimum_cc']
	args['number_high_cc_channels_threshold'] = conf.number_high_cc_channels_threshold if conf.number_high_cc_channels_threshold else config['number_high_cc_channels_threshold']
	args['number_high_cc_stations_threshold'] = conf.number_high_cc_stations_threshold if conf.number_high_cc_stations_threshold else config['number_high_cc_stations_threshold']
	args['max_number_channels'] = conf.max_number_channels if conf.max_number_channels else config['max_number_channels']
	args['win0'],args['win1'] = (conf.win0 if conf.win0 else config['win0']),(conf.win1 if conf.win1 else config['win1'])
	args['win0_save_tplt'] = conf.win0_save_tplt if conf.win0_save_tplt else config['win0_save_tplt']
	args['win1_save_tplt'] = conf.win1_save_tplt if conf.win1_save_tplt else config['win1_save_tplt']
	args['halfwin_for_local_mad'] = conf.halfwin_for_local_mad if conf.halfwin_for_local_mad else config['halfwin_for_local_mad']
	args['number_stations_threshold_for_template'] = conf.number_stations_threshold_for_template if conf.number_stations_threshold_for_template else config['number_stations_threshold_for_template']
	args['mad_threshold'] = conf.mad_threshold if conf.mad_threshold else config['mad_threshold']
	args['too_close_detections_to_remain'] = conf.too_close_detections_to_remain if conf.too_close_detections_to_remain else config['too_close_detections_to_remain']
	args['device'] = torch.device('cuda:%d'%conf.cuda if torch.cuda.is_available() else 'cpu')
	if torch.cuda.is_available():print('CUDA:%d is used'%conf.cuda)
	else:print('CPU is used')
	args['ctlg_path'] = conf.ctlg_path if conf.ctlg_path else config['ctlg_path']
	args['ctlg_nrows'] = conf.ctlg_nrows if conf.ctlg_nrows else config['ctlg_nrows']
	args['root'] = conf.root if conf.root else config['root']
	args['fvelo'] = conf.fvelo if conf.fvelo else config['fvelo']
	args['tplt_folder'] = conf.tplt_folder if conf.tplt_folder else config['tplt_folder']
	args['tplt_path'] = os.path.join(config['tplt_folder'],config['tplt_path'])
	args['tplt_log'] = os.path.join(config['tplt_folder'],config['tplt_log'])
	if args['plot'] and (not os.path.exists(args['tplt_log'])):os.makedirs(args['tplt_log'])
	args['detection_folder'] = os.path.join(experiment,'catalog')
	args['log'] = os.path.join(experiment,config['log'])
	args['max_number_channels_in_one_station'] = conf.max_number_channels_in_one_station if conf.max_number_channels_in_one_station else config['max_number_channels_in_one_station']
	radius = config['radius']
	if conf.nx:radius['nx'] = conf.nx
	if conf.ny:radius['ny'] = conf.ny
	if conf.nz:radius['nz'] = conf.nz
	if conf.dx:radius['dx'] = conf.dx
	if conf.dy:radius['dy'] = conf.dy
	if conf.dz:radius['dz'] = conf.dz
	args['radius'] = radius
	if not os.path.exists(experiment):os.makedirs(experiment)
	if not os.path.exists(args['detection_folder']):os.makedirs(args['detection_folder'])
	if not os.path.exists(args['log']):os.makedirs(args['log'])
	with open(os.path.join(experiment,'config.yml'),'w') as outfile:yaml.dump(args,outfile,default_flow_style=False)
	return args

def torch_normalize_by_max(x):
	x = x-torch.mean(x,axis=1,keepdims=True)
	return x/torch.max(torch.abs(x),axis=1,keepdims=True)[0]

def plot_detection_overlaped(idx,left,win0,sampling_rate,SHIFT,win1,template,continuous,MASK_ZERO,
		channels,CC,stack,ot,peak_mad,local_mad,width,evloc,dists,fdate,scale=6):
	for axis in ['right','left','top']:mpl.rcParams['axes.spines.%s'%axis] = False
	win = int((win0+win1)*sampling_rate)
	start_continuous = int(idx-left+win0*sampling_rate)
	nchan = len(SHIFT)
	shifts = torch.sort(SHIFT)
	I = torch.argsort(dists).cpu()
	b,e = shifts.values[0]/sampling_rate-win0,shifts.values[-1]/sampling_rate+win1
	b,e = b.cpu().numpy(),e.cpu().numpy()
	B,E = int(b*sampling_rate),int(e*sampling_rate)
	offset = int(shifts.values[0]-(B+win0*sampling_rate))
	half = int((E-B)/2)
	continuous = continuous[:,start_continuous+B:start_continuous+E]
	nC,nT = torch_normalize_by_max(continuous),torch_normalize_by_max(template)
	fig,ax = plt.subplots()
	for k,i in enumerate(I):ax.plot(np.arange(E-B)/sampling_rate+b,nC[i].cpu()+k*2,color='#bab2b0',lw=1)
	for k,i in enumerate(I):ax.plot((np.arange(win)+int(SHIFT[i])+offset)/sampling_rate-win0,nT[i].cpu()+k*2,color='k',lw=.5)
	for k,i in enumerate(I):ax.text(e,k*2,'%.2f'%CC[i,idx-left+SHIFT[i]],color=('r' if MASK_ZERO[i,idx-left+SHIFT[i]]==False else 'b'),va='center')
	for k,i in enumerate(I):ax.text(b,k*2,channels[i][0],color='k',ha='right',va='center')
	cc = stack[idx-half:idx+half]
	cc_range = torch.max(cc)-torch.min(cc)
	cc_modified = (cc-torch.min(cc))/cc_range*scale
	cc_peak = (cc[half]-torch.min(cc))/cc_range*scale
	baseline = -torch.min(cc)/cc_range
	ax.scatter(x=[b+half/sampling_rate],y=[cc_peak.cpu()+2*(nchan-1)+1],color='r')
	ax.plot(np.arange(2*half)/sampling_rate+b,cc_modified.cpu()+(nchan-1)*2+1,color='gray',lw=.5)
	ax.text(b,(nchan-1)*2+1+scale*baseline.cpu(),'Stack CC',color='k',ha='right',va='center')
	ax.text(e,(nchan-1)*2+1+scale*baseline.cpu(),'%.2f'%stack[idx],color='r',va='center')
	ax.set_yticks([])
	ax.set_xlabel('Seconds since %s (%f)'%(str(ot),calc_relative_time(ot)))
	ax.set_title('%.1f MAD;%.1f localMAD;%.1f peakedness with Template %d (%s)'%(peak_mad,local_mad,width,evloc['evid'],str(evloc['ot'])))
	ax.set_xlim(b,e)
	ax.set_ylim(-1,(nchan-1)*2+1+scale+.1)
	fig.set_figheight(15);fig.set_figwidth(10)
	plt.tight_layout()
	plt.savefig(os.path.join(fdate,'%d_%s.pdf'%(evloc['evid'],str(ot))))
	plt.close()

def reweight_by_common_station(l):
	C = []
	for x in l:C.append(1/l.count(x))
	return np.array(C)

def calc_MAD(x):return (x-torch.median(x))/torch.median(torch.abs(x-torch.median(x)))
def calc_lMAD(x,halfwin):
	n = len(x)
	l = []
	for i in torch.arange(n):
		b = max(i-halfwin,0)
		e = min(i+halfwin+1,n)
		t = x[b:e]
		l.append((x[i]-torch.median(t))/torch.median(torch.abs(t-torch.median(t))))
	return l

def stack_all_channels_of_CC_and_calculate_MAD(x,mask,shifts,snr,reweight,cc_threshold,device):
	reweight = torch.unsqueeze(reweight,1)
	Lefts,rights = torch.min(shifts,1)[0],torch.max(shifts,1)[0]
	n_node,_ = shifts.shape
	_,nt = x.shape
	ArrayStack = torch.zeros((n_node,nt)).to(device)
	MASK = torch.ones_like(ArrayStack).to(device)
	MAD = torch.zeros_like(ArrayStack).to(device)
	N_high_cc_channels = torch.zeros((2,n_node,nt)).to(device)
	N_high_cc_stations = torch.zeros_like(N_high_cc_channels).to(device)
	for node,shift in enumerate(shifts):
		weight = ((snr/torch.sum(snr))+1/(shift*torch.sum(1/shift)))/2
		mask_length = rights[node]-Lefts[node]
		c = torch.zeros_like(x).to(device)
		m = torch.zeros_like(mask).to(device)
		w = torch.ones_like(x).to(device)*torch.unsqueeze(weight,1)
		for i,s in enumerate(shift):
			c[i] = torch.roll(x[i],-int(s-Lefts[node]))
			m[i] = torch.roll(mask[i],-int(s-Lefts[node]))
		positive_flag = c>=cc_threshold
		negative_flag = c<=-cc_threshold
		N_high_cc_channels[0][node] = torch.sum(positive_flag,0)
		N_high_cc_channels[1][node] = torch.sum(negative_flag,0)
		N_high_cc_stations[0][node] = torch.sum(positive_flag*reweight,0)
		N_high_cc_stations[1][node] = torch.sum(negative_flag*reweight,0)
		c*=w;w[m]=0
		stack = torch.sum(c,0)/torch.sum(w,0)
		ArrayStack[node] = stack
		MASK[node,-mask_length:] = 0
		MAD[node,:-mask_length] = calc_MAD(stack[:-mask_length])
	return ArrayStack,MAD,MASK,Lefts,N_high_cc_channels,N_high_cc_stations

def calc_peak_properties(cc,tid,flag,args,wlen=.2):
	c = cc*flag
	halfwin = int(args['sampling_rate']*args['halfwin_for_local_mad'])
	subcc = c[tid-halfwin:tid+halfwin+1]
	lmad = calc_MAD(subcc)[halfwin]
	results = peak_widths(subcc.cpu(),[halfwin,],rel_height=1,wlen=int(wlen*args['sampling_rate']))
	width,height = results[0][0],c[tid]-results[1][0]
	return lmad*flag,width,height

def deduplicate_in_time(t,c,distance,device='cpu'):
	device = torch.device(device)
	t,c = t.to(device),c.to(device)
	ordered_c,order = torch.sort(c,descending=True)
	mi,ma = torch.min(t),torch.max(t)
	m = -torch.ones(ma+1-mi).to(device)*999
	m[t-mi] = c
	mask = torch.ones(ma+1-mi).to(device)
	I = []
	for i,v in zip(order,ordered_c):
		time = t[i]-mi
		b,e = max(0,time-distance),min(time+distance,ma)+1
		if mask[time] or torch.max(m[b:e])==v:
			mask[b:e] = 0
			I.append(i)
	return torch.LongTensor(I).to(device)

def count_high_cc_channels_and_high_cc_stations(gids,tids,CC,mads,lefts,shifts,reweight,cc_threshold,channel_threhsold,station_threshold,device):
	I,N0,N1 = [],[],[]
	for i,(gid,tid) in enumerate(zip(gids,tids)):
		mad = mads[gid,tid];flag = (-1,1)[mad>0]
		left = lefts[gid];shift = shifts[gid]
		arraycc = CC[torch.arange(len(shift)),tid-left+shift]
		over_stack = arraycc*flag>=cc_threshold
		n0 = torch.sum(over_stack)
		n1 = torch.sum(over_stack*reweight)
		if n1<=station_threshold or n0<=channel_threhsold:continue
		I.append(i);N0.append(n0);N1.append(n1)
	I = torch.LongTensor(I).to(device)
	return I,N0,N1

def detect(stacks,mads,lefts,N_high_cc_channels,N_high_cc_stations,
		mad_threshold,number_high_cc_channels_threshold,
		number_high_cc_stations_threshold,distance,device):
	z = torch.zeros_like(mads)
	flag = torch.zeros_like(mads)
	peak_flag = (mads>=mad_threshold)*(N_high_cc_channels[0]>number_high_cc_channels_threshold)*(N_high_cc_stations[0]>number_high_cc_stations_threshold)
	peak_flag[:,1:] = peak_flag[:,1:]*(torch.diff(stacks,dim=1)>=0)
	peak_flag[:,:-1] = peak_flag[:,:-1]*(torch.diff(stacks,dim=1)<=0)
	trough_flag = (mads<=-mad_threshold)*(N_high_cc_channels[1]>number_high_cc_channels_threshold)*(N_high_cc_stations[1]>number_high_cc_stations_threshold)
	trough_flag[:,1:] = trough_flag[:,1:]*(torch.diff(stacks,dim=1)<=0)
	trough_flag[:,:-1] = trough_flag[:,:-1]*(torch.diff(stacks,dim=1)>=0)
	detection_flag = peak_flag+trough_flag
	z[detection_flag] = stacks[detection_flag]
	flag[detection_flag] = 1
	for i,left in enumerate(lefts):
		z[i] = torch.roll(z[i],-int(left))
		flag[i] = torch.roll(flag[i],-int(left))
	T = torch.where(torch.max(flag,0)[0]>0)[0]
	G = torch.argmax(torch.abs(z[:,T]),dim=0)
	if len(T):
		iT = deduplicate_in_time(T,torch.abs(mads[G,T+lefts[G]]),distance).to(device)
		return G[iT],T[iT]+lefts[G[iT]]
	else:
		return [],[]

def create_file(fdate_detection,fname,header):
	if not os.path.exists(fdate_detection):os.makedirs(fdate_detection)
	if not os.path.exists(fname):
		f=open(fname,'w');f.write(header)
	else:f = open(fname,'a')
	return f

def write_catalog(gids,tids,N_high_cc_channels,N_high_cc_stations,folder,lefts,evloc,grids,MADs,stack,shifts,template,continuous,MASK_ZERO,channels,CC,dists,args):
	date = os.path.basename(folder)
	candidate_t = UTCDateTime(date)
	fdate_detection = os.path.join(args['detection_folder'],date)
	fname = os.path.join(fdate_detection,'raw_ctlg.txt')
	header = 'ot relative_ot template_id lon lat dep Times_of_MAD Times_of_local_MAD CC peakedness_cc n_high_cc_channels n_high_cc_stations\n'
	f = create_file(fdate_detection,fname,header)
	for tid,gid in zip(tids,gids):
		mad = MADs[gid,tid]
		factor = (-1,1)[mad>0]
		flag = (1,0)[mad>0]
		n_high_cc_channels = N_high_cc_channels[flag,gid,tid]
		n_high_cc_stations = N_high_cc_stations[flag,gid,tid]
		left = lefts[gid]
		local_MAD,prominence,width = calc_peak_properties(stack[gid],tid,factor,args)
		detect_ot = candidate_t+float(tid-left)/args['sampling_rate']+args['win0']
		relative_ot = calc_relative_time(detect_ot)
		evlo = evloc['evlo']+grids[gid][0]
		evla = evloc['evla']+grids[gid][1]
		evdp = evloc['evdp']+grids[gid][2]
		stack_CC = stack[gid,tid]
		shift = shifts[gid]
		line0 = '%s %f %d %f %f %f %f %f %f %f %d %f'%(detect_ot,relative_ot,evloc['evid'],evlo,evla,evdp,mad,
			local_MAD,stack_CC,prominence/width,n_high_cc_channels,n_high_cc_stations)
		f.write(line0+'\n')
		print(line0+'\n'+'='*10)
		if args['plot']:
			fdate_log = os.path.join(args['log'],date)
			if not os.path.exists(fdate_log):os.makedirs(fdate_log)
			plot_detection_overlaped(tid,left,args['win0'],args['sampling_rate'],shift,
				args['win1'],template,continuous,MASK_ZERO,channels,CC,stack[gid],
				detect_ot,mad,local_MAD,prominence/width,evloc,dists,fdate_log,scale=6)
	f.close()