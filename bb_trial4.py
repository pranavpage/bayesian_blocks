
import numpy as np
import glob
import os
import time
from matplotlib.backends.backend_pdf import PdfPages
start_time = time.time()
import numpy.ma as ma
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})
from astropy.io import fits
from astropy.table import Table
from astropy.stats import sigma_clipped_stats as scs
from astropy.stats import bayesian_blocks as bb
from astropy.visualization import hist
np.set_printoptions(threshold=np.inf)
from make_lightcurves import getlc_clean

def piece(bin_edges, bin_rates, t_bin):
    time = np.arange(bin_edges[0], bin_edges[-1], t_bin)
    #binning
    bin_widths = (bin_edges[1:] - bin_edges[:1])
    condlist=np.zeros((bin_rates.size, time.size), dtype=float)
    for i in range(bin_rates.size):
    #making condlist
        condlist[i] = (time>=bin_edges[i])&(time<bin_edges[i+1])
    rates = np.piecewise(x=time, condlist=condlist, funclist=bin_rates)
    return time, rates

def bb_segments(grb_name, t_bin, quad ,  p0=0.05):
    os.chdir('/home/pranav/Project_Daksha/Bayesian Blocks/grb_temp/'+ grb_name)
    #changed directory to GRB folder

    data = fits.getdata('lc_{}_4_Q{}.lc'.format(t_bin, quad))
    time = data['TIME']
    lc_detrend, mask_lc, _ , _ , _ =getlc_clean(tbin=t_bin, quad = 'lc_{}_4_Q{}.lc'.format(t_bin, quad), threshold = 0, filtertype='savgol', filterorder = 2, filterwidth = 100, tclip=5 )
    _,med,_ = scs(data['RATE'])

    edges_bb = bb(time, np.around(lc_detrend + med), sigma = np.sqrt(abs(lc_detrend+med)+1), p0=p0)
    edges_bb[0] = edges_bb[0] - t_bin/2
    edges_bb[-1] = edges_bb[-1] + t_bin/2
        #check if this can be done in a better way, also for 0.1s and 10s binnings

    bin_widths = edges_bb[1:] - edges_bb[:-1]
    bin_counts = np.histogram(time, bins=edges_bb, weights = lc_detrend)[0]
    bin_rates = bin_counts/bin_widths

    p_time, p_rates= piece(edges_bb, bin_rates, t_bin)
    return p_time, p_rates

def n_sigma_bb(grb_name, t_bin, N=3, p0=0.05):
    points=[None]*4
    masks = [None]*4
    for quad in range(4):
        time, _ = bb_segments(grb_name, t_bin, quad, p0)
        if quad==0 :
            start = time[0]
            stop = time[-1]
        else :
            start = min(time[0], start)
            stop = max(time[-1], stop)

    for quad in range(4):
        points[quad]=[]
        time, rates = bb_segments(grb_name, t_bin, quad, p0)
        rates = np.pad(rates, (int(time[0]-start),int(stop-time[-1])), 'median')
        time = np.pad(time, (int(time[0]-start),int(stop-time[-1])), 'linear_ramp', end_values=(start, stop))
        masks[quad]=np.zeros_like(time)
        mean, median, sigma = scs(rates)
        for k in range(time.size):
            if(rates[k]>= mean + N*sigma):
                points[quad].append(time[k])
                masks[quad][k]=1

    masks = np.array(masks)
    summed = np.sum(masks, axis=0)
    peaks = time[np.where(summed>=2)]
    return peaks

def plot_lc(grb_name, t_bin, trigger=None,  N=3, p0=0.05):
    quadnames = ['A', 'B', 'C','D']
    peaks = n_sigma_bb(grb_name, t_bin, N, p0)
    for quad in range(4):
        time, rates = bb_segments(grb_name, t_bin, quad, p0)
        lc_detrend, mask_lc, _ , _ , _ =getlc_clean(tbin=t_bin, quad = 'lc_{}_4_Q{}.lc'.format(t_bin, quad), threshold = 0, filtertype='savgol', filterorder = 2, filterwidth = 100, tclip=5 )
        if quad == 0:
            ax = plt.subplot(2,2,quad+1)
        else:
            plt.subplot(2, 2, quad+1, sharex=ax)
        plt.plot(time, rates, color='r', label="Bayesian Blocks segmented lc")
        plt.xlabel('Time(s)')
        plt.ylabel('Rate(counts/s)')
        lcmin= lc_detrend.min()
        lcmax=lc_detrend.max()
        plt.plot(time, lc_detrend, alpha=0.5, label ="Detrended lc")
        plt.vlines(trigger, -50, 0, color='g', label="Trigger(from the catalog)" )
        plt.title("Quadrant {}".format(quadnames[quad]))
        plt.vlines(peaks, ymin=lcmin,ymax=lcmax,linestyle='dashed',linewidth=0.5, alpha=0.5, color='b', label="Peaks")
#        plt.legend(loc=1)
        plt.xlim(trigger-10, trigger+10)
        plt.ylim(-100, 200)
    plt.suptitle("Bayesian Blocks on {} with N={}, p0={}".format(grb_name, N, p0))
    plt.subplots_adjust(hspace=1)
#    plt.savefig("{}_full_{}.png".format(grb_name,t_bin), dpi=200)
    plt.savefig("{}_trigger_{}.png".format(grb_name,t_bin), dpi=400)
#    plt.show()


os.chdir('/home/pranav/Project_Daksha/Bayesian Blocks')
grblist = np.genfromtxt(fname='grblist.txt', usecols=(0,1,2), dtype=None)
os.chdir('/home/pranav/Project_Daksha/Bayesian Blocks/grb_temp')
#for quad in range(4):
#    time, rates = bb_segments(grb_name = "GRB200207A", t_bin=1.0, quad=quad)
#    print(time, rates)
#plot_lc(grb_name = "GRB200306A", t_bin=1.0)
grb_list = glob.glob("*GRB200306A*")
for grb_name in grb_list:
    trigger = float(grblist[grblist[:,1].tolist().index(grb_name),2])
    plot_lc(grb_name=grb_name, t_bin=1.0, trigger=trigger)
    print("{} done".format(grb_name))
    print("--- Left---")
    print(grb_list[grb_list.index(grb_name)+1:])
    print("------")
print("--- %s seconds ---" % (time.time() - start_time))
