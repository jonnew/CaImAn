import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.interpolate import UnivariateSpline
from typing import Dict, Tuple, List
from IPython.display import display, HTML
#from ipywidgets import interact, interactive, fixed, interact_manual, HBox, VBox, IntSlider, Play, jslink, Output
#import ipywidgets as widgets
from IPython.display import clear_output
import bqplot
from bqplot import (
    LogScale, LinearScale, OrdinalColorScale, ColorAxis, OrdinalScale,
    Axis, Scatter, Lines, CATEGORY10, Label, Figure, Tooltip, Toolbar, Bars
)
import traitlets
import qgrid
#%matplotlib inline
from event_widgets import *
from caiman_interface import context

def setup_context(context):
    if len(context.cnmf_results) > 0:
        signal_data = context.cnmf_results[1]
    else:
        signal_data = None
    return signal_data

def est_baseline(sig_array: np.ndarray):
    min_r = 0.5
    s_copy = sig_array.copy()
    x = np.arange(len(s_copy))
    y = s_copy
    #plt.scatter(x,y)
    slope, intercept = 0.0,0.0
    #for i in range(N):
    i: int = 1
    while(True):
        #linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        #print(r_value, p_value, std_err)
        #print(slope,intercept)
        #remove points in signal above line
        t = [ (si,s_copy[si]) for si in range(len(s_copy)) if s_copy[si] < (slope * si + intercept)]
        if len(t) < 50:
            break
        #print(len(t))
        x = np.array([xs[0] for xs in t])
        y = np.array([ys[1] for ys in t])
        #plt.scatter(x,y)
        if abs(r_value) > min_r:
            break
        i += 1
        if i > 100:
            break
    return slope, intercept

def detect_events(signal, signal_log, thresh=1.025, min_amplitude=0.015, fr=0.03333):
    # list of events, each event is itself a list of the form
    # start (index), peak value, peak index, end (index); need to convert indices to time later
    events: List = []
    #main loop
    in_event = False
    cur_event: List = [None, None, None, None] #start, peak value, peak index, end (list indices)
    #baseline = np.percentile(signal, 1)
    #Estimate baseline
    #signal_log = np.power(10,signal) #convert to log scale
    #slope,intercept = est_baseline(signal_log)
    #baseline = 1.01 * intercept #initialize baseline (increase by small amount because it underestimates baseline)
    #thresh = min_thresh * baseline
    for i in range(len(signal)): #loop through each element in signal array
        #b = signal[i]
        bp = signal_log[i]
        #track events
        if in_event:
            if bp <= (thresh): #event has ended
                #print("Event end: #%s" % (len(events),))
                cur_peak_val = np.max(signal[cur_event[0]:i])
                #NOTE! cur_peak_index is index relative to subset of full signal array, must convert to absolute index
                cur_peak_index = signal[cur_event[0]:i].tolist().index(cur_peak_val)
                #print(cur_event[0])
                cur_peak_index += cur_event[0]
                cur_event[1:] = cur_peak_val, cur_peak_index, i   #peak value, peak index, end index
                #if crosses min threshold, this is not obvious noise, save this event, otherwise ignore
                adj_peak_amplitude = np.power(10,cur_peak_val) - thresh
                if adj_peak_amplitude >= min_amplitude:
                    if None in cur_event:
                        print("Error! `cur_event` not completed before adding to `events`")
                    events.append(cur_event)
                #reset cur_event
                cur_event = [None, None, None, None]
                in_event = False
        else: #not in event, but let's check if we are starting a new event
            if bp > thresh: #if current signal crosses baseline level
                #print("Event started: #%s" % (len(events),))
                in_event = True
                cur_event[0] = i #create new event with start filled, peak and end not yet known
    #print("Done! Found %s events." %(len(events)))
    return events

#event start, event peak, event end, amplitude, duration, rise-time, decay-time, area
def analyze_event(signal, event, fr):
    start_time = event[0] * fr + fr
    end_time = event[3] * fr + fr
    peak_time = event[2] * fr + fr
    amplitude = event[1]
    duration = end_time - start_time
    rise_time = peak_time - start_time
    decay_time = end_time - peak_time
    area = np.trapz(signal[event[0]:event[3]],dx=0.33)
    #calculuate half-width
    spike = signal[event[0]:event[3]]
    x = np.arange(len(spike))
    try:
        spline = UnivariateSpline(x,spike-np.max(spike)/2, s=0)
        r1, r2 = spline.roots() # find the roots
    except Exception:
        r1,r2 = 0,0
    half_width = (r2-r1)*fr + fr
    return start_time, end_time, peak_time, amplitude, duration, half_width, rise_time, decay_time, area

def compute_stats(signal, events, fr):
    results = np.zeros((len(events),9))
    columns=['Start', 'End', 'Peak Time','Amplitude', 'Duration', 'Half-Width', 'rise-time', 'decay-time', 'Area']
    df = pd.DataFrame(data=results, columns=columns)
    #event: [start index, peak value, peak index, end index]
    c = 0
    for event in events:
        df.iloc[c,:] = analyze_event(signal, event, fr)
        c += 1
    return df


def frames_to_time(a, fr=0.0333):
    return a * fr + fr

def show_start_menu():
    start_event_btn.on_click(show_plot)
    settings_box = HBox([min_thresh_widget, min_ampl_widget, fr_widget])
    display(VBox([settings_box,start_event_btn]))

def analyze_roi(signal):
    min_thresh = float(min_thresh_widget.value)
    min_amplitude = float(min_ampl_widget.value)
    fr = float(fr_widget.value) #duration of each frame in seconds
    signal_log = np.power(10, signal)
    ####
    slope,intercept = est_baseline(signal_log)
    baseline = 1.01 * intercept
    thresh = min_thresh * baseline
    events = detect_events(signal, signal_log, thresh, min_amplitude, fr)
    #time.sleep(0.01)
    event_results = compute_stats(signal, events, fr)
    #plt.plot(signal_log) #plot raw signal
    thresh_line_y = np.repeat(np.log10(thresh),len(signal))#; need to convert back to orig scale
    #plt.plot(thresh_line_y) #threshold
    slope,intercept = est_baseline(signal)
    #x_range = np.arange(len(signal))
    x_range = np.arange(0,frames_to_time(len(signal)), step=fr)
    y_base = np.multiply(slope, x_range) + intercept #for baseline line
    #plt.plot(x_range,y_thresh)
    events_x = [frames_to_time(x[2]) for x in events]
    events_y = [y[1] for y in events] #[np.power(10,y[1]) for y in events]
    return x_range, y_base, events_x, events_y, thresh_line_y, event_results

def analyze_all(signals):
    #list of tuples, each tuple is an ROI
    results = []
    #print(len(signals))
    for i in range(len(signals)): #for each ROI
        #print(i)
        results.append(analyze_roi(signals[i]))

    return results


#convert list of tuples from analyze_all to a dataframe
def convToDF(results):
    #rows = rois * #events
    #df_init = np.zeros((len(results),9))
    columns=['ROI', 'Start', 'End', 'Peak Time','Amplitude', 'Duration', 'Half-Width', 'rise-time', 'decay-time', 'Area']
    df = pd.DataFrame(columns=columns)
    #event: [start index, peak value, peak index, end index]
    c = 0
    for result in results:
        _, _, _, _, _, event_results = result
        #print("Result {}".format(event_results.shape))
        #df.iloc[c,:] = event_results
        event_results['ROI'] = c + 1
        df = pd.concat([df, event_results], ignore_index=True)
        c += 1
    #print(len(df))
    df = df[columns]
    return df

def show_plot(_):
    clear_output()
    show_start_menu()
    signal_data = setup_context(context)
    roi_slider_widget.max = signal_data.shape[0]
    #print(signal_data.shape)
    ###
    results = analyze_all(signal_data)
    df_results = convToDF(results)
    #print(len(df_results))
    ###
    #plt.scatter(x, y, c=['red'], alpha=0.9) #x is indices, y is signal value'
    # === === ===
    cur_roi = (roi_slider_widget.value - 1)
    signal = signal_data[cur_roi]
    #x_range, y_base, events_x, events_y, thresh_line_y, event_results = analyze_roi(signal)
    x_range, y_base, events_x, events_y, thresh_line_y, event_results = results[cur_roi]

    x_sc = LinearScale()
    y_sc = LinearScale()
    scat_tt = Tooltip(fields=['index'], formats=['i'], labels=['Event#'])
    sig_line = Lines(scales={'x': x_sc, 'y': y_sc},
                 stroke_width=3, colors=['blue'], display_legend=True, labels=['Ca Signal'])

    thresh_line = Lines(x=x_range, y=thresh_line_y, scales={'x': x_sc, 'y': y_sc},
                 stroke_width=3, colors=['orange'], display_legend=True, labels=['Threshold'])

    base_line = Lines(x=x_range, y=y_base, scales={'x': x_sc, 'y': y_sc},
                 stroke_width=3, colors=['green'], display_legend=True, labels=['Baseline'])

    events_scat = Scatter(x=events_x, y=events_y, scales={'x': x_sc, 'y': y_sc}, tooltip=scat_tt,
                 stroke_width=3, colors=['red'], display_legend=True, labels=['Detected Events'])

    ax_x = Axis(scale=x_sc, grid_lines='solid', label='Time (seconds)')
    ax_y = Axis(scale=y_sc, orientation='vertical', tick_format='0.2f',
                grid_lines='solid', label='Amplitude')

    fig = Figure(marks=[sig_line, base_line, thresh_line, events_scat], axes=[ax_x, ax_y], title='Event Detection & Analysis',
           legend_location='top-left')
    tb0 = Toolbar(figure=fig)

    out = Output()
    event_results_widget = qgrid.QgridWidget(df=df_results, show_toolbar=True)
    def update_plot(change):
        cur_roi2 = (roi_slider_widget.value - 1)
        new_signal = signal_data[cur_roi2]
        #x_range, y_base, events_x, events_y, thresh_line_y, event_results = analyze_roi(new_signal)
        x_range, y_base, events_x, events_y, thresh_line_y, event_results = results[cur_roi2]
        sig_line.y = new_signal
        sig_line.x = x_range
        thresh_line.x = x_range
        thresh_line.y = thresh_line_y
        base_line.x = x_range
        base_line.y = y_base
        events_scat.x = events_x
        events_scat.y = events_y
        #event_results_widget.df = event_results
        indx_range = df_results.loc[df_results['ROI'] == (cur_roi2+1)].index.values.tolist()
        min_ind = indx_range[0]
        max_ind = indx_range[-1]
        event_results_widget._handle_qgrid_msg_helper({
        'field': "index",
        'filter_info': {
            'field': "index",
            'max': max_ind,
            'min': min_ind,
            'type': "slider"
        },
        'type': "filter_changed"
    })
    def dl_events_click(_):
        event_results_widget.get_changed_df().to_csv(path_or_buf = context.working_dir + 'events_data.csv')
        print("Data saved to current working directory as: events_data.csv")
    dl_events_data_btn.on_click(dl_events_click)

    update_plot({'new':1})
    roi_slider_widget.observe(update_plot, names='value')
    fig_widget = VBox([HBox([VBox([roi_slider_widget,tb0,fig])])])
    display(fig_widget,event_results_widget,dl_events_data_btn)
    #display(df_results)
#show_start_menu()
