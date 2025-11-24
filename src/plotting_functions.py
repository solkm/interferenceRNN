import numpy as np

def sliding_window_avg(measure, n_avg, sem=None):
    """
    Returns a sliding window average of a measure over n_avg epochs.
    measure_sw[0] is the mean, measure_sw[1] is the SEM.
        The SEM is computed from standard deviation over the window OR from the 
        individual SEMs if provided.
    """
    n_steps = measure.shape[0] - n_avg
    measure_sw = np.zeros((2, n_steps))
    for i in range(n_steps):
        measure_sw[0, i] = np.mean(measure[i:i+n_avg])
        if sem is not None:
            measure_sw[1, i] = np.sqrt(np.mean(sem[i:i+n_avg]**2))
        else:
            measure_sw[1, i] = np.std(measure[i:i+n_avg])/np.sqrt(n_avg)

    return measure_sw

def set_violin_color(vplots, color, alpha=0.1):
    for key in vplots.keys():
        if key == 'bodies':
            for v in vplots[key]:
                v.set_edgecolor(color)
                v.set_facecolor(color)
                v.set_alpha(alpha)
        else:
            vplots[key].set_color(color)

def angles_0to90(a):
    angles = a.copy()
    angles[angles > 90] = 180 - angles[angles > 90]
    return angles