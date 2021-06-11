import numpy as np
import matplotlib.pyplot as pl

def plot_fun_heatmap(f, xxx, yyy, **kwargs):
    fff = np.array([[ f([x,y]) for y in yyy ] for x in xxx])
    cp = pl.contourf( xxx, yyy, fff.T, cmap=pl.cm.rainbow, **kwargs)
    return cp

def plot_fun(f, start, end, **kwargs):
    
    if len(start) == 1:
        start = start if start is not None else 0
        end = end if end is not None else 1
        start, end = [x[0] if np.array(x).ndim > 0 else x for x in [start, end] ]            

        xxx = np.linspace(start, end, 100)
        yyy = [ f(np.array([x])) for x in xxx]
        pl.plot(xxx, yyy, **kwargs)

    elif len(start) == 2:
        start = start if start is not None else [0,0]
        end = end if end is not None else [1,1]

        return plot_fun_heatmap(
            f = f,
            xxx = np.linspace(start[0], end[0], 30),
            yyy = np.linspace(start[1], end[1], 30),
        )
