
import matplotlib.pyplot as plt


def arrange_subplots(pltc):
    ''' arrange_subplots(pltc) -> fig, axes
    Arranges a given number of plots to well formated subplots.
    
    Args:
        pltc: The number of plots.
    
    Returns:
        fig: The figure.
        axes: A list of axes of each subplot.
    '''
    cols = int(np.floor(np.sqrt(pltc)))
    rows = int(np.ceil(pltc/cols))
    fig, axes = plt.subplots(cols,rows)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes]) #fix format so it can be used consistently.
    
    return fig, axes.flatten()