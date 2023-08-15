import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)


def plot_single_w(sol, time, logscale=False, which_nodes=None, cmap="viridis", title=None, save=False, plotname=None):
    """"Display a single edge count matrix.

        sol: Solution object with the results of the simulation of interest.
        time: Float, point in time at which the edge count matrix should be plotted.
        logscale: Bool indicating whether a logscale (True) or a linear scale (False) should be used for the color bar.
        which_nodes: Array indicating from which to which node W should be displayed. Use this if you want to plot a
                     certain section of W. Set to None (default) if the whole matrix should be displayed.
        cmap: String: colormap to use for showing the edge counts.
        title:  String: title for the plot.
        save: Bool indicating whether the plot should be saved.
        plotname: String, relative path and filename for the plot if save==True.
                  Do not forget the file extension, e.g. '.png'.
        """
    w = sol.w(t=time)
    vmax = np.max(w)
    vmin = np.min(w)

    # Define the normalization method to map to the colors.
    if logscale:
        norm = colors.SymLogNorm(vmin=vmin, vmax=vmax, linthresh=0.001)
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)  # linear

    fig, ax = plt.subplots()
    mat = ax.matshow(w, cmap=cmap, norm=norm)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.set_ylabel("in")
    ax.xaxis.set_label_position('top')
    ax.set_xlabel("out")

    if which_nodes is not None:
        ax.set_xlim([which_nodes[0] - 0.5, which_nodes[1] + 0.5])
        ax.set_ylim([which_nodes[1] + 0.5, which_nodes[0] - 0.5])

    if title is not None:
        ax.set_title(title, y=-0.1)

    # Find evenly spaced ticks for a logscale.
    if logscale:
        ticks = np.around(np.geomspace(np.ceil(vmin), np.floor(vmax), num=5), decimals=3)
        cbar = fig.colorbar(mat, ticks=ticks, ax=ax)
        cbar.set_ticklabels(str(t) for t in ticks)
        cbar.set_label("edge count (logscale)")

    else:
        cbar = fig.colorbar(mat, ax=ax)
        cbar.set_label("edge count")

    if save:
        if plotname is None:
            raise Exception("A name for the plot file has to be provided if it should be saved.")
        plt.savefig(plotname, dpi=400)

    plt.show()


def plot_w(sols, times, nrows, ncols, figsize, logscale, which_nodes=None, cmap="viridis", subtitles=None, title=None,
           save=False, plotname=None):
    """Display the edge count matrix at different points in time.

        sols: List of solution object(s) of MCMC simulations. Has to be a list even if only one solution is given!
        times: Array with points in time at which W(t) should be plotted. If sol is a list of solutions, the same times
               will be used for every solution.
        nrows: Number of rows of subplots.
        ncols: Number of columns of subplots.
        figsize: List indicating the size of the figure ([x, y]).
        logscale: Bool indicating whether a logscale (True) or a linear scale (False) should be used for the color bar.
        which_nodes: Array indicating from which to which nodes W should be displayed. Use this if you want to plot a
                     certain section of W. Should have the same length as times and be in the same order.
                     Set to None (default) if the whole matrix should be displayed at all times.
                     Example: which_nodes = np.array([[0, 3], [0, 10]]) if you want to plot W at two different points in
                              time (or at the same time twice) t1 and t2 and at t1 you only want to see elements w_ij
                              from i,j = 0 to 3 but at t2 i,j = 0 to 10.
        cmap: String: colormap to use for showing the edge counts.
        subtitles: List of strings with titles for the individual subplots.
        title:  String: title for the whole plot.
        save: Bool indicating whether the plot should be saved.
        plotname: String, relative path and filename for the plot if save==True.
                  Do not forget the file extension, e.g. '.png'.
        """
    # Extract the edge count matrices from the solution object(s).
    w_t = []

    for sol in sols:
        for t_i in times:
            w_t.append(sol.w(t=t_i))

    # Compute the range of the edge counts.
    vmin = min(np.min(w) for w in w_t)
    vmax = max(np.max(w) for w in w_t)

    # Define the normalization method to map to the colors.
    if logscale:
        norm = colors.SymLogNorm(vmin=vmin, vmax=vmax, linthresh=0.001)
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)  # linear

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True)

    for idx, w in enumerate(w_t):
        if nrows == 1:
            row = 0
            col = idx
            mat = axs[col].matshow(w, cmap=cmap, norm=norm)
        else:
            row, col = divmod(idx, ncols)
            mat = axs[row][col].matshow(w, cmap=cmap, norm=norm)

        # Set axis range.
        if which_nodes is not None:
            axs[row][col].set_xlim([which_nodes[idx][0] - 0.5, which_nodes[idx][1] + 0.5])
            axs[row][col].set_ylim([which_nodes[idx][1] + 0.5, which_nodes[idx][0] - 0.5])

        # Set ticks and tick labels on the y axis only if the subplot is on the left and on the x axis only if it is on
        # top.
        if row == 0:
            if nrows == 1:
                axs[col].tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
            else:
                axs[row][col].tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        else:
            axs[row][col].tick_params(top=True, bottom=False, labeltop=False, labelbottom=False)

        # Set x label only if the subplot is on top and y label only if the subplot is on the left.
        if row == 0:
            if nrows == 1:
                axs[col].xaxis.set_label_position('top')
                axs[col].set_xlabel(" ")
            else:
                axs[row][col].xaxis.set_label_position('top')
                axs[row][col].set_xlabel(" ")
        if col == 0:
            if nrows == 1:
                axs[col].set_ylabel("nodes")
            else:
                axs[row][col].set_ylabel("nodes")

        # Set the title for the subplot.
        if subtitles is not None:
            if nrows == 1:
                axs[col].set_title(subtitles[idx], y=-0.2)
            else:
                axs[row][col].set_title(subtitles[idx], y=-0.2)

    if title is not None:
        fig.suptitle(title)

    # Show the colorbar.
    if logscale:
        ticks = np.around(np.geomspace(np.ceil(vmin), np.floor(vmax), num=5), decimals=3)
        cbar = fig.colorbar(mat, ticks=ticks, ax=axs)
        cbar.ax.set_yticklabels(str(t) for t in ticks)
        cbar.set_label("edge count (logscale)")
    else:
        cbar = fig.colorbar(mat, ax=axs)
        cbar.set_label("edge count")

    if save:
        if plotname is None:
            raise Exception("A name for the plot file has to be provided if it should be saved.")
        plt.savefig(plotname, dpi=400)

    plt.show()


def plot_single_microconf(sol, time, node_grid=True, edge_grid=False, which_nodes=None, title=None, cmap='binary',
                          save=False, plotname=None):
    """Visualize the micro-configuration of a network at a certain time.
    Create a plot where existing edges between nodes are marked by black squares. No color means that the respective
    in- and out-degrees of two nodes are not connected via an edge.
    The cell in line i and row j of the grid represents the potential edge from out-degree j to in-degree i.

    sol: Solution object with the results of the simulation of interest.
    time: Float, point in time at which the micro-configuration should be plotted.
    node_grid: Bool indicating whether a grid should be drawn that groups the in- and out-degrees according to which
               nodes they belong to ("macro-configuration grid"). Default=True.
    edge_grid: Bool indicating whether a grid should be drawn that separates all pairs of in- and out-degrees from each
               other ("micro-configuration grid"). This is only recommended if z is not too large or the network is only
               plotted partly (see which_nodes). Otherwise the plot might get very busy. Default=False.
    which_nodes: Array indicating from which to which node the micro-configuration should be displayed. Use this if
                 interested in only a specific part of the network. Set to None (default) if the whole network should be
                 displayed.
    title:  String: title for the plot.
    cmap: String: Colormap for the plot. If the default 'binary' map is used, existing edges are displayed in black,
                  non-existing edges in white.
    save: Bool indicating whether the plot should be saved.
    plotname: String, relative path and filename for the plot if save==True.
              Do not forget the file extension, e.g. '.png'.
    """
    mc = sol.microconf(t=time)

    fig, ax = plt.subplots()
    ax.matshow(mc, cmap=cmap)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.set_ylabel("in")
    ax.xaxis.set_label_position('top')
    ax.set_xlabel("out")

    if edge_grid:
        ax.set_xticks(np.arange(-0.5, sol.network.z), minor=True)
        ax.set_yticks(np.arange(-0.5, sol.network.z), minor=True)
        # Hide the minor ticks.
        ax.tick_params(which='minor', top=False, bottom=False, left=False, right=False,
                       labeltop=True, labelbottom=False, labelleft=False, labelright=False)
        ax.grid(which='minor', c='grey')

    if node_grid:
        ax.vlines(np.cumsum(sol.network.x) - 0.5, ymin=-0.5, ymax=sol.network.z-0.5, color='deepskyblue', lw=3)
        ax.hlines(np.cumsum(sol.network.y) - 0.5, xmin=-0.5, xmax=sol.network.z-0.5, color='deepskyblue', lw=3)

    if which_nodes is not None:
        ax.set_xlim([np.sum(sol.network.x[:which_nodes[0]]) - 0.5,
                     np.sum(sol.network.x[:which_nodes[1] + 1]) - 1 + 0.5])
        ax.set_ylim([np.sum(sol.network.y[:which_nodes[1] + 1]) - 1 + 0.5,
                     np.sum(sol.network.y[:which_nodes[0]]) - 0.5])

    if title is not None:
        ax.set_title(title, y=-0.1)

    if save:
        if plotname is None:
            raise Exception("A name for the plot file has to be provided if it should be saved.")
        plt.savefig(plotname, dpi=400)

    plt.show()


def plot_microconfs(sols, times, nrows, ncols, figsize, node_grid=None, edge_grid=None, which_nodes=None,
                    cmap='binary', subtitles=None, title=None, save=False, plotname=None):
    """Visualize the micro-configuration of one or several networks at multiple time points of one or several
    MCMC-simulations.

    sols: List of solution object(s) of MCMC simulations. Has to be a list even if only one solution is given!
    times: Array with points in time at which the micro-configuration should be plotted. If sol is a list of solutions,
           the same times will be used for every solution.
    nrows: Number of rows of subplots.
    ncols: Number of columns of subplots.
    figsize: List indicating the size of the figure ([x, y]).
    node_grid: List of bools indicating whether a grid should be drawn that groups the in- and out-degrees according to
               which nodes they belong to ("macro-configuration grid"). Should be of length nrows * ncols. If not
               passed, a node grid is drawn for every subplot.
    edge_grid: List of bools indicating whether a grid should be drawn that separates all pairs of in- and out-degrees
               from each other ("micro-configuration grid"). This is only recommended if z is not too large or the
               network is only plotted partly (see which_nodes). Otherwise the plot might get very busy. Should be of
               length nrows * ncols. If not passed, no edge grids are drawn.
    which_nodes: Array indicating from which to which nodes the micro-configuration should be displayed. Use this if
                 interested in only a specific part of the network. Should have the same length as times and be in the
                 same order.
                 Set to None (default) if the whole network should be displayed at all times.
                 Example: which_nodes = np.array([[0, 3], [0, 10]]) if you want to plot the micro-configuration at two
                          different points in time (or at the same time twice) t1 and t2 and at t1 you only want to see
                          edges from i,j = 0 to 3 but at t2 i,j = 0 to 10.
    cmap: String : Colormap for the plot. If the default 'binary' map is used, existing edges are displayed in black,
                   non-existing edges in white.
    subtitles: List of strings with titles for the individual subplots.
    title:  String: title for the whole plot.
    save: Bool indicating whether the plot should be saved.
    plotname: String, relative path and filename for the plot if save==True.
              Do not forget the file extension, e.g. '.png'.
    """
    # Extract the edge count matrices from the solution object(s).
    mc_t = []

    for sol in sols:
        for t_i in times:
            mc_t.append(sol.microconf(t=t_i))

    # Settings for the grids.
    if node_grid is None:
        node_grid = [True] * nrows * ncols
    if edge_grid is None:
        edge_grid = [False] * nrows * ncols

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for idx, mc in enumerate(mc_t):
        if nrows == 1:
            row = 0
            col = idx
            axs[col].matshow(mc, cmap=cmap)
        else:
            row, col = divmod(idx, ncols)
            axs[row][col].matshow(mc, cmap=cmap)

        # Set ticks and tick labels on the y axis only if the subplot is on the left and on the x axis only if it is on
        # top.
        if row == 0:
            if nrows == 1:
                axs[col].tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
            else:
                axs[row][col].tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        else:
            axs[row][col].tick_params(top=True, bottom=False, labeltop=False, labelbottom=False)

        # Set x label only if the subplot is on top and y label only if the subplot is on the left.
        if row == 0:
            if nrows == 1:
                axs[col].xaxis.set_label_position('top')
                axs[col].set_xlabel("out")
            else:
                axs[row][col].xaxis.set_label_position('top')
                axs[row][col].set_xlabel("out")
        if col == 0:
            if nrows == 1:
                axs[col].set_ylabel("in")
            else:
                axs[row][col].set_ylabel("in")

        # Grids.
        if edge_grid[idx]:
            axs[row][col].set_xticks(np.arange(-0.5, sols[int(idx/len(times))].network.z), minor=True)
            axs[row][col].set_yticks(np.arange(-0.5, sols[int(idx/len(times))].network.z), minor=True)
            # Hide the minor ticks.
            axs[row][col].tick_params(which='minor', top=False, bottom=False, left=False, right=False,
                                      labeltop=True, labelbottom=False, labelleft=False, labelright=False)
            axs[row][col].grid(which='minor', c='grey')

        if node_grid[idx]:
            axs[row][col].vlines(np.cumsum(sols[int(idx/len(times))].network.x) - 0.5, ymin=-0.5,
                                 ymax=sols[int(idx/len(times))].network.z - 0.5, color='deepskyblue', lw=3)
            axs[row][col].hlines(np.cumsum(sols[int(idx/len(times))].network.y) - 0.5, xmin=-0.5,
                                 xmax=sols[int(idx/len(times))].network.z - 0.5, color='deepskyblue', lw=3)

        # Set axis range.
        if which_nodes is not None:
            axs[row][col].set_xlim([np.sum(sols[int(idx / len(times))].network.x[:which_nodes[idx][0]]) - 0.5,
                                    np.sum(
                                        sols[int(idx / len(times))].network.x[:which_nodes[idx][1] + 1]) - 1 + 0.5])
            axs[row][col].set_ylim(
                [np.sum(sols[int(idx / len(times))].network.y[:which_nodes[idx][1] + 1]) - 1 + 0.5,
                 np.sum(sols[int(idx / len(times))].network.y[:which_nodes[idx][0]]) - 0.5])

        # Set the title for the subplot.
        if subtitles is not None:
            if nrows == 1:
                axs[col].set_title(subtitles[idx], y=-0.2)
            else:
                axs[row][col].set_title(subtitles[idx], y=-0.2)

    if title is not None:
        fig.suptitle(title)

    if save:
        if plotname is None:
            raise Exception("A name for the plot file has to be provided if it should be saved.")
        plt.savefig(plotname, dpi=400)

    plt.show()


def plot_bound(sols, labels=[""], figsize=[6.4, 4.8], ylim=None, t_range=None, vlines=None, vlabels=None, vstyle=None,
               title=None, save=False, plotname=None):
    """Plot the time evolution of the total number of bound edges.

        sols: List of Solution object(s) of the simulations under consideration. Even if only one solution is given,
              this has to be a list with one element only!
        labels: List of strings of labels for the different sols (optional).
        figsize: List indicating the size of the figure ([x, y]). Default is the default figsize from matplotlib
                 ([6.4, 4.8]).
        ylim: List of two floats: limits for the y axis (optional).
        t_range: List of two floats: limits for the time axis (optional). If not given, the whole time evolution will be
                 plotted.
        vlines: List or array of vertical lines to plot, e.g. to mark special events/perturbations (optional).
        vlabels: List of strings: Labels for vlines (optional). Vertical lines do not necessarily require labels.
        vstyle: List of strings indicating the line style for vlines. Should have the same length as vlines and should
                always be given if vlines is given.
        title: String: Title for the plot.
        save: Bool indicating whether the plot should be saved.
        plotname: String: relative path and filename for the plot if save==True.
                  Do not forget the file extension, e.g. '.png'.
        """
    # TODO: break axis if wanted

    if len(labels) != len(sols):
        labels = np.full_like(sols, "")

    if t_range is None:
        t_range = np.array([min([sol.time[0] for sol in sols]), max([sol.time[-1] for sol in sols])])

    fig, ax = plt.subplots(figsize=figsize)
    for idx, sol in enumerate(sols):
        if len(sols) == 1:
            plt.plot(sol.time, sol.network.z - sol.z_hat_t(t_fin=t_range[-1]), c="royalblue", label=labels[idx])
        else:
            plt.plot(sol.time, sol.network.z - sol.z_hat_t(t_fin=t_range[-1]), label=labels[idx])
    if not all([l == "" for l in labels]):
        plt.legend()
    plt.xlabel("time")
    plt.ylabel(r"z - $\hat{\rm z}$(t)")

    if ylim is not None:
        plt.ylim(ylim)
    plt.xlim(t_range)

    # Vertical lines.
    if vlines is not None:
        for idx in range(len(vlines)):
            plt.axvline(vlines[idx], c="gray", ls=vstyle[idx])

    # Labels for vertical lines.
    if vlabels is not None:
        ax_t = ax.secondary_xaxis('top')
        ax_t.set_xticks(ticks=vlines, labels=vlabels)
        ax_t.tick_params(labelsize=8)

    ax.grid()

    if title is not None:
        plt.title(title)

    if save:
        if plotname is None:
            raise Exception("A name for the plot file has to be provided if it should be saved.")
        plt.savefig(plotname, dpi=400)

    plt.show()


def plot_lifetime_pdf(sols, stepsize, colors=None, labels=[""], figsize=[6.4, 4.8], xscale='linear', yscale='linear',
                      ylim=None, title=None, save=False, plotname=None):
    """Plot a histogram of the lifetimes of all synapses that were eliminated during the simulation (probability density
    function).

    sols: List of Solution object(s) of the simulation(s) of interest. Has to be a list even if only one solution is
         passed.
    stepsize: Float, stepsize in seconds used for the simulation(s). All simulations should have used the same stepsize!
    colors: List of colors to use for plotting the pdf of the different solutions. Only has to be passed if more than
            one solution object is passed. In that case it is mandatory!
    labels: List of strings: Labels for the different solution objects. Should have the same length as sols. Optional.
            If not passed, there will be no legend.
    figsize: List indicating the size of the figure ([x, y]). Default is matplotlib's default figsize of [6.4, 4.8].
    xscale: String indicating the x axis' scale. Options are: 'linear' (default), 'log', ('logit', 'symlog').
    yscale: String indicating the y axis' scale. Options are: 'linear' (default), 'log', ('logit', 'symlog').
    ylim: List of two floats: limits for the y axis (optional).
    title: String: Title for the plot.
    save: Bool indicating whether the plot should be saved.
    plotname: String: relative path and filename for the plot if save==True.
              Do not forget the file extension, e.g. '.png'.
    """
    if len(labels) != len(sols):
        labels = np.full_like(sols, "")

    plt.figure(figsize=figsize)

    if len(sols) == 1:
        plt.hist(sols[0].extra['lifetimes'],
                 bins=int((sols[0].extra['lifetimes'].max() - sols[0].extra['lifetimes'].min()) / (100 * stepsize)),
                 align='mid', color='crimson', label=labels[0],
                 weights=np.full_like(sols[0].extra['lifetimes'], 1 / len(sols[0].extra['lifetimes'])))
    else:
        for idx, sol in enumerate(sols):
            plt.hist(sol.extra['lifetimes'],
                     bins=int((sol.extra['lifetimes'].max() - sol.extra['lifetimes'].min()) / (100 * stepsize)),
                     align='mid', color=colors[idx], alpha=0.5, label=labels[idx],
                     weights=np.full_like(sol.extra['lifetimes'], 1 / len(sol.extra['lifetimes'])))

    plt.xscale(xscale)
    plt.yscale(yscale)

    plt.xlabel("synaptic lifetime [s]")
    plt.ylabel(r"pdf f(t) [s$^{-1}$]")
    if title is not None:
        plt.title(title)
    if not all([l == "" for l in labels]):
        plt.legend()

    if ylim is not None:
        plt.ylim(ylim)
    plt.grid()

    if save:
        if plotname is None:
            raise Exception("A name for the plot file has to be provided if it should be saved.")
        plt.savefig(plotname, dpi=400)

    plt.show()


def plot_survival_func(sols, labels=[""], colors=['limegreen'], figsize=[6.4, 4.8], xscale='linear', yscale='linear',
                       title=None, save=False, plotname=None):
    """Plot the survival function S(t) of all synapses.

    sols: List of solution object(s) of the simulation(s) of interest. Should always be a list even if only one solution
          is passed!
    labels: List of strings, labels for the survival functions of the different solutions.
    colors: List of strings, colors for the curves belonging to the respective solutions. Optional if only one solution
            is passed.
    figsize: List indicating the size of the figure ([x, y]). Default is matplotlib's default figsize of [6.4, 4.8].
    xscale: String indicating the x axis' scale. Options are: 'linear' (default), 'log', ('logit', 'symlog').
    yscale: String indicating the y axis' scale. Options are: 'linear' (default), 'log', ('logit', 'symlog').
    title: String: Title for the plot.
    save: Bool indicating whether the plot should be saved.
    plotname: String: relative path and filename for the plot if save==True.
              Do not forget the file extension, e.g. '.png'.
    """
    if len(labels) != len(sols):
        labels = np.full_like(sols, "")

    # TODO: pass a function fit the data to
    # should probably be a lambda function (?)
    # return and print optimized params + errors
    # plot fit func on top
    times = []
    fracs = []
    for sol in sols:
        t, f = sol.survival_func()
        times.append(t)
        fracs.append(f)

    plt.figure(figsize=figsize)

    for i in range(len(sols)):
        plt.plot(times[i], fracs[i], '.-', c=colors[i], label=labels[i])

    plt.xlabel("t [s]")
    plt.ylabel("S(t)")
    if not all([l == "" for l in labels]):
        plt.legend()
    if title is not None:
        plt.title(title)

    plt.xscale(xscale)
    plt.yscale(yscale)

    plt.grid()

    if save:
        if plotname is None:
            raise Exception("A name for the plot file has to be provided if it should be saved.")
        plt.savefig(plotname, dpi=400)

    plt.show()


def plot_lifetime_dist(sols, labels=[""], colors=['magenta'], figsize=[6.4, 4.8], xscale='linear', yscale='linear',
                       title=None, save=False, plotname=None):
    """Plot the lifetime distribution F(t) of all synapses.

        sols: List of solution object(s) of the simulation(s) of interest. Should always be a list even if only one
              solution is passed!
        labels: List of strings, labels for the lifetime distributions of the different solutions.
        colors: List of strings, colors for the curves belonging to the respective solutions. Optional if only one
                solution is passed.
        figsize: List indicating the size of the figure ([x, y]). Default is matplotlib's default figsize of [6.4, 4.8].
        xscale: String indicating the x axis' scale. Options are: 'linear' (default), 'log', ('logit', 'symlog').
        yscale: String indicating the y axis' scale. Options are: 'linear' (default), 'log', ('logit', 'symlog').
        title: String: Title for the plot.
        save: Bool indicating whether the plot should be saved.
        plotname: String: relative path and filename for the plot if save==True.
                  Do not forget the file extension, e.g. '.png'.
        """
    if len(labels) != len(sols):
        labels = np.full_like(sols, "")

    # TODO: pass a function fit the data to
    # should probably be a lambda function (?)
    # return and print optimized params + errors
    # plot fit func on top
    times = []
    fracs = []
    for sol in sols:
        t, f = sol.lifetime_dist()
        times.append(t)
        fracs.append(f)

    plt.figure(figsize=figsize)

    for i in range(len(sols)):
        plt.plot(times[i], fracs[i], '.-', c=colors[i], label=labels[i])

    plt.xlabel("t [s]")
    plt.ylabel("F(t)")
    if not all([l == "" for l in labels]):
        plt.legend()
    if title is not None:
        plt.title(title)

    plt.xscale(xscale)
    plt.yscale(yscale)

    plt.grid()

    if save:
        if plotname is None:
            raise Exception("A name for the plot file has to be provided if it should be saved.")
        plt.savefig(plotname, dpi=400)

    plt.show()
