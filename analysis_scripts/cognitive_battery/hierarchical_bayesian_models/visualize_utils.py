import matplotlib.pyplot as plt


def retrieve_config(config_fig, metric_type):
    rope_start, rope_end = config_fig[metric_type]["rope_start"], config_fig[metric_type]["rope_end"]
    xmin, xmax = config_fig[metric_type]["xmin"], config_fig[metric_type]["xmax"]
    figsize = (config_fig[metric_type]["figsizex"], config_fig[metric_type]["figsizey"])
    dpi = 1500
    y_offset, step_offset = 0, 0.5
    return rope_start, rope_end, xmin, xmax, figsize, dpi, y_offset, step_offset


def set_ax_deltas(ax, xmin, xmax):
    # Display and set properties for x and y axes
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    # Set color for x and y axis ticks
    ax.xaxis.set_tick_params(color='black', direction='out')
    ax.yaxis.set_tick_params(color='black', direction='out')
    # Make the labels of the y-ticks smaller
    ax.tick_params(axis='y', which='both', labelsize='small')
    ax.tick_params(axis='x', which='both', labelsize='small')
    ax.grid(color='grey', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Value')
    ax.set_xlim([xmin, xmax])
    ax.set_title('94% HDI')
    ax.set_facecolor("white")
    # Set the spines' line width
    line_width = plt.rcParams['axes.linewidth']
    # Set top and right spines to be visible and set their line width
    ax.spines['top'].set_visible(True)
    ax.spines['top'].set_linewidth(line_width)
    ax.spines['right'].set_visible(True)
    ax.spines['right'].set_linewidth(line_width)
    # Optionally, if you want to match the color as well, you can set it to the same as the bottom spine
    ax.spines['top'].set_color(ax.spines['bottom'].get_edgecolor())
    ax.spines['right'].set_color(ax.spines['bottom'].get_edgecolor())
    return ax
