import json
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from os import listdir
from os.path import join, isfile


def load_evaluation(filename):
    """Load single evaluation file. Adds the filename for reference"""
    data = None
    with open(filename) as fin:
        try:
            data = json.loads(fin.read())
            data['filename'] = filename
        except json.JSONDecodeError:
            print(f"Could not JSON decode {filename}")
    return data


def load_evaluation_dir(dirname):
    """sLoad all evaluation files in a directory"""
    evaluations = [load_evaluation(join(dirname, f))\
                   for f in listdir(dirname) if isfile(join(dirname, f))]
    return [ev for ev in evaluations if ev]


def get_benchmark_df(evaluations, benchmark, metric="cosine", eval_metric="pearson"):
    """Selects the most important data from a benchmark and returns a DF"""
    benchmark_data = {}
    for ev in evaluations:
        if ev['benchmark'] == benchmark and ev['metric'] == metric:
            value = ev['evaluation']
            value = value['pearson']['r'] if eval_metric == 'pearson' \
                else value['spearman']['rho']
            benchmark_data[ev['tag']] = {
                'value': value,
                'timestamp': ev['timestamp'],
                'class': ev['class'],
                'model_url': ev['model_url']
            }
    return pd.DataFrame(benchmark_data).transpose()


def plot_benchmark(bmark_df, title, savefile=""):
    """Plots a benchmark dataframe and optionally saves the file"""
    ax = bmark_df["value"].sort_values(ascending=True).plot.barh()
    annotate_barh(ax)
    plt.title(title)
    if savefile:
        plt.savefig(savefile)

        
#from https://stackoverflow.com/questions/1855884/determine-font-color-based-on-background-color
def contrast_color(color, blackish='black', whiteish='whitesmoke'):
    """Selects white(ish) or black(ish) for text to contrast over some RGB"""
    luminance = (0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2])
    if luminance > 0.6:
        return blackish
    return whiteish


#TODO: Revisar
#https://matplotlib.org/3.1.3/gallery/statistics/barchart_demo.html
#para mejor manejo de la posición del texto.
# Hacer el blog post despues con la solucion
##http://eduardofv.com/wp-admin/post.php?post=517&action=edit
#https://colab.research.google.com/drive/1kwKuOwim7ngYmFSRjkVYMi5_K6WA9vmD
def annotate_barh(ax, fontsize='small'):
    """Adds value labels inside the bars of a barh plot"""
    plt.draw()
    for patch in ax.patches:
        value = patch.get_width()
        formatter = ax.get_xaxis().get_major_formatter()
        #label = formatter.format_data(value)
        #label = formatter.format_pct(x=value, display_range=10)
        label = f"{patch.get_width():1.4f}"
        p_x = patch.get_width()
        p_y = patch.get_y()
        #Put an invisible text to measure it's dimensions
        txt = plt.text(p_x, p_y, label, fontsize=fontsize, alpha=0.0)
        bbox = txt.get_window_extent().transformed(ax.transData.inverted())
        t_w = bbox.width * 1.1
        t_h = bbox.height
        p_y += (patch.get_height() - t_h)/1.5
        if t_w > 0.9 * patch.get_width():
            plt.text(p_x, p_y, label, fontsize=fontsize)
        else:
            p_x -= t_w
            col = contrast_color(patch.get_facecolor())
            plt.text(p_x, p_y, label, fontsize=fontsize, color=col)

