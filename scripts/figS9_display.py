"""V203 SI Fig. S9: display frozen six-/five-domain income coefficients."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def render(source: Path, output: Path) -> None:
    """Plot archived estimates and clustered intervals; fit no model."""
    data = pd.read_csv(source)
    plt.rcParams.update({'font.family':'Arial','font.size':10,'axes.spines.top':False,
                        'axes.spines.right':False,'svg.fonttype':'none'})
    fig, axes = plt.subplots(1,2,figsize=(9,4.2),sharey=True)
    years = [2011,2016,2021,2024]
    for ax, model, title in zip(axes,['Income only','Income + market composition'],['Income only','Adjusted model']):
        for label, offset, colour in [('Six-domain primary',-.11,'#2B70AF'),('Five-domain sensitivity',.11,'#BD5A67')]:
            rows = data.loc[data.specification.eq(label)&data.model.eq(model)&data.predictor.eq('log_income')].set_index('year').loc[years]
            values = rows.standardized_beta.to_numpy()
            ax.errorbar(values,np.arange(4)+offset,
                        xerr=np.array([values-rows.ci_low.to_numpy(),rows.ci_high.to_numpy()-values]),
                        fmt='o',color=colour,capsize=3,markersize=5,lw=1.2,label=label)
        ax.axvline(0,color='#7B858C',ls='--',lw=.8)
        ax.set_title(title,loc='left',pad=10)
        ax.set_yticks(range(4),years)
        ax.set_ylim(3.4,-.4)
        ax.set_xlim(-.02,.40)
        ax.set_xlabel('Standardized income coefficient (95% CI)',labelpad=7)
        ax.tick_params(axis='both',labelsize=10)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.015))
    fig.subplots_adjust(left=.08,right=.98,bottom=.22,top=.90,wspace=.22)
    for extension in ('png','svg','pdf'):
        fig.savefig(output/f'FigS9_Six_Five_Income_Comparison.{extension}',dpi=600,facecolor='white')
    plt.close(fig)
