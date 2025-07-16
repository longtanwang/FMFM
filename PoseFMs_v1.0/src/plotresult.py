import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plotprob(a,b,qualifiedid,name,outputdir):
    """
    Generates and saves a complex, multi-panel summary plot for a single
    seismic event's polarity analysis.

    This function creates four subplots to visualize:
    1. The main waveform with probability overlays (ax1).
    2. The probability density function of the amplitude threshold (ax2).
    3. The probability density function of the arrival time (ax3).
    4. A summary bar chart of the final polarity probabilities (ax4).

    Args:
        a (object): An object containing the waveform data, expected to have attributes
                    like .longtimestamp, .denselongdata, .cut, and .timestamp.
        b (object): An object containing the analysis results, expected to have
                    attributes like .timeprob, .upthreshold, .downthreshold, .num,
                    .Apeak, .arrivalestimate, etc.
        qualifiedid (int): The index of the specific, qualified solution to plot
                           from the analysis results.
        name (str): The identifier for the event/station, used in titles and filenames.
        outputdir (str): The directory path where the final plot image will be saved.
    """
    fig = plt.figure(figsize=(15, 9))
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    timeprob=b.timeprob[qualifiedid]
    ax1 = fig.add_axes([0.43, 0.33, 0.55, 0.6])
    ax1.plot(a.longtimestamp, a.denselongdata, linewidth=2.5, color='k',linestyle='-')
    ax1.plot(a.longtimestamp, abs(a.denselongdata), linewidth=2.5, color='k', linestyle=':',alpha=0.9)
    b.upthreshold[-1]=b.upthreshold[-2]*1.5
    colori=np.zeros([b.num,3])
    prob1=timeprob / (np.array(b.upthreshold) - np.array(b.downthreshold))
    alphacoefficient=(0.75-0.03)/np.max(prob1)
    tendindex=np.zeros(b.num)
    tchange=np.zeros(b.num)
    achange=np.zeros(b.num)
    downt=np.zeros(b.num)
    upt=np.zeros(b.num)
    for i in range(0,b.num):
        if (b.Apeak[i][0] > 0):
            colori[i, :] = np.array([1, 0, 0])
        if (b.Apeak[i][0] < 0):
            colori[i, :] = np.array([0, 1, 0])
        if(abs(b.Apeak[i][0])>0):
            tchange[i]=a.longtimestamp[a.cut[i]+1]-a.longtimestamp[a.cut[i]]
            achange[i]=a.denselongdata[a.cut[i]+1]-a.denselongdata[a.cut[i]]
            downt[i]=tchange[i]/achange[i]*(b.downthreshold[i]-abs(a.denselongdata[a.cut[i]]))+a.longtimestamp[a.cut[i]]
            upt[i]=tchange[i]/achange[i]*(b.upthreshold[i]-abs(a.denselongdata[a.cut[i]]))+a.longtimestamp[a.cut[i]]
            ax1.plot([downt[i],downt[i]],[b.downthreshold[i], -1 * b.upthreshold[-1]],linewidth=0.001, color='k',linestyle='--',alpha=0.3)
            ax1.plot([upt[i], upt[i]], [b.upthreshold[i], -1 * b.upthreshold[-1]], linewidth=0.001, color='k',linestyle='--',alpha=0.3)
            ax1.plot([0,downt[i]],[b.downthreshold[i], b.downthreshold[i]], linewidth=0.001, color='k',linestyle='--',alpha=0.3)
            ax1.plot([0,upt[i]], [b.upthreshold[i], b.upthreshold[i]], linewidth=0.001, color='k', linestyle='--',alpha=0.3)
            ax1.fill_between([0,downt[i],upt[i]], [b.downthreshold[i],b.downthreshold[i],b.upthreshold[i]], [b.upthreshold[i],b.upthreshold[i],b.upthreshold[i]], color=colori[i, :], alpha=0.03+timeprob[i]/(b.upthreshold[i]-b.downthreshold[i])*alphacoefficient)
            ax1.fill_betweenx([-1*b.upthreshold[-2],b.downthreshold[i],b.upthreshold[i]],[downt[i],downt[i],upt[i]],[upt[i],upt[i],upt[i]],color=colori[i, :], alpha=0.03+timeprob[i]/(b.upthreshold[i]-b.downthreshold[i])*alphacoefficient)
        if(b.Apeak[i][0]==0):
            downt[i]=a.longtimestamp[-1]
            upt[i]=a.longtimestamp[-1]+0.1*a.timestamp[-1]
            ax1.plot([downt[i], downt[i]], [b.downthreshold[i], -1 * b.upthreshold[-1]], linewidth=0.001, color='k',
                     linestyle='--', alpha=0.3)
            ax1.plot([upt[i], upt[i]], [b.upthreshold[i], -1 * b.upthreshold[-1]], linewidth=0.001, color='k',
                     linestyle='--', alpha=0.3)
            ax1.plot([0, downt[i]], [b.downthreshold[i], b.downthreshold[i]], linewidth=0.001, color='k', linestyle='--',
                     alpha=0.3)
            ax1.plot([0, upt[i]], [b.upthreshold[i], b.upthreshold[i]], linewidth=0.001, color='k', linestyle='--',
                     alpha=0.3)
            ax1.fill_between([0, downt[i], upt[i]], [b.downthreshold[i], b.downthreshold[i], b.upthreshold[i]],
                             [b.upthreshold[i], b.upthreshold[i], b.upthreshold[i]], color=colori[i, :],
                             alpha=0.03 + timeprob[i] / (b.upthreshold[i] - b.downthreshold[i]) * alphacoefficient)
            ax1.fill_betweenx([-1 * b.upthreshold[-2], b.downthreshold[i], b.upthreshold[i]],
                              [downt[i], downt[i], upt[i]], [upt[i], upt[i], upt[i]], color=colori[i, :],
                              alpha=0.03 + timeprob[i] / (b.upthreshold[i] - b.downthreshold[i]) * alphacoefficient)
    ax1.set(xlim=(0,np.max(a.densetimestamp)+0.1*a.timestamp[-1]), ylim=(-1*b.upthreshold[-2],b.upthreshold[-1]))
    ax1.tick_params(direction='out',size=20)
    ax1.set_yticks(np.array([-1*b.upthreshold[-2],0,b.upthreshold[-1]]))
    ax1.set_yticklabels(['%.2f' % (-1*b.upthreshold[-2]),'%.2f'%(0), '%.2f' % (b.upthreshold[-1])], fontweight='bold')
    ax1.set_xticks(np.linspace(0, a.timestamp[-1], 5))
    ax1.set_xticklabels(
        ['%.2f' % (0), '%.2f' % (a.timestamp[-1] / 4), '%.2f' % (a.timestamp[-1]/ 4 * 2), '%.2f' % (a.timestamp[-1]/ 4 * 3),
         '%.2f' % (a.timestamp[-1])], fontweight='bold')
    bwith = 2
    ax1.spines['bottom'].set_linewidth(bwith)
    ax1.spines['left'].set_linewidth(bwith)
    ax1.spines['top'].set_linewidth(bwith)
    ax1.spines['right'].set_linewidth(bwith)

    ax2 = fig.add_axes([0.1, 0.93-0.6/(b.upthreshold[-1]+b.upthreshold[-2])*b.upthreshold[-1], 0.25, 0.6/(b.upthreshold[-1]+b.upthreshold[-2])*b.upthreshold[-1]])
    ax2.invert_xaxis()
    for i in range(0,b.num):
        ax2.fill_betweenx(np.array([b.downthreshold[i],b.upthreshold[i]]),np.zeros(2), np.array([prob1[i],prob1[i]]), color=colori[i, :],
                             alpha=1)
        ax2.plot(np.array([0,prob1[i]]),
                 [b.upthreshold[i], b.upthreshold[i]], linewidth=0.001, color='k',
                 linestyle='--')
        ax2.plot(np.array([0,prob1[i]]),
                 [b.downthreshold[i], b.downthreshold[i]], linewidth=0.001, color='k',
                 linestyle='--')
    ax2.set(ylim=(0, b.upthreshold[-1]))
    ax2.set_xticks(np.linspace(0,np.max(prob1),5))
    ax2.set_xticklabels(['%.2f'%(0),'%.2f'%(np.max(prob1)/4),'%.2f'%(np.max(prob1)/4*2),'%.2f'%(np.max(prob1)/4*3),'%.2f'%(np.max(prob1))], fontweight='bold')
    ax2.set_yticks(np.linspace(0, b.upthreshold[-1],5))
    ax2.set_yticklabels(['%.2f'%(0),'%.2f'%(b.upthreshold[-1]/4),'%.2f'%(b.upthreshold[-1]/4*2),'%.2f'%(b.upthreshold[-1]/4*3),'%.2f'%(b.upthreshold[-1])], fontweight='bold')
    ax2.tick_params(direction='out', size=20)
    ax2.set_ylabel(r'$\mathbf{\varepsilon_{threshold}}$',weight='bold')
    ax2.set_xlabel(r'PDF of $\mathbf{\varepsilon_{threshold}}$',weight='bold')
    bwith = 2
    ax2.spines['bottom'].set_linewidth(bwith)
    ax2.spines['left'].set_linewidth(bwith)
    ax2.spines['top'].set_linewidth(bwith)
    ax2.spines['right'].set_linewidth(bwith)

    ax3 = fig.add_axes([0.43, 0.1, 0.55, 0.15],sharex=ax1)
    for i in range(0,b.num):
        ax3.plot([downt[i],downt[i]],[0,np.max(prob1)*1.1],linewidth=0.001, color='k', linestyle='-',alpha=0.3)
        ax3.plot([upt[i],upt[i]], [0, np.max(prob1)*1.1], linewidth=0.001, color='k', linestyle='-', alpha=0.3)
        ax3.plot([downt[i], upt[i]], [prob1[i], prob1[i]], linewidth=0.001, color='k', linestyle='-', alpha=0.3)
        ax3.fill_between([downt[i],upt[i]],[0,0],[prob1[i], prob1[i]],color=colori[i, :],alpha=1)
    ax3.plot([b.arrivalestimate,b.arrivalestimate],[0,np.max(prob1)*1.1],linewidth=2.3, color='k', linestyle=':',alpha=0.9)
    ax3.set(ylim=(0,np.max(prob1)*1.1))
    ax3.set_yticks(np.linspace(0, np.max(prob1), 3))
    ax3.set_yticklabels(['%.2f'%(0),'%.2f'%(np.max(prob1) / 2),'%.2f'%(np.max(prob1))], fontweight='bold')
    ax3.set_xticks(np.linspace(0,a.timestamp[-1],5))
    ax3.set_xticklabels(['%.2f'%(0),'%.2f'%(a.timestamp[-1]/4),'%.2f'%(a.timestamp[-1]/4*2),'%.2f'%(a.timestamp[-1]/4*3),'%.2f'%(a.timestamp[-1])], fontweight='bold')

    ax3.tick_params(direction='out', size=20)
    ax3.set_ylabel('PDF of Time',weight='bold')
    ax3.set_xlabel('Time/s', weight='bold')
    bwith=2
    ax3.spines['bottom'].set_linewidth(bwith)
    ax3.spines['left'].set_linewidth(bwith)
    ax3.spines['top'].set_linewidth(bwith)
    ax3.spines['right'].set_linewidth(bwith)

    ax4 = fig.add_axes([0.1, 0.13, 0.23, 0.05])
    y=[1,1,1]
    width=[np.float(b.polarityup),np.float(b.polarityunknown),np.float(b.polaritydown)]
    leftcoordinate=[0,np.float(b.polarityup),np.float(b.polarityup+b.polarityunknown)]
    colorii=[[1,0,0],[0.7,0.7,0.7],[0,1,0]]
    labelii=['Up','Unknown','Down']
    for i in range(0, 3):
        ax4.barh(y[i], width[i], height=1,left=leftcoordinate[i], color=colorii[i], label=labelii[i])
    ax4.set_xticks(np.array([]))
    ax4.text(0, 2,'Up:%.1f%%'%(np.float(b.polarityup)*100))
    ax4.text(0.85, 2,'Down:%.1f%%'%(np.float(b.polaritydown)*100))
    ax4.set_yticks([])
    ax4.set(xlim=(0, 1),ylim=(0.5,1.5))
    ax4.plot([0.5,0.5],[0,1.5],linewidth=2.5, color='k', linestyle=':',alpha=1)
    bwith = 2
    ax4.spines['bottom'].set_linewidth(bwith)
    ax4.spines['left'].set_linewidth(bwith)
    ax4.spines['top'].set_linewidth(bwith)
    ax4.spines['right'].set_linewidth(bwith)
    ax4.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.8))

    fig.text(0.21,0.38,'%s'%(name.split('.txt')[0]),{'fontweight':'bold','fontsize':25},horizontalalignment='center')
    #fig.text(0.24, 0.33,r'$\mathbf{A_{peak}}$'+': %.3f'%(b.Apeakestimate),{'fontweight':'bold','fontsize':15})
    #fig.text(0.24, 0.28,r'$\mathbf{\sigma}$'+': %.3f'%(b.sigmaestimate),{'fontweight':'bold','fontsize':15})
    fig.text(0.08, 0.33,'Arrivaltime'+': %.3fs'%(b.arrivalestimate),{'fontweight':'bold','fontsize':15})
    #fig.text(0.08, 0.28,'Polarity Up'+': %.3f'%(b.polarityestimation),{'fontweight':'bold','fontsize':15})
    fig.savefig('%s'%(outputdir)+'%s_%d.pdf'%(name,qualifiedid),dpi=100)

