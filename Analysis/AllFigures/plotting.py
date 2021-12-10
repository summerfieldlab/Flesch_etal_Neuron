import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import LinearSegmentedColormap
from scipy.linalg import expm 
from numpy.linalg import norm
from datetime import datetime
from scipy.spatial.distance import squareform,pdist
from sklearn.utils import shuffle
from sklearn.manifold import MDS
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D 
from scipy.stats import multivariate_normal
from scipy.stats import zscore, ttest_1samp

def plot_roi_betas(betas,titlestr='ROI: ',modelrdms=['Grid','Rotated Grid', 'Orthogonal','Parallel','Branchiness','Leafiness','Diagonal']):
    mm = 1/25.4
    plt.figure(figsize=(30*mm,50*mm), dpi=300)
    print(titlestr +':')
    for ii in range(7):
        bctx = plt.bar(ii,np.mean(betas[:,ii]),yerr=np.std(betas[:,ii])/(len(betas[:,ii])**.5),width=0.6,color=(0.6,0.7,.8),edgecolor='k',linewidth=0.5)
        plt.scatter(np.repeat(ii,len(betas[:,ii]))+np.random.randn(len(betas[:,ii]))*0.01,betas[:,ii],color=(.8,.8,.8),alpha=.4,zorder=3,s=1,edgecolors='k',linewidth=0.5)
        plt.plot([-2,8],[0,0],'k-',linewidth=0.5)
        plt.xticks(ticks=np.arange(7),labels=modelrdms,rotation=90,fontsize=6)
        t,p = ttest_1samp(betas[:,ii],0,alternative='greater')
        # bonferroni correction: 
        # p *=7
        print(f'{modelrdms[ii]}: \t t= {t:.2f}, \t p= {p:.4f}')
        if t > 1.96:
            if p <0.0001/7:
                ts = '*'*4
            elif p <0.001/7:
                ts = '*'*3
            elif p<0.01/7:
                ts = '*'*2
            elif p<0.05/7:
                ts = '*'
            plt.text(ii,0.15,ts,{'fontsize':6,'ha':'center','fontweight':'bold'})
    ax = plt.gca()  
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(titlestr,fontsize=6)
    ax.set_yticks(np.arange(-0.2,0.22,0.2))
    ax.set(xlabel='model RDM', ylabel=r'$\beta$ estimate (a.u.)')
    ax.set_xlim(-0.5,6.5)
    plt.yticks(fontsize=6)


def plot_bars_humandata(baseline,scanner,col1=(20/255, 106/255, 245/255),col2=(81/255, 174/255, 240/255)):
    plt.style.use('default')    
    
    b = plt.bar(np.asarray([1,2]),[baseline.mean(),scanner.mean()],width=.8,linewidth=1)
    b[0].set_color((col1))
    b[1].set_color(col2)
    b[0].set_edgecolor('None')
    b[1].set_edgecolor('None')
    idx_1 = np.ones((1,30))+np.random.randn(30)*5e-2
    idx_2 = 2*np.ones((1,30))+np.random.randn(30)*5e-2
    plt.plot(np.concatenate((idx_1,idx_2)),np.concatenate((baseline,scanner)),color=(.8, .8, .8),alpha=.8,zorder=3,linewidth=.5)
    plt.scatter(idx_1,baseline,s=8,color=(col1),zorder=4,edgecolor='black',alpha=.4,linewidth=.5)
    plt.scatter(idx_2,scanner,s=8,color=col2,zorder=4,edgecolor='black',alpha=.4,linewidth=.5)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.errorbar(np.asarray([1,2]),[baseline.mean(),scanner.mean()],np.std(np.concatenate([baseline,scanner]),axis=1)/np.sqrt(30),linestyle='None',linewidth=1,color='black',zorder=6)
    _ = plt.xticks([1,2],labels=['baseline','scanner'],fontsize=6,rotation=90)
    plt.xlim((0,3))


def set_choicemat_axlabels(ax):
    ax.set_xlabel('branchiness',fontsize=6)
    ax.set_ylabel('leafiness',fontsize=6)
    ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)


def comp_norms(weights):
    norms = np.empty((8,N_RUNS))
    for ii in range(8):
        for jj in range(N_RUNS):
            norms[ii,jj] = np.linalg.norm(weights[ii,jj,:,:].flatten())
    return norms

def plot_norms(norms,titlestr,SCALE_WHXS,zorder=1,colors=np.repeat([1,0,0],8,axis=0)):
    for ii in range(8):
        plt.bar(ii,np.mean(norms[ii,:],0),yerr=np.std(norms[ii,:],0)/np.sqrt(len(norms[ii,:])),zorder=zorder,color=colors[ii,:])
    # plt.title(titlestr,fontsize=14)
    plt.ylabel('norm')
    plt.xticks(ticks=np.arange(0,8),labels=[str(i) for i in SCALE_WHXS])
    # plt.xlabel('scaling factor')


def helper_make_colormap(basecols=np.array([[63,39,24], [64,82,21], [65,125,18], [66,168,15], [68,255, 10]])/255,n_items=5, monitor=False):
    '''
    creates a colormap and returns both the cmap object
    and a list of rgb tuples
    inputs:
    -basecols: nump array with rgb colors spanning the cmap
    -n_items: sampling resolution of cmap
    outputs:
    -cmap: the cmap object
    -cols: np array of colors spanning cmap
    '''
    # turn basecols into list of tuples     
    basecols = list(map(lambda x: tuple(x),basecols))
    # turn basecols into colour map
    cmap = LinearSegmentedColormap.from_list('tmp',basecols,N=n_items)
    # if desired, plot results
    if monitor:
        plt.figure()
        plt.imshow(np.random.randn(20,20),cmap=cmap)
        plt.colorbar()
    cols = np.asarray([list(cmap(c)[:3]) for c in range(n_items)])

    return cmap, cols
        



def plot_grid3(xyz,line_colour='green',line_width=2,fig_id=1):
    # %matplotlib qt    
    x,y = np.meshgrid(np.arange(0,5),np.arange(0,5))
    x = x.flatten()
    y = y.flatten()
    try: xyz 
    except NameError: xyz = np.stack((x,y,np.ones((25,))),axis=1)
    bl = np.stack((x,y),axis=1)
    plt.figure(fig_id)    
    ax = plt.gca()
    for ii in range(0,4):
        for jj in range(0,4):
            p1 = xyz[(bl[:,0]==ii) & (bl[:,1]==jj),:].ravel()
            p2 = xyz[(bl[:,0]==ii+1) & (bl[:,1]==jj),:].ravel()        
            plt.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],linewidth=line_width,color=line_colour)
            p2 = xyz[(bl[:,0]==ii) & (bl[:,1]==jj+1),:].ravel()
            plt.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],linewidth=line_width,color=line_colour)
    ii = 4
    for jj in range(0,4):
        p1 = xyz[(bl[:,0]==ii) & (bl[:,1]==jj),:].ravel()
        p2 = xyz[(bl[:,0]==ii) & (bl[:,1]==jj+1),:].ravel()
        plt.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],linewidth=line_width,color=line_colour)

    jj = 4
    for ii in range(0,4):
        p1 = xyz[(bl[:,0]==ii) & (bl[:,1]==jj),:].ravel()
        p2 = xyz[(bl[:,0]==ii+1) & (bl[:,1]==jj),:].ravel()
        plt.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],linewidth=line_width,color=line_colour)
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    ax.set_xlim([-2,2])
    ax.set_ylim([-2,2])
    ax.set_zlim([-2,2])

def plot_grid2(xy,line_colour='green',line_width=1,fig_id=1,n_items=5):
    '''
    n_items: number of items along one dimension covered by grid
    '''
    # %matplotlib qt    
    x,y = np.meshgrid(np.arange(0,n_items),np.arange(0,n_items))
    x = x.flatten()
    y = y.flatten()    
    try: xy 
    except NameError: xy = np.stack((x,y),axis=1)
    bl = np.stack((x,y),axis=1)
    fig = plt.figure(fig_id)
    
    for ii in range(0,n_items-1):
        for jj in range(0,n_items-1):
            p1 = xy[(bl[:,0]==ii) & (bl[:,1]==jj),:].ravel()
            p2 = xy[(bl[:,0]==ii+1) & (bl[:,1]==jj),:].ravel()        
            plt.plot([p1[0],p2[0]],[p1[1],p2[1]],linewidth=line_width,color=line_colour)
            p2 = xy[(bl[:,0]==ii) & (bl[:,1]==jj+1),:].ravel()
            plt.plot([p1[0],p2[0]],[p1[1],p2[1]],linewidth=line_width,color=line_colour)
    ii = n_items-1
    for jj in range(0,n_items-1):
        p1 = xy[(bl[:,0]==ii) & (bl[:,1]==jj),:].ravel()
        p2 = xy[(bl[:,0]==ii) & (bl[:,1]==jj+1),:].ravel()
        plt.plot([p1[0],p2[0]],[p1[1],p2[1]],linewidth=line_width,color=line_colour)

    jj = n_items-1
    for ii in range(0,n_items-1):
        p1 = xy[(bl[:,0]==ii) & (bl[:,1]==jj),:].ravel()
        p2 = xy[(bl[:,0]==ii+1) & (bl[:,1]==jj),:].ravel()
        plt.plot([p1[0],p2[0]],[p1[1],p2[1]],linewidth=line_width,color=line_colour)
    ax = plt.gca()
    # ax.axes.xaxis.set_ticklabels([])
    # ax.axes.yaxis.set_ticklabels([])  
    ax.set_xlim([-2,2])
    ax.set_ylim([-2,2])
    


def scatter_mds_3(xyz,task_id='a',fig_id=1):
    if task_id=='both':
        n_items = 50
        ctxMarkerEdgeCol = [(0,0,.5),(255/255,102/255,0)]
    elif task_id=='a':
        n_items = 25
        ctxMarkerEdgeCol = (0,0,.5)
    elif task_id=='b':
        ctxMarkerEdgeCol = (255/255,102/255,0)
    elif task_id == 'avg':
        ctxMarkerEdgeCol = None

    ctxMarkerCol = 'white'
    ctxMarkerSize = 20
    itemMarkerSize = 2
    scat_b = np.arange(1,15,2)    
    _,scat_l  = helper_make_colormap(basecols=np.array([[63,39,24], [64,82,21], [65,125,18], [66,168,15], [68,255, 10]])/255,n_items=5,monitor=False)

    b,l = np.meshgrid(np.arange(0,5),np.arange(0,5))
    b = b.flatten()
    l = l.flatten()   
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
    
    
    fig = plt.figure(fig_id)
    ax = plt.gca()
    if task_id == 'both':
        b = np.concatenate((b,b),axis=0)
        l = np.concatenate((l,l),axis=0)
        
        for ii in range(0,n_items//2):
            
            plt.plot([x[ii],x[ii]],[y[ii],y[ii]],[z[ii],z[ii]],marker='s',markerfacecolor=ctxMarkerCol,markeredgecolor=ctxMarkerEdgeCol[0],markersize=ctxMarkerSize,markeredgewidth=2)
            plt.plot([x[ii],x[ii]],[y[ii],y[ii]],[z[ii],z[ii]],marker='h',markeredgecolor=scat_l[l[ii],:],markerfacecolor=scat_l[l[ii],:],markersize=scat_b[b[ii]])
        for ii in range(n_items//2,n_items):
            
            plt.plot([x[ii],x[ii]],[y[ii],y[ii]],[z[ii],z[ii]],marker='s',markerfacecolor=ctxMarkerCol,markeredgecolor=ctxMarkerEdgeCol[1],markersize=ctxMarkerSize,markeredgewidth=2)
            plt.plot([x[ii],x[ii]],[y[ii],y[ii]],[z[ii],z[ii]],marker='h',markeredgecolor=scat_l[l[ii],:],markerfacecolor=scat_l[l[ii],:],markersize=scat_b[b[ii]])
    else:
        for ii in range(0,n_items):
            
            plt.plot([x[ii],x[ii]],[y[ii],y[ii]],[z[ii],z[ii]],marker='s',markerfacecolor=ctxMarkerCol,markeredgecolor=ctxMarkerEdgeCol,markersize=ctxMarkerSize,markeredgewidth=2)
            plt.plot([x[ii],x[ii]],[y[ii],y[ii]],[z[ii],z[ii]],marker='h',markeredgecolor=scat_l[l[ii],:],markerfacecolor=scat_l[l[ii],:],markersize=scat_b[b[ii]])




def scatter_mds_2(xyz,task_id='a',fig_id=1,flipdims=False,items_per_dim=5,flipcols=False,marker_scale=1):
    
    if flipcols==True:
        col1 = (0, 0, .5)
        col2 = (255/255,102/255,0)
    else:
        col1 = (255/255,102/255,0)
        col2 = (0, 0, .5)

    if task_id=='both':
        n_items = items_per_dim**2*2
        ctxMarkerEdgeCol = [col1,col2]
    elif task_id=='a':
        n_items = items_per_dim**2
        ctxMarkerEdgeCol = col1
    elif task_id=='b':
        n_items = items_per_dim**2
        ctxMarkerEdgeCol = col2
    elif task_id == 'avg':
        n_items = items_per_dim**2
        ctxMarkerEdgeCol = 'k'

    ctxMarkerCol = 'white'
    ctxMarkerSize = 4*marker_scale
    itemMarkerSize = 1
    scat_b = np.linspace(.5,2.5,items_per_dim)*marker_scale
    # # scat_l = np.array([[3,252,82], [3,252,177], [3,240,252], [3,152,252], [3,73,252]])/255
    # scat_l = np.array([[63,39,24], [64,82,21], [65,125,18], [66,168,15], [68,255, 10]])/255
    # # scat_l = np.array([[0,0,0], [.2,.2,.2],[.4,.4,.4],[.6,.6,.6],[.8,.8,.8]])
    _,scat_l  = helper_make_colormap(basecols=np.array([[63,39,24], [64,82,21], [65,125,18], [66,168,15], [68,255, 10]])/255,n_items=items_per_dim,monitor=False)

    b,l = np.meshgrid(np.arange(0,items_per_dim),np.arange(0,items_per_dim))
    if flipdims==True:
        l,b = np.meshgrid(np.arange(0,items_per_dim),np.arange(0,items_per_dim))

    b = b.flatten()
    l = l.flatten()   
    x = xyz[:,0]
    y = xyz[:,1]

    if task_id == 'both':
        b = np.concatenate((b,b),axis=0)
        l = np.concatenate((l,l),axis=0)
        fig = plt.figure(fig_id)

        for ii in range(0,n_items//2):            
            plt.plot([x[ii],x[ii]],[y[ii],y[ii]],marker='s',markerfacecolor=ctxMarkerCol,markeredgecolor=ctxMarkerEdgeCol[0],markersize=ctxMarkerSize,markeredgewidth=.5)
            plt.plot(x[ii],y[ii],marker='h',markeredgecolor=scat_l[l[ii],:],markerfacecolor=scat_l[l[ii],:],markersize=scat_b[b[ii]])

        for ii in range(n_items//2,n_items):            
            plt.plot(x[ii],y[ii],marker='s',markerfacecolor=ctxMarkerCol,markeredgecolor=ctxMarkerEdgeCol[1],markersize=ctxMarkerSize,markeredgewidth=.5)
            plt.plot(x[ii],y[ii],marker='h',markeredgecolor=scat_l[l[ii],:],markerfacecolor=scat_l[l[ii],:],markersize=scat_b[b[ii]])
    else:
        for ii in range(0,n_items):
            
            plt.plot(x[ii],y[ii],marker='s',markerfacecolor=ctxMarkerCol,markeredgecolor=ctxMarkerEdgeCol,markersize=ctxMarkerSize,markeredgewidth=.5)
            plt.plot(x[ii],y[ii],marker='h',markeredgecolor=scat_l[l[ii],:],markerfacecolor=scat_l[l[ii],:],markersize=scat_b[b[ii]])


def plot_MDS_embeddings_2D(embedding,fig,fig_id=2,axlims=2.5,flipdims=False,monk=False,flipcols=False):
        

        # if flipdims==True:
        #     col1 = (0, 0, .5)
        #     col2 = (255/255,102/255,0)

        if flipcols==True:
            col1 = (0, 0, .5)
            col2 = (255/255,102/255,0)
        else:
            col1 = (255/255,102/255,0)
            col2 = (0, 0, .5)

        if monk==True:
            n_items = 6
            ii_half = 36
        else:            
            n_items = 5
            ii_half = 25

        plt.subplot(1,2,1)    
        plot_grid2(embedding[0:ii_half,[0,1]],line_width=.5,line_colour=col1,fig_id=fig_id,n_items=n_items)
        scatter_mds_2(embedding[0:ii_half,[0,1]],fig_id=fig_id,task_id='a',flipdims=flipdims,items_per_dim=n_items,flipcols=flipcols)
        plot_grid2(embedding[ii_half:,[0,1]],line_width=.5,line_colour=col2,fig_id=fig_id,n_items=n_items)
        scatter_mds_2(embedding[ii_half:,[0,1]],fig_id=fig_id,task_id='b',flipdims=flipdims,items_per_dim=n_items,flipcols=flipcols)
        ax = plt.gca()
        ax.set_xlim([-axlims,axlims])
        ax.set_ylim([-axlims,axlims])    
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        plt.xlabel('dim 1',fontsize=6)
        plt.ylabel('dim 2',fontsize=6)
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_aspect('equal', 'box')

        plt.subplot(1,2,2)
        plot_grid2(embedding[0:ii_half,[1,2]],line_width=.5,line_colour=col1,fig_id=fig_id,n_items=n_items)
        scatter_mds_2(embedding[0:ii_half,[1,2]],fig_id=fig_id,task_id='a',flipdims=flipdims,items_per_dim=n_items,flipcols=flipcols)
        plot_grid2(embedding[ii_half:,[1,2]],line_width=.5,line_colour=col2,fig_id=fig_id,n_items=n_items)
        scatter_mds_2(embedding[ii_half:,[1,2]],fig_id=fig_id,task_id='b',flipdims=flipdims,items_per_dim=n_items,flipcols=flipcols)
        ax = plt.gca()
        ax.set_xlim([-axlims,axlims])
        ax.set_ylim([-axlims,axlims])
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        plt.xlabel('dim 2',fontsize=6)
        plt.ylabel('dim 3',fontsize=6)
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_aspect('equal', 'box')

def scatter_displaceddots():
    
    
    n_items = 25
    ctxMarkerEdgeCol = (0,0,0)
    ctxMarkerCol = 'white'
    ctxMarkerSize = 35
    itemMarkerSize = 5
    itemMarkerCol = 'k'
    scat_b = np.linspace(-.4,.4,5)
    scat_l = np.linspace(-.4,.4,5)
    
    b,l = np.meshgrid(np.linspace(0,10,5),np.linspace(0,10,5))
    b = b.flatten()
    l = l.flatten()   
    
    b_d,l_d = np.meshgrid(scat_b,scat_l)
    b_d = b_d.flatten()
    l_d = l_d.flatten()
    
    x = b+b_d
    y = l+l_d

    fig = plt.figure(figsize=(5.84,4.47))
    for ii in range(0,n_items):
        
        plt.plot(b[ii],l[ii],marker='s',markerfacecolor=ctxMarkerCol,markeredgecolor=ctxMarkerEdgeCol,markersize=ctxMarkerSize,markeredgewidth=.5)
        plt.plot(x[ii],y[ii],marker='o',markeredgecolor=(.4,.4,.4),markerfacecolor=(.4,.4,.4),markersize=itemMarkerSize*1.8)
        plt.plot(x[ii],y[ii],marker='o',markeredgecolor=itemMarkerCol,markerfacecolor=itemMarkerCol,markersize=itemMarkerSize)
    plt.xticks(ticks=np.linspace(0,10,5),labels=np.arange(1,6),fontsize=12)
    plt.yticks(ticks=np.linspace(0,10,5),labels=np.arange(1,6),fontsize=12)
    plt.xlabel('translation along x-axis')
    plt.ylabel('translation along y-axis')
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def scatter_richlearning():

    n_items = 25
    ctxMarkerEdgeCol = (0,0,0)
    ctxMarkerCol = 'white'
    ctxMarkerSize = 35
    itemMarkerSize = 5
    itemMarkerCol = 'k'
    scat_b = np.linspace(-.4,.4,5)
    scat_l = np.linspace(-.4,.4,5)
    
    b,l = np.meshgrid(np.linspace(0,10,5),np.linspace(0,10,5))
    b = b.flatten()
    l = l.flatten()   
    
    b_d,l_d = np.meshgrid(scat_b,scat_l)
    b_d = b_d.flatten()
    l_d = l_d.flatten()
    
    # linearity step
    fig = plt.figure(figsize=(5.84,4.47))
    b,l = np.meshgrid(np.linspace(0,10,5),np.linspace(0,10,5))
    b = b.flatten()
    l = l.flatten()   
    x = b+b_d-10
    y = l+l_d
    b = b-10 

    for ii in range(0,n_items):
        
        plt.plot(b[ii],l[ii],marker='s',markerfacecolor=ctxMarkerCol,markeredgecolor=ctxMarkerEdgeCol,markersize=ctxMarkerSize,markeredgewidth=2)
        plt.plot(x[ii],y[ii],marker='o',markeredgecolor=(.4,.4,.4),markerfacecolor=(.4,.4,.4),markersize=itemMarkerSize*1.8)
        plt.plot(x[ii],y[ii],marker='o',markeredgecolor=itemMarkerCol,markerfacecolor=itemMarkerCol,markersize=itemMarkerSize)
    # plt.xticks(ticks=np.linspace(0,10,5),labels=np.arange(1,6),fontsize=12)
    # plt.yticks(ticks=np.linspace(0,10,5),labels=np.arange(1,6),fontsize=12)
    plt.xlabel('translation along x-axis')
    plt.ylabel('translation along y-axis')
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.grid()
    b,l = np.meshgrid(np.linspace(0,10,5),np.linspace(0,10,5))
    b = b.flatten()
    l = l.flatten()     
    x = b+b_d
    y = l+l_d-10
    l=l-10

    for ii in range(0,n_items):
        
        plt.plot(b[ii],l[ii],marker='s',markerfacecolor=ctxMarkerCol,markeredgecolor=ctxMarkerEdgeCol,markersize=ctxMarkerSize,markeredgewidth=2)
        plt.plot(x[ii],y[ii],marker='o',markeredgecolor=(.4,.4,.4),markerfacecolor=(.4,.4,.4),markersize=itemMarkerSize*1.8)
        plt.plot(x[ii],y[ii],marker='o',markeredgecolor=itemMarkerCol,markerfacecolor=itemMarkerCol,markersize=itemMarkerSize)
    # plt.xticks(ticks=np.linspace(0,10,5),labels=np.arange(1,6),fontsize=12)
    # plt.yticks(ticks=np.linspace(0,10,5),labels=np.arange(1,6),fontsize=12)
    plt.xlabel('translation along x-axis')
    plt.ylabel('translation along y-axis')
    plt.grid()
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # nonlinearity step
    fig = plt.figure(figsize=(5.84,4.47))
    b,l = np.meshgrid(np.linspace(0,10,5),np.linspace(0,10,5))
    b = b.flatten()
    l = l.flatten()   
    x = b+b_d-10
    y = l+l_d
    b = b -10
    b[b<=0] = 0
    x[x<=0] = 0
    for ii in range(0,n_items):
        
        plt.plot(b[ii],l[ii],marker='s',markerfacecolor=ctxMarkerCol,markeredgecolor=ctxMarkerEdgeCol,markersize=ctxMarkerSize,markeredgewidth=2)
        plt.plot(x[ii],y[ii],marker='o',markeredgecolor=(.4,.4,.4),markerfacecolor=(.4,.4,.4),markersize=itemMarkerSize*1.8)
        plt.plot(x[ii],y[ii],marker='o',markeredgecolor=itemMarkerCol,markerfacecolor=itemMarkerCol,markersize=itemMarkerSize)
    # plt.xticks(ticks=np.linspace(0,10,5),labels=np.arange(1,6),fontsize=12)
    # plt.yticks(ticks=np.linspace(0,10,5),labels=np.arange(1,6),fontsize=12)
    plt.xlabel('translation along x-axis')
    plt.ylabel('translation along y-axis')
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.grid()
    b,l = np.meshgrid(np.linspace(0,10,5),np.linspace(0,10,5))
    b = b.flatten()
    l = l.flatten()
    x = b+b_d
    y = l+l_d-10
    l=l-10
    y[y<=0] = 0
    l[l<=0] = 0

    for ii in range(0,n_items):
        
        plt.plot(b[ii],l[ii],marker='s',markerfacecolor=ctxMarkerCol,markeredgecolor=ctxMarkerEdgeCol,markersize=ctxMarkerSize,markeredgewidth=2)
        plt.plot(x[ii],y[ii],marker='o',markeredgecolor=(.4,.4,.4),markerfacecolor=(.4,.4,.4),markersize=itemMarkerSize*1.8)
        plt.plot(x[ii],y[ii],marker='o',markeredgecolor=itemMarkerCol,markerfacecolor=itemMarkerCol,markersize=itemMarkerSize)
    # plt.xticks(ticks=np.linspace(0,10,5),labels=np.arange(1,6),fontsize=12)
    # plt.yticks(ticks=np.linspace(0,10,5),labels=np.arange(1,6),fontsize=12)
    plt.xlabel('translation along x-axis')
    plt.ylabel('translation along y-axis')
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.grid()

# def plot_mds(xyz,ndims=2,fig_id=1):

#     if ndims==3:
#         fig = plt.figure(fig_id)
#         ax = fig.add_subplot(111,projection='3d')
#         plot_grid3(data[0:25,:],line_colour=(0, 0, .5),fig_id=fig_id)
#         plot_grid3(data[25:,:],line_colour=(255/255,102/255,0),fig_id=fig_id)
#         scatter_mds_3(data,fig_id=2,task_id='both')
        
#     elif ndims==2:
#         fig = plt.figure(fig_id)
#         plot_grid2(data[0:25,:],line_colour=(0, 0, .5),fig_id=fig_id)
#         plot_grid2(data[25:,:],line_colour=(255/255,102/255,0),fig_id=fig_id)
#         scatter_mds_2(data,fig_id=2,task_id='both')
#     ax = plt.gca()
#     ax.axes.xaxis.set_ticklabels([])
#     ax.axes.yaxis.set_ticklabels([]) 
#     plt.show()





def plot_lcurves(results,runlabel='scale_id',runvalue=np.arange(1,11)):

    
    fig=plt.figure(figsize=(20, 10), dpi= 80, facecolor='w', edgecolor='k')
    lines = []
    labels = [runlabel + ' = ' + str(ii) for ii in runvalue]
    for ii in range(len(runvalue)):
        plt.subplot(2,2,1)
        plt.plot(np.log(np.mean(results['w_relchange_hxs'][ii,:,:],0)))
        plt.title('log relchange \n w_stim')
        plt.xlabel('iter')
        plt.ylabel('log((||wt||-||w0||)/||w0||)')

        plt.subplot(2,2,2)
        plt.plot(np.log(np.mean(results['w_relchange_hxc'][ii,:,:],0)))
        plt.title('log relchange \n w_context')
        plt.xlabel('iter')
        plt.ylabel('log((||wt||-||w0||)/||w0||)')

        plt.subplot(2,2,3)
        plt.plot(np.log(np.mean(results['w_relchange_yh'][ii,:,:],0)))
        plt.title('log relchange \n w_out')
        plt.xlabel('iter')
        plt.ylabel('log((||wt||-||w0||)/||w0||)')

        plt.subplot(2,2,4)
        line, = plt.plot(np.mean(results['all_losses'][ii,:,:],0))
        lines.append(line)
        plt.title('loss')
        plt.xlabel('iter')
        plt.ylabel('loss')
        plt.tight_layout()
    for ii in range(1,5):
        plt.subplot(2,2,ii)
        plt.legend(lines,labels)


def plot_rdms_mds(results,n_dims=3,runlabel='scale_id',runvalue=np.arange(1,11)):
    # %matplotlib inline
    fig=plt.figure(figsize=(15, 25), dpi= 80, facecolor='w', edgecolor='k')
    n_rows = results['all_x_hidden'].shape[0]
    col_idx = 1
    for ii in range(n_rows):
        plt.subplot(n_rows,5,col_idx+0)
        rdm = squareform(pdist(np.mean(results['all_x_hidden'][ii,:,:,:],0).T,'euclidean'))
        plt.imshow(rdm)
        plt.title(runlabel + ' =' + str(runvalue[ii]) + ' | hidden layer (linearity)')

        plt.subplot(n_rows,5,col_idx+1)
        rdm = squareform(pdist(np.mean(results['all_y_hidden'][ii,:,:,:],0).T,'euclidean'))
        plt.imshow(rdm)
        plt.title(runlabel + ' =' + str(runvalue[ii]) + ' | hidden layer (relu)')

        plt.subplot(n_rows,5,col_idx+2)
        rdm = squareform(pdist(np.mean(results['all_y_out'][ii,:,:,:],0).T))
        plt.imshow(rdm)
        plt.title(runlabel + ' =' + str(runvalue[ii]) + ' | output')

        if n_dims == 3:
            embedding = MDS(n_components=3)
            xyz = embedding.fit_transform(np.mean(results['all_x_hidden'][ii,:,:,:],0).T)
            ax = fig.add_subplot(n_rows,5,col_idx+3, projection='3d')
            ax.scatter(xyz[0:25,0],xyz[0:25,1],xyz[0:25,2],marker='o')
            ax.scatter(xyz[25:,0],xyz[25:,1],xyz[25:,2],marker='^')    
            plt.title('hidden layer (linearity)')

            embedding = MDS(n_components=3)
            xyz = embedding.fit_transform(np.mean(results['all_y_hidden'][ii,:,:,:],0).T)
            ax = fig.add_subplot(n_rows,5,col_idx+4, projection='3d')
            ax.scatter(xyz[0:25,0],xyz[0:25,1],xyz[0:25,2],marker='o')
            ax.scatter(xyz[25:,0],xyz[25:,1],xyz[25:,2],marker='^')    
            plt.title('hidden layer (relu)')
        elif n_dims == 2:
            embedding = MDS(n_components=2)
            xy = embedding.fit_transform(squareform(pdist(np.mean(results['all_x_hidden'][ii,:,:,:],0).T,'euclidean')))
            plt.subplot(n_rows,5,col_idx+3)
            plt.scatter(xy[0:25,0],xy[0:25,1],marker='o')
            plt.scatter(xy[25:,0],xy[25:,1],marker='^')    
            plt.title('hidden layer (linearity)')
            embedding = MDS(n_components=2)
            xy = embedding.fit_transform(squareform(pdist(np.mean(results['all_y_hidden'][ii,:,:,:],0).T,'euclidean')))
            plt.subplot(n_rows,5,col_idx+4)
            plt.scatter(xy[0:25,0],xy[0:25,1],marker='o')
            plt.scatter(xy[25:,0],xy[25:,1],marker='^')    
            plt.title('hidden layer (relu)')

        plt.tight_layout()
        col_idx +=5



def plot_losses(results,y,runlabel='scale_id',runvalue=np.arange(1,11)):
    n_factors = results['all_y_out'].shape[0]
    n_runs = results['all_y_out'].shape[1]
    # plot learning curves (loss)
    l_mu = np.mean(results['all_losses'][:,:,-1],1)
    l_err = np.std(results['all_losses'][:,:,-1],1)/np.sqrt(n_runs)
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.errorbar(np.arange(1,n_factors+1),l_mu,yerr=l_err)
    plt.title('endpoint training loss')
    plt.xlabel(runlabel)
    plt.ylabel('loss')
    # plot endpoint accuracy 
    plt.subplot(1,2,2)
    y_ = (results['all_y_out']>0.5).astype('int')
    y = y>0.5
    acc = np.empty((n_factors,n_runs))
    for ii in range(n_factors):
        for jj in range(n_runs):
            acc[ii,jj] = np.mean(y_[ii,jj,0,:]==y)
    acc_mu = np.mean(acc,1)
    acc_err = np.std(acc,1)/np.sqrt(n_runs)
    plt.errorbar(np.arange(1,n_factors+1),acc_mu,yerr=acc_err)
    plt.title('endpoint accuracy')
    plt.xlabel(runlabel)
    plt.ylabel('accuracy')
    plt.ylim((.5,1))


def plot_ctxcorrelation(results,runlabel='scale_id',runvalue=np.arange(1,11),coeffs=[]):
       
    if not len(coeffs):
        n_factors = results['all_w_hxc'].shape[0]
        n_runs = results['all_w_hxc'].shape[1]
        coeffs = np.empty((n_factors,n_runs))
        for ii in range(n_factors):
            for jj in range(n_runs):
                c = np.corrcoef(results['all_w_hxc'][ii,jj,:,0],results['all_w_hxc'][ii,jj,:,1])
                coeffs[ii,jj] = c[0,1]

    coeffs_mu = np.mean(coeffs,1)
    coeffs_err = np.std(coeffs,1)/np.sqrt(coeffs.shape[1])
    print(coeffs.shape)
    print(coeffs.shape[1])
    print(coeffs.shape[0])
    print(coeffs_mu.shape)
    print(coeffs_err.shape)
    plt.figure()
    plt.errorbar(np.arange(1,coeffs.shape[0]+1),coeffs_mu,yerr=coeffs_err)
    plt.title('correlation between context weights')
    plt.xlabel(runlabel)
    plt.ylabel('pearson correlation')
    plt.ylim((-1,1))



def plot_rsa_fits(c_mu,c_err,runvalue=np.arange(1,11),runlabel='init weight variance',titlestr='Model Fits'):
    # plt.errorbar(runvalue,c_mu[:,0],yerr=c_err[:,0],color=(87/255, 66/255, 245/255),marker='o',linestyle='-')
    plt.errorbar(runvalue,c_mu[:,0],yerr=c_err[:,0],color=(39/255, 140/255, 145/255),marker='o',linestyle='-',linewidth=.5,markersize=2)
    
    plt.errorbar(runvalue,c_mu[:,1],yerr=c_err[:,1],color=(34/255, 76/255, 128/255),marker='o',linestyle='-',linewidth=.5,markersize=2)
    plt.errorbar(runvalue,c_mu[:,2],yerr=c_err[:,2],color=(159/255, 45/255, 235/255),marker='o',linestyle='-',linewidth=.5,markersize=2)
    plt.xticks(ticks=np.arange(0,4,.5),labels=np.arange(0,4,.5))
    plt.legend(['grid model','orthogonal model','parallel model'],fontsize=5,frameon=False)
    plt.xlabel(runlabel,fontsize=6)
    plt.ylabel(r'$\beta$ estimate (a.u.)',fontsize=6)
    plt.title(titlestr,fontsize=6)    
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.ylim((-0.1,0.8))
    plt.xlim((-0.1,3.1))
    