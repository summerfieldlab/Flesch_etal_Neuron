import math
import matplotlib.pyplot as plt 
import numpy as np 

def rotate_axes(x,y,theta):
    # theta is in degrees
    theta_rad = theta * (math.pi/180)  # convert to radians
    x_new = x * math.cos(theta_rad) + y * math.sin(theta_rad)
    y_new =  -x * math.sin(theta_rad) + y * math.cos(theta_rad)
    return x_new, y_new

def rotate(X, theta, axis='x'):
    '''Rotate multidimensional array `X` `theta` degrees around axis `axis`'''
    theta = theta * (math.pi/180)  # convert to radians
    c, s = np.cos(theta), np.sin(theta)
    if axis == 'x': return np.dot(X, np.array([
        [1.,  0,  0],
        [0 ,  c, -s],
        [0 ,  s,  c]
        ]))
    elif axis == 'y': return np.dot(X, np.array([
        [c,  0,  -s],
        [0,  1,   0],
        [s,  0,   c]
        ]))
    elif axis == 'z': return np.dot(X, np.array([
        [c, -s,  0 ],
        [s,  c,  0 ],
        [0,  0,  1.],
        ]))


def plot_grid2(xy,line_colour='green',line_width=2,fig_id=1):
    # %matplotlib qt    
    x,y = np.meshgrid(np.arange(0,5),np.arange(0,5))
    x = x.flatten()
    y = y.flatten()    
    try: xy 
    except NameError: xy = np.stack((x,y),axis=1)
    bl = np.stack((x,y),axis=1)
    fig = plt.figure(fig_id)
    
    for ii in range(0,4):
        for jj in range(0,4):
            p1 = xy[(bl[:,0]==ii) & (bl[:,1]==jj),:].ravel()
            p2 = xy[(bl[:,0]==ii+1) & (bl[:,1]==jj),:].ravel()        
            plt.plot([p1[0],p2[0]],[p1[1],p2[1]],linewidth=line_width,color=line_colour)
            p2 = xy[(bl[:,0]==ii) & (bl[:,1]==jj+1),:].ravel()
            plt.plot([p1[0],p2[0]],[p1[1],p2[1]],linewidth=line_width,color=line_colour)
    ii = 4
    for jj in range(0,4):
        p1 = xy[(bl[:,0]==ii) & (bl[:,1]==jj),:].ravel()
        p2 = xy[(bl[:,0]==ii) & (bl[:,1]==jj+1),:].ravel()
        plt.plot([p1[0],p2[0]],[p1[1],p2[1]],linewidth=line_width,color=line_colour)

    jj = 4
    for ii in range(0,4):
        p1 = xy[(bl[:,0]==ii) & (bl[:,1]==jj),:].ravel()
        p2 = xy[(bl[:,0]==ii+1) & (bl[:,1]==jj),:].ravel()
        plt.plot([p1[0],p2[0]],[p1[1],p2[1]],linewidth=line_width,color=line_colour)
    ax = plt.gca()
    # ax.axes.xaxis.set_ticklabels([])
    # ax.axes.yaxis.set_ticklabels([])  
    # ax.set_xlim([-2,2])
    # ax.set_ylim([-2,2])
    ax.set_aspect('equal','box')



def scatter_mds_2(xyz,task_id='a',fig_id=1):
    if task_id=='both':
        n_items = 50
        ctxMarkerEdgeCol = [(0,0,.5), "orange"]
    elif task_id=='a':
        n_items = 25
        ctxMarkerEdgeCol = (0,0,.5)
    elif task_id=='b':
        n_items = 25
        ctxMarkerEdgeCol = 'orange'
    elif task_id == 'avg':
        n_items = 25
        ctxMarkerEdgeCol = 'k'

    ctxMarkerCol = 'white'
    ctxMarkerSize = 20
    itemMarkerSize = 2
    scat_b = np.arange(5,15,2)
    # scat_l = np.array([[3,252,82], [3,252,177], [3,240,252], [3,152,252], [3,73,252]])/255
    scat_l = np.array([[63,39,24], [64,82,21], [65,125,18], [66,168,15], [68,255, 10]])/255
    # scat_l = np.array([[0,0,0], [.2,.2,.2],[.4,.4,.4],[.6,.6,.6],[.8,.8,.8]])

    l,b = np.meshgrid(np.arange(0,5),np.arange(0,5))
    b = b.flatten()
    l = l.flatten()   
    x = xyz[:,0]
    y = xyz[:,1]
    
    

    if task_id == 'both':
        b = np.concatenate((b,b),axis=0)
        l = np.concatenate((l,l),axis=0)
        fig = plt.figure(fig_id)
        for ii in range(0,n_items//2):
            
            plt.plot([x[ii],x[ii]],[y[ii],y[ii]],marker='s',markerfacecolor=ctxMarkerCol,markeredgecolor=ctxMarkerEdgeCol[0],markersize=ctxMarkerSize,markeredgewidth=2)
            plt.plot(x[ii],y[ii],marker='h',markeredgecolor=scat_l[l[ii],:],markerfacecolor=scat_l[l[ii],:],markersize=scat_b[b[ii]])
        for ii in range(n_items//2,n_items):
            
            plt.plot(x[ii],y[ii],marker='s',markerfacecolor=ctxMarkerCol,markeredgecolor=ctxMarkerEdgeCol[1],markersize=ctxMarkerSize,markeredgewidth=2)
            plt.plot(x[ii],y[ii],marker='h',markeredgecolor=scat_l[l[ii],:],markerfacecolor=scat_l[l[ii],:],markersize=scat_b[b[ii]])
    else:
        for ii in range(0,n_items):
            
            plt.plot(x[ii],y[ii],marker='s',markerfacecolor=ctxMarkerCol,markeredgecolor=ctxMarkerEdgeCol,markersize=ctxMarkerSize,markeredgewidth=2)
            plt.plot(x[ii],y[ii],marker='h',markeredgecolor=scat_l[l[ii],:],markerfacecolor=scat_l[l[ii],:],markersize=scat_b[b[ii]])


def plot_2d_mds(xyz,label,tx=0,ty=0,tz=0):

    xyz_rot = rotate(xyz,tx,axis='x')
    xyz_rot = rotate(xyz_rot,ty,axis='y')
    xyz_rot = rotate(xyz_rot,tz,axis='z')

    plt.close()
    fig = plt.figure(2,figsize=(9.8, 4.2), dpi= 80, facecolor='w', edgecolor='k')
        
    plt.subplot(1,2,1)
    plot_grid2(xyz_rot[0:25,[0,1]],line_colour=(0, 0, .5),fig_id=2)
    plot_grid2(xyz_rot[25:,[0,1]],line_colour="orange",fig_id=2)
    scatter_mds_2(xyz_rot[:,[0,1]],fig_id=2,task_id='both')
    ax = plt.gca()
    # ax.set_xlim([-3,3])
    # ax.set_ylim([-3,3])    
    xls = plt.xticks()[0]
    plt.xticks(ticks=[xls[0],xls[-1]],fontsize=14)
    yls = plt.yticks()[0]
    plt.yticks(ticks=[yls[0],yls[-1]],fontsize=14)
    # plt.title('Rich Regime',fontsize=14)
    plt.xlabel('Dimension 1',fontsize=14)
    plt.ylabel('Dimension 2',fontsize=14)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.title('INDSCAL for ROI '+ label)

    plt.subplot(1,2,2)
    plot_grid2(xyz_rot[0:25,[1,2]],line_colour=(0, 0, .5),fig_id=2)
    plot_grid2(xyz_rot[25:,[1,2]],line_colour="orange",fig_id=2)
    scatter_mds_2(xyz_rot[:,[1,2]],fig_id=2,task_id='both')
    ax = plt.gca()
    # ax.set_xlim([-3,3])
    # ax.set_ylim([-3,3])
    xls = plt.xticks()[0]
    plt.xticks(ticks=[xls[0],xls[-1]],fontsize=14)
    yls = plt.yticks()[0]
    plt.yticks(ticks=[yls[0],yls[-1]],fontsize=14)
    plt.xlabel('Dimension 2',fontsize=14)
    plt.ylabel('Dimension 3',fontsize=14)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.title('INDSCAL for ROI '+ label)