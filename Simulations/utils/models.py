import numpy as np
from scipy.linalg import expm
from numpy.linalg import norm
import torch
import torch.nn.functional as F
from torchvision import models
from torch import nn


class CNN(nn.Module):
    """same model as in Flesch et al, 2018, but without Dropout

    """
    def __init__(self,weightvars=None):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)               
        self.fc1 = nn.Linear(4608, 1024)        
        self.fc2 = nn.Linear(1024, 512)
        self.o = nn.Linear(512,1)

        if weightvars != None:
            # custom init 
            nn.init.normal_(self.conv1.weight,mean=0,std=weightvars[0]) 
            nn.init.normal_(self.conv2.weight,mean=0,std=weightvars[1]) 
            nn.init.normal_(self.conv3.weight,mean=0,std=weightvars[2])
            nn.init.normal_(self.fc1.weight,mean=0,std=weightvars[3]) 
            nn.init.normal_(self.fc2.weight,mean=0,std=weightvars[4])  
            nn.init.normal_(self.o.weight,mean=0,std=weightvars[5])  

    def forward(self, x):
        # x = x.view(-1, 3, 96, 96)
        self.c1 = F.relu(self.conv1(x))        
        self.c2 = F.relu(self.conv2(self.c1))        
        self.c3 = F.relu(self.conv3(self.c2))                  
        self.flatten = self.c3.reshape(self.c3.shape[0],-1)
        self.f1 = F.relu(self.fc1(self.flatten))
        self.f2 = F.relu(self.fc2(self.f1))        
        self.out = self.o(self.f2)
        return self.out

    def get_layers(self):
        layers = [l for l in self.children()]
        return layers

    def get_activations(self,x):
        layers = self.get_layers()
        activations = []
        for l in layers[:-3]:
            x = l(x)
            activations.append(x)
        x = torch.flatten(x,1)
        for l in layers[-3:]:
            x = l(x)
            activations.append(x)
        
        return activations



class SmallCNN(nn.Module):
    """similar to CNN above, but with only single FC layer
    """
    def __init__(self,weightvars=None):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)              
        self.fc1 = nn.Linear(4608, 512)                
        self.o = nn.Linear(512,1)

        if weightvars != None:
            # custom init 
            nn.init.normal_(self.conv1.weight,mean=0,std=weightvars[0]) 
            nn.init.normal_(self.conv2.weight,mean=0,std=weightvars[1]) 
            nn.init.normal_(self.conv3.weight,mean=0,std=weightvars[2])
            nn.init.normal_(self.fc1.weight,mean=0,std=weightvars[3])             
            nn.init.normal_(self.o.weight,mean=0,std=weightvars[4])  

    def forward(self, x):        
        self.c1 = F.relu(self.conv1(x))        
        self.c2 = F.relu(self.conv2(self.c1))        
        self.c3 = F.relu(self.conv3(self.c2))                  
        self.flatten = self.c3.reshape(self.c3.shape[0],-1)
        self.f1 = F.relu(self.fc1(self.flatten))        
        self.out = self.o(self.f1)
        return self.out

    def get_layers(self):
        layers = [l for l in self.children()]
        return layers

    def get_activations(self,x):
        layers = self.get_layers()
        activations = []
        for l in layers[:-2]:
            x = l(x)
            activations.append(x)
        x = torch.flatten(x,1)
        for l in layers[-2:]:
            x = l(x)
            activations.append(x)
        return activations

    

class CustomAlexNet(nn.Module):
    """conv layers of (pretrained) AlexNet, with two FC layers on top
    """
    def __init__(self,pretrained=True):
        super().__init__()
        
        alexnet = models.alexnet(pretrained=pretrained,progress=True)
        self.features = nn.Sequential(*list(alexnet.features)[:-1])
        
        self.fc1 = nn.Linear(6400,512)
        self.o = nn.Linear(512,1)

        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x,1)        
        x = F.relu(self.fc1(x))
        self.out = self.o(x)
        return self.out

    def get_layers(self):
        layers = [l for l in self.features.children()]
        layers += [self.fc1] 
        layers += [self.o]
        return layers

    def get_activations(self,x):
        layers = self.get_layers()
        activations = []
        for l in layers[:-2]:
            x = l(x)
            activations.append(x)
        x = torch.flatten(x,1)
        for l in layers[-2:]:
            x = l(x)
            activations.append(x)
            
        return activations



class NTK():
    ''' implements a Neural Tangent Kernel (Jacot et al, 2018) 
        as analytic solution to the learning dynamics in the large weight 
        limit 

        INPUTS:
        n_in = number of input units 
        n_ctx = number of context units 
        n_hidden = number of hidden units 
        n_out = number of output units 
        lrate = SGD learning rate 
        scale_whxs = weight scale for input-to-hidden weights 
        scale_whxc = weight scale for context-to-hidden weights 
        scale_wyh = weight scale for output weights 
    '''

    def __init__(self, x_stim, x_ctx, y, n_in=25, n_ctx=2, n_hidden=100, n_out=1, lrate=5e3,  scale_whxs=1e-4, scale_whxc=0.25, scale_wyh=1/100):
        # set parameters
        self.n_in = n_in
        self.n_ctx = n_ctx
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.eta = lrate

        # initialise weights
        self.w_hxs = scale_whxs*np.random.randn(self.n_hidden, self.n_in)
        self.w_hxc = scale_whxc*np.random.randn(self.n_hidden, self.n_ctx)
        self.w_yh = scale_wyh*np.random.randn(self.n_out, self.n_hidden)
        self.b_yh = np.repeat(0.1, n_out)
        self.b_hx = np.repeat(0.1, n_hidden)

        # set dataset
        self.x_stim = x_stim
        self.x_ctx = x_ctx
        self.y = y

        # initialise dgradient vect
        self.all_gradients = np.empty(
            (self.n_hidden*(self.n_in+self.n_ctx+1)+self.n_out*(self.n_hidden+1), 50))

        # get gradients
        self.collect_gradients()

    def fprop(self, x_stim, x_ctx, y):
        ''' forward pass through network
        '''
        self.h_in = self.w_hxs.dot(
            x_stim) + self.w_hxc.dot(x_ctx) + self.b_hx[:, np.newaxis]
        self.h_out = self.relu(self.h_in)
        self.y_ = self.w_yh.dot(self.h_out)+self.b_yh
        self.l = self.loss(y, self.y_)

    def collect_gradients(self):
        ''' collects and vectorises all gradients for NTK computation
        '''

        for ii in range(self.x_stim.shape[1]):
            xsi = self.x_stim[:, ii].reshape(25, 1)
            xci = self.x_ctx[:, ii].reshape(2, 1)
            yi = self.y[0][ii].reshape(1, 1)
            # fprop
            self.fprop(xsi, xci, yi)

            # partial derivatives
            dl_dy = self.deriv_loss(self.y_, yi)
            dy_dh = self.w_yh
            dy_dw = self.h_out
            dho_dhi = self.deriv_relu(self.h_in)
            dhi_dws = xsi
            dhi_dwc = xci
            # individual gradents
            self.dl_dwyh = dl_dy.dot(dy_dw.T)
            self.dl_dbyh = dl_dy

            self.dl_dwhxs = (dl_dy.T.dot(dy_dh)*dho_dhi.T).T.dot(dhi_dws.T)
            self.dl_dwhxc = (dl_dy.T.dot(dy_dh)*dho_dhi.T).T.dot(dhi_dwc.T)
            self.dl_dbhx = (dl_dy.T.dot(dy_dh)*dho_dhi.T).T

            grad_vect = np.concatenate((self.dl_dwyh.flatten(), self.dl_dbyh.flatten(
            ), self.dl_dwhxs.flatten(), self.dl_dwhxc.flatten(), self.dl_dbhx.flatten()))
            self.all_gradients[:, ii] = grad_vect

    def H(self):
        ''' dot product of gradients '''
        return np.dot(self.all_gradients.T, self.all_gradients)

    def get_dynamics(self, t_end=0.5, n_steps=500):
        ''' evaluates NTK at n_steps training steps 
            and returns resulting training dynamics
        '''
        self.fprop(self.x_stim, self.x_ctx, self.y)
        u0 = self.y_ - self.y
        # evaluate the loss for each time point
        ts = np.linspace(0, t_end, num=n_steps)
        yh = np.empty((n_steps, self.x_stim.shape[1]))
        self.l = np.empty((n_steps))
        for ii in range(n_steps):
            t = ts[ii]
            u = np.dot(u0, expm(-self.H()*t))

            yh[ii, :] = self.y-u
            self.l[ii] = self.loss(self.y, yh[ii, :])
        return self.l

    def relu(self, x):
        return x*(x > 0)

    def deriv_relu(self, x):
        return (x > 0).astype('double')

    def loss(self, y_, y):
        return .5*norm(y_-y, 2)**2

    def deriv_loss(self, y_, y):
        return (y_-y)


class MLP():
    ''' implements simple feedforward MLP with a single hidden layer 
        of relu nonlinearities. trained with SGD on MSE loss 
        INPUTS:
        n_in = number of input units 
        n_ctx = number of context units 
        n_hidden = number of hidden units 
        n_out = number of output units 
        lrate = SGD learning rate 
        scale_whxs = weight scale for input-to-hidden weights 
        scale_whxc = weight scale for context-to-hidden weights 
        scale_wyh = weight scale for output weights 
    '''

    def __init__(self, n_in, n_ctx, n_hidden, n_out, lrate,  scale_whxs, scale_whxc, scale_wyh):
        # set parameters
        self.n_in = n_in
        self.n_ctx = n_ctx
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.eta = lrate

        # initialise weights
        self.w_hxs = scale_whxs*np.random.randn(self.n_hidden, self.n_in)
        self.w_hxc = scale_whxc*np.random.randn(self.n_hidden, self.n_ctx)
        self.w_yh = scale_wyh*np.random.randn(self.n_out, self.n_hidden)
        self.b_yh = np.repeat(0.1, n_out)
        self.b_hx = np.repeat(0.1, n_hidden)

    def fprop(self, x_stim, x_ctx, y):
        self.h_in = self.w_hxs.dot(
            x_stim) + self.w_hxc.dot(x_ctx) + self.b_hx[:, np.newaxis]
        self.h_out = self.relu(self.h_in)
        self.y_ = self.w_yh.dot(self.h_out)+self.b_yh
        self.l = self.loss(y, self.y_)

    def bprop(self, x_stim, x_ctx, y):
        # partial derivatives
        dl_dy = self.deriv_loss(self.y_, y)
        dy_dh = self.w_yh
        dy_dw = self.h_out
        dho_dhi = self.deriv_relu(self.h_in)
        dhi_dws = x_stim
        dhi_dwc = x_ctx
        # backward pass:
        self.dl_dwyh = dl_dy.dot(dy_dw.T)
        self.dl_dbyh = dl_dy

        self.dl_dwhxs = (dl_dy.T.dot(dy_dh)*dho_dhi.T).T.dot(dhi_dws.T)
        self.dl_dwhxc = (dl_dy.T.dot(dy_dh)*dho_dhi.T).T.dot(dhi_dwc.T)
        self.dl_dbhx = (dl_dy.T.dot(dy_dh)*dho_dhi.T).T

    def update(self):
        # weight updates
        self.w_yh = self.w_yh - self.eta*self.dl_dwyh
        self.b_yh = self.b_yh - self.eta*np.sum(self.dl_dbyh, axis=1)

        self.w_hxs = self.w_hxs - self.eta*self.dl_dwhxs
        self.w_hxc = self.w_hxc - self.eta*self.dl_dwhxc
        self.b_hx = self.b_hx - self.eta*np.sum(self.dl_dbhx, axis=1)

    def train(self, x_stim, x_ctx, y):
        self.fprop(x_stim, x_ctx, y)
        self.bprop(x_stim, x_ctx, y)
        self.update()

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def deriv_sigmoid(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def relu(self, x):
        return x*(x > 0)

    def deriv_relu(self, x):
        return (x > 0).astype('double')

    def loss(self, y_, y):
        return .5*norm(y_-y, 2)**2

    def deriv_loss(self, y_, y):
        return (y_-y)


class MLP_L2(MLP):
    ''' augments MLP defined above with L2 regulariser 
        additional inputs:
        lmbd = regularisation strength 
    '''

    def __init__(self, lmbd=1, **kwargs):
        super().__init__(**kwargs)
        self.lmbd = lmbd

    def update(self):
        '''
        L2-regularised optimisation step       
        '''

        self.w_yh = self.w_yh - self.eta*(self.dl_dwyh + self.lmbd*self.w_yh)
        self.b_yh = self.b_yh - self.eta*np.sum(self.dl_dbyh, axis=1)

        self.w_hxs = self.w_hxs - self.eta * \
            (self.dl_dwhxs + self.lmbd*self.w_hxs)
        self.w_hxc = self.w_hxc - self.eta * \
            (self.dl_dwhxc + self.lmbd*self.w_hxc)
        self.b_hx = self.b_hx - self.eta*np.sum(self.dl_dbhx, axis=1)

    def loss(self, y_, y):
        ''' MSE loss with L2 regulariser 
            J(theta) = MSE(y^,y)_theta +L2(theta)
         '''
        return .5*norm(y_-y, 2)**2 + self.lmbd*np.sum(np.concatenate((np.square(self.w_hxs.flatten()), np.square(self.w_hxc.flatten()), np.square(self.w_yh.flatten())), axis=0))


class MLP_rew():
    '''
    implements a simple feedforward neural network with a single hidden layer 
    trained with reward signal
    '''

    def __init__(self, n_in, n_ctx, n_hidden, n_out, lrate,  scale_whxs, scale_whxc, scale_wyh):
        '''
        mlp trained with reward signal (sigmoidal output)
        '''
        # set parameters
        self.n_in = n_in
        self.n_ctx = n_ctx
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.eta = lrate

        # initialise weights
        self.w_hxs = scale_whxs*np.random.randn(self.n_hidden, self.n_in)
        self.w_hxc = scale_whxc*np.random.randn(self.n_hidden, self.n_ctx)
        self.w_yh = scale_wyh*np.random.randn(self.n_out, self.n_hidden)
        self.b_yh = np.repeat(0.1, n_out)
        self.b_hx = np.repeat(0.1, n_hidden)

    def fprop(self, x_stim, x_ctx, y):
        ''' implements a forward pass through the network '''
        self.h_in = self.w_hxs.dot(
            x_stim) + self.w_hxc.dot(x_ctx) + self.b_hx[:, np.newaxis]
        self.h_out = self.relu(self.h_in)
        self.o_in = self.w_yh.dot(self.h_out)+self.b_yh
        self.y_ = self.sigmoid(self.o_in)
        self.l = self.rewloss(y, self.y_)

    def bprop(self, x_stim, x_ctx, y):
        ''' implements a backward pass through the network '''
        # partial derivatives
        dl_dy = self.deriv_rewloss(self.y_, y)
        dy_do = self.deriv_sigmoid(self.o_in)
        dy_dh = self.w_yh
        dy_dw = self.h_out
        dho_dhi = self.deriv_relu(self.h_in)
        dhi_dws = x_stim
        dhi_dwc = x_ctx
        # backward pass:
        self.dl_dwyh = (dl_dy*dy_do).dot(dy_dw.T)
        self.dl_dbyh = (dl_dy*dy_do.T)

        self.dl_dwhxs = ((dl_dy*dy_do).T.dot(dy_dh)*dho_dhi.T).T.dot(dhi_dws.T)
        self.dl_dwhxc = ((dl_dy*dy_do).T.dot(dy_dh)*dho_dhi.T).T.dot(dhi_dwc.T)
        self.dl_dbhx = ((dl_dy*dy_do).T.dot(dy_dh)*dho_dhi.T).T

    def update(self):
        ''' performs weight updates with SGD'''
        # weight updates
        self.w_yh = self.w_yh - self.eta*self.dl_dwyh
        self.b_yh = self.b_yh - self.eta*np.sum(self.dl_dbyh, axis=1)

        self.w_hxs = self.w_hxs - self.eta*self.dl_dwhxs
        self.w_hxc = self.w_hxc - self.eta*self.dl_dwhxc
        self.b_hx = self.b_hx - self.eta*np.sum(self.dl_dbhx, axis=1)

    def train(self, x_stim, x_ctx, y):
        ''' performs a single training step (forward & backward pass, weight updates'''
        self.fprop(x_stim, x_ctx, y)
        self.bprop(x_stim, x_ctx, y)
        self.update()

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def deriv_sigmoid(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def relu(self, x):
        return x*(x > 0)

    def deriv_relu(self, x):
        return (x > 0).astype('double')

    def rewloss(self, y_, y):
        return np.mean(-1*(y_*y))

    def deriv_rewloss(self, y_, y):
        return -1*y


class MLP_L2_rew(MLP_rew):
    ''' 
    same as MLP_rew, but with additional L2 regulariser
    '''

    def __init__(self, *args, lmbd=1):
        super().__init__(*args)
        self.lmbd = lmbd

    def update(self):
        # weight updates
        self.w_yh = self.w_yh - self.eta*(self.dl_dwyh + self.lmbd*self.w_yh)
        self.b_yh = self.b_yh - self.eta*np.sum(self.dl_dbyh, axis=1)

        self.w_hxs = self.w_hxs - self.eta * \
            (self.dl_dwhxs + self.lmbd*self.w_hxs)
        self.w_hxc = self.w_hxc - self.eta * \
            (self.dl_dwhxc + self.lmbd*self.w_hxc)
        self.b_hx = self.b_hx - self.eta*np.sum(self.dl_dbhx, axis=1)

    def rewloss(self, y_, y):
        return np.mean(-1*(y_*y)) + self.lmbd*np.sum(np.concatenate((np.square(self.w_hxs.flatten()), np.square(self.w_hxc.flatten()), np.square(self.w_yh.flatten())), axis=0))
