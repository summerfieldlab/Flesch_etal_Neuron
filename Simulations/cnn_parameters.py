import argparse 


def boolean_string(s):
    '''
    helper function, turns string into boolean variable    
    '''
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'



# parameters 
parser = argparse.ArgumentParser(description='CNN Simulations')


# network parameters 
parser.add_argument('--model',default='CNN',type=str, help='which model (CNN, SmallCNN, Alex, ALexPT')

parser.add_argument('--weight_init',default=None,type=list,help='initial weight scale for each layer')

# optimiser parameters
parser.add_argument('--learning_rate', default=1e-4,type=float, help='learning rate for SGD')


# training parameters 
parser.add_argument('--cuda', default=True, type=boolean_string, help='run model on GPU')
# parser.add_argument('--n_runs', default=30, type=int, help='number of independent training runs')
parser.add_argument('--n_epochs', default=200, type=int, help='number of training epochs')

parser.add_argument('--log-interval',default=5,type=int,help='log very n epochs')
parser.add_argument('--bs_train',default=128,type=int,help='training batch size')
parser.add_argument('--bs_test',default=128,type=int,help='test batch size')

# io params
parser.add_argument('--verbose',default=True, type=boolean_string, help='verbose mode, print all logs to stdout')
parser.add_argument('--save_results',default=True,type=boolean_string,help='save model and results (yes/no)')
parser.add_argument('--save_dir',default='simu1',help='save dir for model outputs')