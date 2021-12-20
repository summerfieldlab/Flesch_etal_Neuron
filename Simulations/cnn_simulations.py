import torch
from pathlib import Path
from utils.trainer import train_cnn
from utils.models import SmallCNN, CNN, CustomAlexNet
from cnn_parameters import parser
from utils.nnet import get_device
from utils.data import gen_datasets
import pickle
from joblib import Parallel, delayed

def rich_init(m):
    """applies rich initialisation to cnn
       note: the standard init is already in the rich regime
       and slightly more stable than this.

    Args:
        m (pytorch model layer): instance of neural network layer
    """
    if isinstance(m,torch.nn.Conv2d):
        # print(m.__class__.__name__)
        torch.nn.init.normal_(m.weight,mean=0,std=0.005)
    elif isinstance(m,torch.nn.Linear):
        # print(m.__class__.__name__)
        torch.nn.init.normal_(m.weight,mean=0,std=0.004)
        torch.nn.init.zeros_(m.bias)

def lazy_init(m):
    """applies lazy initialisation to cnn

    Args:
        m (pytorch model layer): instance of model layer
    """
    if isinstance(m,torch.nn.Conv2d):
        # print(m.__class__.__name__)
        torch.nn.init.normal_(m.weight,mean=0,std=0.1)
    elif isinstance(m,torch.nn.Linear):
        # print(m.__class__.__name__)
        torch.nn.init.normal_(m.weight,mean=0,std=0.02)
        torch.nn.init.zeros_(m.bias)



def run_experiment(args, model, data, save_dir, fname_results, fname_model):
    """performs a neural network experiment

    Args:
        args (argparse): network and training parameters
        model (nn.Module): a pytorch neural network
        data (linst): training and test datasets
        save_dir (path): a path to the location of log and model files
        fname_results (str): file name for results dict file
        fname_model (str): file name for model file
    """

    # train model:
    results = train_cnn(args, model, data)

    # save results and model:
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / fname_results, 'wb') as f:
        pickle.dump(results, f)

    with open(save_dir / fname_model, 'wb') as f:
        pickle.dump(model, f)


def execute_run(i_run, args):
    """loads data, instantiates model, sets paths & parameters and finally launches run_experiment

    Args:
        i_run (int): id of training run
        args (argparse): parameters
    """
    print('run {} / {}'.format(str(i_run), str(args.n_runs)))

    run_id = 'run_' + str(i_run)


    # load data:
    dl_train, dl_val, dl_tn, dl_ts = gen_datasets(
        bs_train=args.bs_train, bs_test=args.bs_test, filepath=args.datadir)

    # Pretrained AlexNet:
    # set params
    args.n_epochs = 200
    args.learning_rate = 1e-4

    save_dir = Path("cnn_results") / 'alexnet_pretrained' / run_id
    fname_results = 'results.pkl'
    fname_model = 'model.pkl'
    model = CustomAlexNet(pretrained=True)
    run_experiment(args, model, [dl_train, dl_val, dl_tn, dl_ts], save_dir, fname_results, fname_model)

    # AlexNet from scratch (this is the one with rich init):
    # set params
    args.n_epochs = 500
    args.learning_rate = 1e-4
    save_dir = Path("cnn_results") / 'alexnet_fromscratch' / run_id
    fname_results = 'results.pkl'
    fname_model = 'model.pkl'
    model = CustomAlexNet(pretrained=False)
    run_experiment(args, model, [dl_train, dl_val, dl_tn, dl_ts], save_dir, fname_results, fname_model)


    # AlexNet from scratch, lazy learning
    # set params
    args.n_epochs = 200
    args.learning_rate = 1e-4
    save_dir = Path("cnn_results") / 'alexnet_fromscratch_lazy' / run_id
    fname_results = 'results.pkl'
    fname_model = 'model.pkl'
    nnet = CustomAlexNet(pretrained=False)
    _ = nnet.apply(lazy_init)
    # some further fine-tuning of output layer init:
    torch.nn.init.normal_(nnet.o.weight,mean=0,std=0.001)
    torch.nn.init.zeros_(nnet.o.bias)
    # start the fun:
    run_experiment(args, nnet, [dl_train, dl_val, dl_tn, dl_ts], save_dir, fname_results, fname_model)


if __name__ == "__main__":
    # get parameters
    args = parser.parse_args(args=[])
    args.device, _ = get_device(args.cuda)
    # override a few defaults
    args.datadir = '../../Data/Simulations/'
    args.n_runs = 30


    # launch a few simulations in parallel
    Parallel(n_jobs=4, verbose=10)(delayed(execute_run)(i_run,args) for i_run in range(args.n_runs)) 
        
       
    
