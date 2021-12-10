import torch 

def get_device(use_cuda):
    if use_cuda and torch.cuda.is_available():    
        cuda_device = torch.device('cuda')
        cuda_properties = torch.cuda.get_device_properties(cuda_device)
    else:
        cuda_device = torch.device('cpu')
        cuda_properties = []
    return cuda_device, cuda_properties


def from_gpu(data):
    '''
    transfers data from gpu back to cpu and 
    converts into Numpy format
    '''

    return data.cpu().detach().numpy()