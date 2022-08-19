# Deep-Learning-on-MPS-Device-Example

def get_default_device():

    """Pick GPU if available, else CPU"""
    
    if (torch.backends.mps.is_available()):
    
        return torch.device('mps')
        
    else:
    
        return torch.device('cpu')
        
        
        
 # Availability  of MPS device in M1 Chip Macbook
 torch.backends.mps.is_available()       
        
