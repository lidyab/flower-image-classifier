import argparse

def get_input_args():
    """ To retrieves and parses the command line arguments  """
    
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    ## Argument 1: Learning Rate
    parser.add_argument('--lr', type = float, default = 0.001, 
                        help = 'Learning Rate') 

    ## Argument 2: Epoches
    parser.add_argument('--epoch', type = int, default = 15, 
                        help = 'Epoches') 

    ## Argument 3: Hidden Layers
    parser.add_argument('--hid', nargs='+', type = int, default = [512, 256], 
                        help = 'Hidden Layers')    
 
    ## Argument 4: Top predicted classes to display 
    parser.add_argument('--topk', type = int, default = 3, 
                        help = 'Top Classes')    
    
    ## Argument 5: Device
    parser.add_argument('--gpu', type = str, default = 'cpu', 
                        help = 'Device: GPU/CPU')   
    
    ## Argument 6: checkpoint save directory
    parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', 
                        help = 'checkpoint save directory')   
    

    ## Assigns variable in_args to parse_args()
    in_args = parser.parse_args()
    return in_args

