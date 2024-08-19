import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from dataloader import dataloader
from get_input_args import get_input_args
import fulyConnected_network
import save_load_checkpoint

def valid(model, validloader, criterion, device):
    # VALIDATION 

    # Changing to validation mode
    model.eval()

    with torch.no_grad():
        tot_valid_loss = 0
        tot_accuracy = 0
        
        for images, labels in validloader:
            images, labels = images.to(device), labels.to(device)
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            tot_valid_loss += loss.item()
    
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            tot_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
        # else:
        valid_loss = tot_valid_loss / len(validloader)
        accuracy = tot_accuracy / len(validloader)
        
    # Changeing to training mode
    model.train()
    return valid_loss, accuracy

def train(data_dir, model, optimizer, epochs, save_to_path, device):

    validloader = dataloader(data_dir, False)
    trainloader = dataloader(data_dir)
    
    criterion = nn.NLLLoss()
    model = model.to(device)

    train_losses, valid_losses = [], []
    for e in range(epochs):
        tot_train_loss = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            
            log_ps = model(images)
            loss = criterion(log_ps, labels)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            tot_train_loss += loss.item()

        train_loss = tot_train_loss / len(trainloader)
        valid_loss, accuracy = valid(model, validloader, criterion, device)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print('Epoch: {}/{}'.format(e+1, epochs),
                'Training Loss: {:.3f}..'.format(train_loss),
                'Validation Loss: {:.3f}..'.format(valid_loss),
                'Accuracy: {:.3f}..'.format(accuracy))

    #Saving Checkpoint
    save_load_checkpoint.save_checkpoint(epochs, model, optimizer, train_losses, valid_losses, save_to_path)


def main():
    in_arg = get_input_args()
    gpu = in_arg.gpu
    save_dir = in_arg.save_dir
    n_hidden = in_arg.hid
    lr = in_arg.lr

    device = torch.device(gpu)

    dat_dir = input('Enter data directory: ')

    n_output = 102
    model = fulyConnected_network.Classifier(n_hidden, n_output, device=device)
    optimizer = optim.Adam(model.parameters(), lr)

    print("\nEnter [0] to TRAIN NEW MODEL; [1] to FURTHER TRAIN a TRAINED MODEL\n")
    training = input('Enter 0 or 1: ')
    try:
        training = int(training)
        if training == 0:
            print("Training UN-TRAINED MODEL from scratch...\n")
            epochs = in_arg.epoch
            train(dat_dir, model, optimizer, epochs, save_dir, device)
        elif training == 1:
            print("Furtheer Training a TRAINED MODEL to improve it's Accuracy...\n")

            checkpoint_path = input('Enter previous checkpoint path: ')

            checkp = save_load_checkpoint.load_checkpoint(checkpoint_path)
            model_state_dict = checkp['model_state_dict']
            optim_state_dict = checkp['optimizer_state_dict']
            n_hidden = checkp['hidden_layers']
            n_output = checkp['out_size']

            model.load_state_dict(model_state_dict)
            optimizer.load_state_dict(optim_state_dict)

            print('\nFrom Previous Training:')
            print('  Training loss: {:.3f}..'.format(checkp['loss']['train_loss'][-1]), 
                'Validation Loss: {:.3f}..'.format(checkp['loss']['valid_loss'][-1]))
            
            epochs = int(input('Enter new Epochs: '))
            print(f'\nFurther training the model with Epochs = {epochs} ...')

            train(dat_dir, model, optimizer, epochs, 'checkpoint2.pth', device)
        else:
            print(f"Error: '{training}' is not a valid number")
    except ValueError:
        print(f"Error: '{training}' is not a number!")

# Call to main function to run the program
if __name__ == "__main__":
    main()