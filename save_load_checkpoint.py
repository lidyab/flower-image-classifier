import torch

def save_checkpoint(epochs, model, optimizer, train_loss, valid_loss, to_path = 'checkpoint.pth'):
    
    checkpoint = {'in_size': model.hidden[0].in_features,
                'out_size': model.output.out_features,
                'hidden_layers': [each_hid.out_features for each_hid in model.hidden],
                'Epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': {'train_loss': train_loss, 'valid_loss': valid_loss},
                }

    torch.save(checkpoint, to_path)

def load_checkpoint(from_path ='checkpoint.pth'):
    checkpoint = torch.load(from_path, weights_only=True)
    
    return checkpoint