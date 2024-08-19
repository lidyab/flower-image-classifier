import torch
import json

import save_load_checkpoint
import fulyConnected_network
from get_input_args import get_input_args
from process_image import process_image

def predict(image_path, model, topk=1):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    model.eval()    # Change to inferencing mode

    image = process_image(image_path).unsqueeze(0)   #Preprocess image
    
    log_ps = model(image)
    ps = torch.exp(log_ps)
    top_p, top_class = ps.topk(topk, dim=1)
    ps_5 = ps[0].tolist()[:topk]

    model.train()    # Change back to training mode

    return ps_5, top_p[0].tolist(), top_class[0].tolist()


def main():
    in_arg = get_input_args()
    topk = in_arg.topk

    img_path = input('Enter image path: ')
    checkpoint_path = input('Enter checkpoint path: ')

    checkp = save_load_checkpoint.load_checkpoint(checkpoint_path)
    model_state_dict = checkp['model_state_dict']
    n_hidden = checkp['hidden_layers']
    n_output = checkp['out_size']
    
    print(f'\nThe top {topk} predicted flower classes are:...')
    model = fulyConnected_network.Classifier(n_hidden, n_output)
    model.load_state_dict(model_state_dict)
    ps_5 , top_p, top_class = predict(img_path, model, topk)

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    top_class_name = []
    for top_c in top_class:
        top_class_name.append(cat_to_name[str(top_c)])
 
    for idx in range(len(top_class_name)):
        print(f"{idx+1}: Flower class name: {top_class_name[idx]}.. probablity: {top_p[idx]:.2f}")

# Call to main function to run the program
if __name__ == "__main__":
    main()