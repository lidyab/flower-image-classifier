import streamlit as st
from PIL import Image
import json
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms

from process_image import process_image
import save_load_checkpoint
import fulyConnected_network
import predict

st.title('Flower Image Classifier')

uploaded_img = st.file_uploader('Choose an Image', type=['jpg', 'jpeg', 'png', 'webp'])

if uploaded_img is not None:
    img = Image.open(uploaded_img)
    st.image(img)

    col1, col2, col3 = st.columns(3)
    with col2:

        bt = st.button('Predict')
    if bt:
        st.markdown(f"### The top 5 predicted flower classes are...")

        checkp = save_load_checkpoint.load_checkpoint('checkpoint2.pth')
        model_state_dict = checkp['model_state_dict']
        n_hidden = checkp['hidden_layers']
        n_output = checkp['out_size']
        
        model = fulyConnected_network.Classifier(n_hidden, n_output)
        model.load_state_dict(model_state_dict)
        ps_5 , top_p, top_class = predict.predict(uploaded_img, model, topk=5)

        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)

        top_class_name = []
        for top_c in top_class:
            top_class_name.append(cat_to_name[str(top_c)])

        pil_img = Image.open(uploaded_img)

        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (10,4))
        ax1.imshow(pil_img)
        ax1.set_title('The Flower')
        # ax2.barh(np.arange(len(ps_5)), ps_5)
        ax2.barh(top_class_name, top_p)
        ax2.set_title('Prediction: Top 5 Classes')
        plt.tight_layout()

        st.pyplot(fig)