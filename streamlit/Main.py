# Graded Unit project
# Streamlit application that allows user to upload an image and get the bird specie
# Karina Sudnicina
# 02.05.2021

import streamlit as st
from pathlib import Path
from fastai.vision import load_learner, open_image
from PIL import Image

# Load the trained model
model_path = Path(__file__).parents[0] / 'data/pickles'
model_filename = 'trained_model.pkl'
model = load_learner(model_path, model_filename)

# Set a title (WITH EMOJI CODES 😋)
st.title("Bird Classifier :hatched_chick: :bird:")

# Need to use except Exception so it does not catch KeyboardInterrupts and allows to close server
try:
    # Make a button for image uploading, allow only jpg format and add it to sidebar
    image = st.sidebar.file_uploader("Upload an image of a bird...", type="jpg")
    if image: st.image(image)
except Exception:
    # If the image is corrupted, display an error and set image to None
    # The image may be corrupted if someone tries to add .jpg at the end of non-jpg file
    st.error('The uploaded image is corrupted. Try another one!')
    image = None

# If the user has uploaded image, display it with a prediction
# Otherwise, display webapp description
if image:
    # Make a spinner so the user knows that classification is taking place,
    # which usually happens very fast and the spinner is barely noticeable,
    # but this will keep the user updated if the web app lags
    with st.spinner('Classifying...'):
        # Open the image in fast.ai readable form
        img = open_image(image)
        # Make a prediction and extract it
        prediction = model.predict(img)[0]
    # Return the prediction to the user
    st.write("This looks like a... **{}**".format(prediction))
else:
    description = "This is a bird classifier that will tell you what type " \
                  "of a bird you saw! 🦉  \nPlease use the sidebar to " \
                  "upload an image and bear in mind that it will only accept " \
                  ".jpg format! ✔  \nIf the image uploaded " \
                  "does not actually have a bird, the computer will try hard " \
                  "and still find a bird that is most similar to whatever " \
                  "there is on a picture. 🦆  \nYou can also use this app " \
                  "to find which bird specie is similar to you or your friends! 😄"
    st.write(description)
