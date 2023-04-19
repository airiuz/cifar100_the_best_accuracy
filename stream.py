import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from random import randint
from glob import glob

def get_images(clas):
    return [cv2.imread(i) for i in glob('streamlit_samples/'+clas+'*')]

cifar100 = load_model('models/cifar100.h5')
st.markdown('<style>body{color: White; background-color: DarkSlateGrey}</style>', unsafe_allow_html=True)

##############################
######   for CIFAR100  #######
##############################

st.title('CIFAR100 Recognizer')
classes = [
       'apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle','bowl','boy','bridge',\
       'bus','butterfly','camel','can','castle','caterpillar','cattle','chair','chimpanzee','clock','cloud',\
       'cockroach','couch','cra','crocodile','cup','dinosaur','dolphin','elephant','flatfish','forest','fox','girl',\
       'hamster','house','kangaroo','keyboard','lamp','lawn_mower','leopard','lion','lizard','lobster','man',\
       'maple_tree','motorcycle','mountain','mouse','mushroom','oak_tree','orange','orchid','otter','palm_tree',\
       'pear','pickup_truck','pine_tree','plain','plate','poppy','porcupine','possum','rabbit','raccoon','ray','road',\
       'rocket','rose','sea','seal','shark','shrew','skunk','skyscraper','snail','snake','spider','squirrel',\
       'streetcar','sunflower','sweet_pepper','table','tank','telephone','television','tiger','tractor','train',\
       'trout','tulip','turtle','wardrobe','whale','willow_tree','wolf','woman','worm',
]
st.header(":white[Sample images for classes]")

clas = st.radio(
"Choose class",
classes, horizontal=True)
images = get_images(clas)
rand = randint(0, 9)
a = cv2.resize(images[rand], (112,112), interpolation = cv2.INTER_AREA)
st.image(a)

st.header(":blue[Using model]")

img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    img_array = np.array(image)


if st.button('Predict'):
    try:
        img_array = cv2.resize(img_array.astype('uint8'), (32, 32))
        img_array = np.expand_dims(img_array, axis=1)
        img_array = img_array.transpose((1,0,2,3))
        val = cifar100.predict(img_array)
        st.write(f'result: {classes[np.argmax(val[0])]}')
        st.bar_chart(val[0])
    except:
        pass