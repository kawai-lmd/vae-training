# Load saved VAE model
from keras.models import load_model

vae = load_model('models/vae_model.h5')
encoder = load_model('models/encoder_model.h5')
decoder = load_model('models/decoder_model.h5')

