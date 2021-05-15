from fastapi import FastAPI
from keras.models import load_model
from utility import load_config

app = FastAPI()

config = load_config("config.yml")
print(config['train']['lr'])
# model = load_model()