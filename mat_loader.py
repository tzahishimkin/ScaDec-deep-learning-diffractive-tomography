import scipy.io as spio
import numpy as np
import os
import requests
from pydrive.auth import GoogleAuth


def get_data_from_file(mat_file):
    data_mat = spio.loadmat(mat_file, squeeze_me=True)
    data = np.array(data_mat['sparse'])
    label = np.array(data_mat['label'])

    swap_channels = 1
    if swap_channels:
        data = np.swapaxes(np.swapaxes(data, 0, 2), 1, 2)
        label = np.swapaxes(np.swapaxes(label, 0, 2), 1, 2)

    expand_dim = 1
    if expand_dim:
        data = np.expand_dims(data, 3)
        label = np.expand_dims(label, 3)

    return data, label


#taken from this StackOverflow answer: https://stackoverflow.com/a/39225039
import requests

def download_file_from_google_drive(url, destination):
    if os.path.exists(destination):
        print("File name "+destination+" exists. Skipping download")
        return

    URL = "https://docs.google.com/uc?export=download"
    id = url.split("id=")[1].split('&')[0]
    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)
    print("Downloaded " + destination + " From URL")

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def download_elipse_dataset(train_path,test_path):
    # Load Train Data:
    destination = train_path
    url = 'https://drive.google.com/uc?id=1FTOgM2vOQaGSokEDtOaPNdBTto6h5yFi&export=download'
    download_file_from_google_drive(url, destination)

    # Load Test Data:
    destination = test_path
    url = 'https://drive.google.com/uc?id=1w_kPao6L2UwhTKIgcr_3o62A6vYYtX_r&export=download'
    download_file_from_google_drive(url, destination)


if __name__=="__main__":

    load_elips_data = True
    if load_elips_data:
        train_path = 'data/train_elips.mat'
        test_path = 'data/test_elips.mat'
        download_elipse_dataset(train_path, test_path)
