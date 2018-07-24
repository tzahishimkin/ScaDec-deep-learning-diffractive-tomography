import scipy.io as spio
import numpy as np
import os
import requests
from pydrive.auth import GoogleAuth


class mat_loader():
    def __init__(self):
        pass

    def get_data(mat_file):
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

    def download_tracking_file_by_id(file_id, download_dir):
        assert 0, "This module currently doesn't work. Need to add clinet_sercet.json. Hint: https://github.com/youtube/api-samples/blob/master/go/client_secrets.json.sample"
        gauth = GoogleAuth(settings_file='../settings.yaml')
        # Try to load saved client credentials
        gauth.LoadCredentialsFile("../credentials.json")
        if gauth.credentials is None:
            # Authenticate if they're not there
            gauth.LocalWebserverAuth()
        elif gauth.access_token_expired:
            # Refresh them if expired
            gauth.Refresh()
        else:
            # Initialize the saved creds
            gauth.Authorize()
        # Save the current credentials to a file
        gauth.SaveCredentialsFile("../credentials.json")

        drive = GoogleDrive(gauth)

        logger.debug("Trying to download file_id " + str(file_id))
        file6 = drive.CreateFile({'id': file_id})
        file6.GetContentFile(download_dir + 'mapmob.zip')
        zipfile.ZipFile(download_dir + 'test.zip').extractall(UNZIP_DIR)
        tracking_data_location = download_dir + 'test.json'
        return tracking_data_location

    def load_from_net(DS):

        if DS == 'elips':
            """
            Load from:
            Train - https://drive.google.com/uc?id=1FTOgM2vOQaGSokEDtOaPNdBTto6h5yFi&export=download
            Test - https://drive.google.com/file/d/1w_kPao6L2UwhTKIgcr_3o62A6vYYtX_r/view
            """
            url = 'https://drive.google.com/uc?id=1FTOgM2vOQaGSokEDtOaPNdBTto6h5yFi&export=download'
            file_id = url.split("id=")[1]
            dat_dir = 'data_temp'
            mat_loader.download_tracking_file_by_id(file_id, dat_dir)

