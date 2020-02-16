#!/usr/bin/env python
# coding: utf-8

# # Write Chunked wav2vec encodings to disk

# In[1]:


import os
import glob
import torch
import torch.nn as nn
F = nn.functional
import pandas as pd
import torchaudio
from fairseq.models.wav2vec import Wav2VecModel
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, Dataset
import gc
from tqdm import tqdm

# In[2]:


os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# torch.cuda.current_device()


# In[3]:

audio_length = 60*16000
question = 'p3q6'
train_or_test = 'train'
dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


# In[4]:


cp = torch.load('/home/karmanya/wav2vec_large.pt');
model = Wav2VecModel.build_model(cp['args'], task=None);
model.load_state_dict(cp['model']);
model.to(dev);


# In[5]:


data_files_path = '/media/nas_mount/Sarthak/ijcai_acl/prompt_wise_16k_audios/prompt3/'
data_files_list = glob.glob(data_files_path+'*.wav')
labels_path = f'/media/nas_mount/Sarthak/ques_wise_models/{question}/labels.csv'
test_path = f'/media/nas_mount/Sarthak/ques_wise_models/{question}/test.csv'


# In[6]:


class AudioDataset(Dataset):
    def __init__(self, df, audio_dir, file_col):
        '''
        Passed Dataset a dataframe of the filenames to train/validate on
        Pass one hot encoded numpy array for labels
        Pass a string for the directory of audio files
        '''
        self.items = df
        self.items['path'] = self.items[file_col].apply(lambda x: x.split('.jpeg')[0]) # Label files are *.jpeg
        self.items['path'] = self.items['path'].apply(lambda x: os.path.join(audio_dir, f'{x}.wav'))
        self.audio_dir = audio_dir
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        '''
        Load the audio waveform of idx from dataframe
        returns a touple of torch tensor and label
        '''
        file = self.items['path'].iloc[idx]
        audio, _ = torchaudio.load(file)
        samples = audio.shape[1]
        if audio.shape[0] > 1:
            audio = torch.mean(audio, axis=0).unsqueeze(dim=0)
        # Trim and pad to audio_length seconds
        if samples < audio_length:
            p1d = (audio_length - samples, 0)
            audio = F.pad(audio, p1d, "constant", 0)
        elif samples > audio_length :
            audio = torch.narrow(audio, 1, 0, audio_length)
        return audio, torch.tensor([idx])


# In[7]:


dataset = AudioDataset(pd.read_csv(labels_path), data_files_path, 'name')
# dataset = AudioDataset(pd.read_csv(test_path), data_files_path, 'name')


# In[8]:


train_dl = DataLoader(dataset, batch_size=8, num_workers=12, shuffle=False)
# train_dl = DataLoader(test_dataset, batch_size=8, num_workers=10, shuffle=False)


from pathlib import Path
path = Path(f'/media/nas_mount/Karmanya/wav2vec_whole/{question}/{train_or_test}')
path.mkdir(parents=True, exist_ok=True)

# In[20]:


for xb, yb in tqdm(train_dl):
    xb = xb.view(-1, audio_length).to(dev)
    model.eval()
    with torch.no_grad():
        xb = model.feature_extractor(xb)
        xb = model.feature_aggregator(xb)
        torch.save(xb, path/f'{yb[0].item()}_{yb[-1].item()}.pt')
        del(xb)
        gc.collect()
