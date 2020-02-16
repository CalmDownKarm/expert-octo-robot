import fire
import glob
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def unbatchify(question, train_test):
    batches = glob.glob(f'/media/nas_mount/Karmanya/wav2vec_whole/{question}/{train_test}/*.pt')
    label = 'labels' if train_test=='train' else 'test'
    labels = pd.read_csv(f'/media/nas_mount/Karmanya/wav2vec_whole/{question}/{label}.csv')
    path = Path(f'/media/nas_mount/Karmanya/wav2vec_whole/{question}/unbatched/{train_test}')
    path.mkdir(parents=True, exist_ok=True)
    for batch in tqdm(batches): 
        start, end  = Path(batch).stem.split('_')
        for name, tensor in zip(labels.iloc[int(start): int(end)+1].name.values, torch.chunk(torch.load(batch, map_location='cpu'), dim=0, chunks=8)):
            pd.np.save(path/f'{name}', tensor.cpu())

if __name__=='__main__':
    fire.Fire(unbatchify)
