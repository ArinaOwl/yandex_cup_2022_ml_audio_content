#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import os
import gc
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from tqdm import tqdm
import random
import annoy
from torch.utils.data import Dataset, DataLoader, Sampler


def set_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED) 
    torch.manual_seed(SEED) 
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  
    os.environ['PYTHONHASHSEED'] = str(SEED)


# Data Loader


class TrainDataset(Dataset):
    def __init__(self, train_df, train_features_dir_path, crop_size = 60):
        self.crop_size = crop_size
        self.train_df = train_df
        self.train_df['index'] = train_df.index
        self.train_size = len(self.train_df)
        self.train_features_dir_path = train_features_dir_path
        
    def _pad_item(self, item):
        padding = (item.shape[1] - self.crop_size) // 2
        item = item[:, padding:padding+self.crop_size]
        return np.transpose(item)

    def __getitem__(self, idxs):
        tracks_features = []
        for idx in idxs:
            row = self.train_df.iloc[idx]
            track_features_file_path = row['archive_features_path']
            track_features = np.load(os.path.join(self.train_features_dir_path, track_features_file_path))
            tracks_features.append(self._pad_item(track_features))
        return np.array(tracks_features)

    def __len__(self):
        return self.train_size


class TestDataset(Dataset):
    def __init__(self, test_df, test_features_dir_path, crop_size = 60):
        self.crop_size = crop_size
        self.test_df = test_df
        self.test_size = len(self.test_df)
        self.test_features_dir_path = test_features_dir_path
        
    def _pad_item(self, item):
        padding = (item.shape[1] - self.crop_size) // 2
        item = item[:, padding:padding+self.crop_size]
        return np.transpose(item)

    def __getitem__(self, idx):
        row = self.test_df.iloc[idx]
        track_features_file_path = row['archive_features_path']
        track_features = np.load(os.path.join(self.test_features_dir_path, track_features_file_path))
        track_features = self._pad_item(track_features)
        return row['trackid'], track_features

    def __len__(self):
        return self.test_size


class TrainSampler(Sampler):
    def __init__(self, dataset):
        rng = np.random.default_rng()
        artistid_groups_indices = dataset.train_df.groupby('artistid', group_keys=False).agg(list)['index']
        self.pairs = list()
        for indices in artistid_groups_indices:
            if len(indices) > 1:
                rng.shuffle(indices)
                for i in range(2, len(indices)+1, 2):
                    self.pairs.append(indices[i-2:i])
        rng.shuffle(self.pairs, axis=0)
        
    def __iter__(self):
        return iter(self.pairs)
    
    def __len__(self):
        return len(self.pairs)


# Loss & Metrics


class NT_Xent(nn.Module):
    def __init__(self, temperature):
        super(NT_Xent, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)
 
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        mask = self.mask_correlated_samples(batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        
        with torch.no_grad():
            top1_negative_samples, _ = negative_samples.topk(1)
            avg_rank = logits.argsort(descending=True).argmin(dim=1).float().mean().cpu().numpy()

        return loss, avg_rank


def get_ranked_list(embeds, top_size, annoy_num_trees = 32):
    annoy_index = None
    annoy2id = []
    id2annoy = dict()
    for track_id, track_embed in embeds.items():
        id2annoy[track_id] = len(annoy2id)
        annoy2id.append(track_id)
        if annoy_index is None:
            annoy_index = annoy.AnnoyIndex(len(track_embed), 'angular')
        annoy_index.add_item(id2annoy[track_id], track_embed)
    annoy_index.build(annoy_num_trees)
    ranked_list = dict()
    for track_id in embeds.keys():
        candidates = annoy_index.get_nns_by_item(id2annoy[track_id], top_size+1)[1:] # exclude trackid itself
        candidates = list(filter(lambda x: x != id2annoy[track_id], candidates))
        ranked_list[track_id.item()] = [annoy2id[candidate].item() for candidate in candidates]
    return ranked_list


def position_discounter(position):
    return 1.0 / np.log2(position+1)   

def get_ideal_dcg(relevant_items_count, top_size):
    dcg = 0.0
    for result_indx in range(min(top_size, relevant_items_count)):
        position = result_indx + 1
        dcg += position_discounter(position)
    return dcg

def compute_dcg(query_trackid, ranked_list, track2artist_map, top_size):
    query_artistid = track2artist_map[query_trackid]
    dcg = 0.0
    for result_indx, result_trackid in enumerate(ranked_list[:top_size]):
        assert result_trackid != query_trackid
        position = result_indx + 1
        discounted_position = position_discounter(position)
        result_artistid = track2artist_map[result_trackid]
        if result_artistid == query_artistid:
            dcg += discounted_position
    return dcg


def eval_submission(submission, gt_meta_info, top_size = 100):
    track2artist_map = gt_meta_info.set_index('trackid')['artistid'].to_dict()
    artist2tracks_map = gt_meta_info.groupby('artistid').agg(list)['trackid'].to_dict()
    ndcg_list = []
    for query_trackid in tqdm(submission.keys()):
        ranked_list = submission[query_trackid]
        query_artistid = track2artist_map[query_trackid]
        query_artist_tracks_count = len(artist2tracks_map[query_artistid])
        ideal_dcg = get_ideal_dcg(query_artist_tracks_count-1, top_size=top_size)
        dcg = compute_dcg(query_trackid, ranked_list, track2artist_map, top_size=top_size)
        try:
            ndcg_list.append(dcg/ideal_dcg)
        except ZeroDivisionError:
            continue
    return np.mean(ndcg_list)


# Train & Inference functions


class Embeddings(nn.Module):
    def __init__(self, hidden_size, sequence_size) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(
            nn.init.trunc_normal_(torch.zeros(1, 1, hidden_size), mean=0.0, std=0.02)
        )
        self.position_embeddings = nn.Parameter(
            nn.init.trunc_normal_(
                torch.zeros(1, sequence_size + 1, hidden_size), mean=0.0, std=0.02
            )
        )
        self.dropout = nn.Dropout(0.0)
        
    def interpolate_pos_encoding(self, embeddings, height, width):
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings
    
    def forward(self, x):
        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        embeddings = embeddings + self.position_embeddings

        #embeddings = self.dropout(embeddings)

        return embeddings


class TransformerNet(nn.Module):
    def __init__(self, output_features_size, sequence_size):
        super().__init__()
        self.output_features_size = output_features_size
        self.embedding = Embeddings(output_features_size, sequence_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.output_features_size,
                                                        nhead=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        return x[:, 0, :]


class SimCLR(nn.Module):
    def __init__(self, encoder, projection_dim):
        super().__init__()
        self.encoder = encoder
        self.n_features = encoder.output_features_size
        self.projection_dim = projection_dim
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, self.projection_dim, bias=False),
        )
        
    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return h_i, h_j, z_i, z_j


def inference(model, dataset, batch_size, device='cpu'):
    embeds = dict()
    dataloader = DataLoader(dataset,  batch_size=batch_size, num_workers=10)
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        tracks_ids, tracks_features = batch
        with torch.no_grad():
            tracks_embeds = model(tracks_features.to(device))
            for track_id, track_embed in zip(tracks_ids, tracks_embeds):
                embeds[track_id] = track_embed.cpu().numpy()
    return embeds


def train(module, train_dataset, val_dataset, batch_size, 
          optimizer, scheduler, criterion, num_epochs, 
          checkpoint_path, device='cpu', top_size = 100):
    max_ndcg = None
    
    pbar = tqdm(range(num_epochs), total=num_epochs)
    
    for epoch in pbar:
        
        dataloader = DataLoader(train_dataset,  batch_size=batch_size, 
                                sampler=TrainSampler(train_dataset), 
                                drop_last=True, num_workers=10)
        qbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for idx, batch in qbar:
            gc.collect()
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            module.train()
            x_i, x_j = batch[:, 0, :, :], batch[:, 1, :, :]
            h_i, h_j, z_i, z_j = module(x_i.to(device), x_j.to(device))
            loss, avg_rank = criterion(z_i, z_j)
            loss.backward()
            optimizer.step()
            print('loss: {0:.3f}, avg_rank: {1:.3f}'.format(loss, avg_rank))
        
        with torch.no_grad():
            model_encoder = module.encoder
            embeds_encoder = inference(model_encoder, val_dataset, batch_size, device=device)
            ranked_list_encoder = get_ranked_list(embeds_encoder, top_size)
            val_ndcg_encoder = eval_submission(ranked_list_encoder, val_dataset.test_df)
            
            model_projector = nn.Sequential(module.encoder, module.projector)
            embeds_projector = inference(model_projector, val_dataset, batch_size, device=device)
            ranked_list_projector = get_ranked_list(embeds_projector, top_size)
            val_ndcg_projector = eval_submission(ranked_list_projector, val_dataset.test_df)
            
            if (max_ndcg is None) or (val_ndcg_encoder > max_ndcg):
                max_ndcg = val_ndcg_encoder
                torch.save(model_encoder.state_dict(), checkpoint_path)
        print('Validation nDCG on epoch: encoder - {0:.3f}, projector - {1:.3f}'.format(
                    val_ndcg_encoder, val_ndcg_projector
                )
            )
        scheduler.step()


def save_submission(submission, submission_path):
    with open(submission_path, 'w') as f:
        for query_trackid, result in submission.items():
            f.write("{}\t{}\n".format(query_trackid, " ".join(map(str, result))))

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    seed = 42
    set_seed(seed)

    TRAINSET_PATH = 'train_features'
    TESTSET_PATH = 'test_features'
    TRAINSET_META_PATH = 'train_meta.tsv'
    TESTSET_META_PATH = 'test_meta.tsv'
    SUBMISSION_PATH = 'submission.txt'
    MODEL_PATH = 'model.pt'
    CHECKPOINT_PATH = 'best.pt'

    BATCH_SIZE = 256
    N_CHANNELS = 512
    SEQ_SIZE = 60
    PROJECTION_DIM = 128
    NUM_EPOCHS = 10
    LR = 1e-4
    TEMPERATURE = 0.1
    VAL_FOLD = 7

    sim_clr = SimCLR(
        encoder = TransformerNet(N_CHANNELS, SEQ_SIZE),
        projection_dim = PROJECTION_DIM
    ).to(device)

    train_meta_info = pd.read_csv(TRAINSET_META_PATH, sep='\t')
    test_meta_info = pd.read_csv(TESTSET_META_PATH, sep='\t')
    validation_meta_info = train_meta_info[train_meta_info['archive_features_path'].str.contains(r'^{}'.format(VAL_FOLD))]
    train_meta_info = train_meta_info.drop(validation_meta_info.index)
    train_meta_info.reset_index(drop=True, inplace=True)
    validation_meta_info.reset_index(drop=True, inplace=True)

    print("Loaded data")
    print("Train set size: {}".format(len(train_meta_info)))
    print("Validation set size: {}".format(len(validation_meta_info)))
    print("Test set size: {}".format(len(test_meta_info)))
    print()

    train_dataset = TrainDataset(train_meta_info, TRAINSET_PATH)
    val_dataset = TestDataset(validation_meta_info, TRAINSET_PATH)
    test_dataset = TestDataset(test_meta_info, TESTSET_PATH)

    print("Train")
    
    optimizer = torch.optim.Adam(sim_clr.parameters(), lr = LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    train(
        module = sim_clr,
        train_dataset = train_dataset,
        val_dataset = val_dataset,
        batch_size = BATCH_SIZE,
        optimizer = optimizer,
        scheduler = scheduler,
        criterion = NT_Xent(temperature = TEMPERATURE),
        num_epochs = NUM_EPOCHS,
        checkpoint_path = CHECKPOINT_PATH,
        device=device
    )

    print("Submission")

    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE)
    model = sim_clr.encoder
    embeds = inference(model, test_dataset, BATCH_SIZE, device=device)
    submission = get_ranked_list(embeds, 100)
    save_submission(submission, SUBMISSION_PATH)
    torch.save(sim_clr.state_dict(), MODEL_PATH)

if __name__ == '__main__':
    main()





