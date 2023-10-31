import pandas as pd
import torch
import re
import time
from transformers import T5Tokenizer, T5EncoderModel



class PairingPredictor():
    def __init__(self, protein_pairs):
        # Parameters
        self.actions = {
            'per_residue': True,
            'per_protein': True
        }
        self.embed_batch_size = 100
        self.models_config = {
            'embedder': 't5_xl_u50',
            'device': 'cuda:0'
        }

        # Data
        self.get_input(protein_pairs) # Get input data
        self.init_embedded_proteins() # Initialize embedded_proteins
        self.output = []

    def get_input(self, protein_pairs: pd.DataFrame):
        # Check that protein_pairs is a DataFrame
        if not isinstance(protein_pairs, pd.DataFrame):
            raise TypeError('protein_pairs must be a DataFrame')
        
        # Check that protein_pairs has the correct columns
        if not all(col in protein_pairs.columns for col in ['seqID_phage', 'sequence_phage', 'seqID_k12', 'sequence_k12']):
            raise ValueError('protein_pairs must have the following columns: seqID_phage, sequence_phage, seqID_bacteria, sequence_bacteria')

        # Sort protein pairs by length of 'sequence_phage' and 'sequence_k12' columns
        # It reduces the number of padding residues needed
        protein_pairs = protein_pairs.sort_values(by=['sequence_phage', 'sequence_k12'], key=lambda x: x.str.len())

        # Store IDs and sequences in a dict
        self.input = {
            'phage': dict(),
            'bacteria': dict()
        }
        self.input['phage']['seqID'] = protein_pairs['seqID_phage'].tolist()
        self.input['phage']['sequence'] = protein_pairs['sequence_phage'].tolist()
        self.input['bacteria']['seqID'] = protein_pairs['seqID_k12'].tolist()
        self.input['bacteria']['sequence'] = protein_pairs['sequence_k12'].tolist()

    def init_embedded_proteins(self):
        # Initialize embedded_proteins
        self.embedded_proteins = {
            'phage': dict(),
            'bacteria': dict()
        }
        if self.actions['per_residue']:
            self.embedded_proteins['phage']['residue_embs'] = []
            self.embedded_proteins['bacteria']['residue_embs'] = []
        if self.actions['per_protein']:
            self.embedded_proteins['phage']['protein_embs'] = []
            self.embedded_proteins['bacteria']['protein_embs'] = []

    def update_actions(self, actions: dict, overwrite_embedded_proteins=False):
        # Check that actions is a dict
        if not isinstance(actions, dict):
            raise TypeError('actions must be a dict')

        # Check that actions has the correct keys
        if not all(key in actions.keys() for key in ['per_residue', 'per_protein']):
            raise ValueError('actions must have the following keys: per_residue, per_protein')

        # Check that actions has the correct values
        if not all(isinstance(value, bool) for value in actions.values()):
            raise ValueError('actions values must be boolean')

        # Update actions
        self.actions.update(actions)

        if not overwrite_embedded_proteins:
            # If embedded_proteins is empty, initialize it
            # Otherwise raise error
            if not self.embedded_proteins['phage'] and not self.embedded_proteins['bacteria']:
                self.init_embedded_proteins()
            else:
                raise ValueError('embedded_proteins is not empty. Set overwrite_embedded_proteins to True to overwrite it')


    def update_embed_batch_size(self, embed_batch_size: int):
        # Check that embed_batch_size is an int
        if not isinstance(embed_batch_size, int):
            raise TypeError('embed_batch_size must be an int')

        # Check that embed_batch_size is positive
        if embed_batch_size <= 0:
            raise ValueError('embed_batch_size must be positive')

        # Update embed_batch_size
        self.embed_batch_size = embed_batch_size
    
    def get_embedder(self):
        # Set device and check if available
        if self.models_config['device'] == 'cuda:0' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available')
        self.device = torch.device(self.models_config['device'])
        
        if self.models_config['embedder'] == 't5_xl_u50':
            # Load T5 tokenizer and model if self.tokenizer and self.embedder are not defined
            if not hasattr(self, 'tokenizer') or not hasattr(self, 'embedder'):
                self.tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
                self.embedder = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc').to(device)
                # only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
                self.embedder.full() if self.device=='cpu' else self.embedder.half()
                self.embedder.eval() # set model to eval mode, we don't want to train it

    def embed_pairs(self):
        # Check that input has been loaded
        if not self.input:
            raise ValueError('input has not been loaded')

        # Embed phage and bacteria proteins
        start = time.time()

        self.embed('phage')
        self.embed('bacteria')

        end = time.time()
        print(f'Embedding time: {end - start} seconds')

    def embed(self, organism: str):
        # Check that organism is a valid organism
        if organism not in self.input.keys():
            raise ValueError('organism must be either "phage" or "bacteria"')
        
        # Check that tokenizer and embedder have been loaded
        if not hasattr(self, 'tokenizer') or not hasattr(self, 'embedder'):
            start = time.time()
            self.get_embedder()
            end = time.time()
            print(f'Tokenizer and embedder loading time: {end - start} seconds')
        
        seq_dict = self.input[organism]
        batch = list()
        for i, (id, seq) in enumerate(zip(seq_dict['seqID'], seq_dict['sequence']), 1):
            # Format and append sequence to batch
            seq_len = len(seq)
            # Replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
            seq = " ".join(list(re.sub(r"[UZOB]", "X", seq)))
            batch.append((id, seq, seq_len))

            # Embed batch if it is full or if it is the last batch
            if i % self.embed_batch_size == 0 or i == len(seq_dict['seqID']):
                ids, seqs, seq_lens = zip(*batch)
                batch = list()

                # add_special_tokens adds extra token at the end of each sequence
                token_encoding = self.tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
                input_ids      = torch.tensor(token_encoding['input_ids']).to(self.device)
                attention_mask = torch.tensor(token_encoding['attention_mask']).to(self.device)
                
                try:
                    with torch.no_grad():
                        # returns: ( batch-size x max_seq_len_in_minibatch x embedding_dim )
                        embedding_repr = self.embedder(input_ids, attention_mask)
                except RuntimeError:
                    print("RuntimeError during embedding for {} (L={})".format(id, seq_len))
                    continue

                for batch_idx, identifier in enumerate(ids): # for each protein in the current mini-batch
                    s_len = seq_lens[batch_idx]
                    # slice off padding --> batch-size x seq_len x embedding_dim  
                    emb = embedding_repr.last_hidden_state[batch_idx,:s_len]
                    if self.actions['per_residue']: # store per-residue embeddings (Lx1024)
                        self.embedded_proteins[organism]["residue_embs"].append(emb.detach().cpu().numpy().squeeze())
                    if self.actions['per_protein']: # apply average-pooling to derive per-protein embeddings (1024-d)
                        protein_emb = emb.mean(dim=0)
                        self.embedded_proteins[organism]["protein_embs"].append(protein_emb.detach().cpu().numpy().squeeze())

    def save_log(self, file_path: str):
        # Save parameters and embedded proteins in a txt file
        with open(file_path, 'w') as f:
            f.write(f'actions: {self.actions}\n\n')
            f.write(f'embed_batch_size: {self.embed_batch_size}\n\n')
            f.write(f'models_config: {self.models_config}\n\n')
            f.write(f'input: {self.input}\n\n')
            f.write(f'embedded_proteins: {self.embedded_proteins}\n\n')
