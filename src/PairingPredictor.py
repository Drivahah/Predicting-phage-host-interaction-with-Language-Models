import pandas as pd
import numpy as np
import torch
import re
import time
import os
from transformers import T5Tokenizer, T5EncoderModel


class PairingPredictor():
    # Get protein pairs file path
    def __init__(self, df_path, debug=None, log=None):
        # Check that debug path exists
        if debug:
            # Create debug file stating time, creating folders if necessary
            if not os.path.exists(os.path.dirname(debug)):
                os.makedirs(os.path.dirname(debug))
            with open(debug, 'w') as f:
                f.write(f'Start time: {time.ctime()}\n\n')

        if log:
            # Create log file stating time, creating folders if necessary
            if not os.path.exists(os.path.dirname(log)):
                os.makedirs(os.path.dirname(log))
            with open(log, 'w') as f:
                f.write(f'Start time: {time.ctime()}\n\n')

        # Parameters
        self.debug = debug
        self.log = log
        self.actions = {
            'per_residue': False,  # Beware of high memory consumption
            'per_protein': True
        }
        self.embed_batch_size = 100
        self.models_config = {
            'embedder': 't5_xl_u50',
            'device': 'cuda:0'
        }

        # Data
        if not os.path.exists(df_path):
            raise FileNotFoundError(f'{df_path} does not exist')
        self.protein_pairs = pd.read_pickle(df_path)
        self.get_input() # Get input data
        self.init_embedded_proteins() # Initialize embedded_proteins
        self.output = []
        self.n_pairs = len(self.protein_pairs)

        # Write parameters in the log file
        if log:
            with open(log, 'a') as f:
                f.write(f'Parameters' + '_' * 70 + '\n')
                f.write(f'actions: {self.actions}\n')
                f.write(f'embed_batch_size: {self.embed_batch_size}\n')
                f.write(f'models_config: {self.models_config}\n\n')

    def get_input(self):
        # Check that protein_pairs is a DataFrame
        if not isinstance(self.protein_pairs, pd.DataFrame):
            raise TypeError('protein_pairs must be a DataFrame')
        
        # Check that protein_pairs has the correct columns
        if not all(col in self.protein_pairs.columns for col in ['seqID_phage', 'sequence_phage', 'seqID_k12', 'sequence_k12']):
            raise ValueError('protein_pairs must have the following columns: seqID_phage, sequence_phage, seqID_bacteria, sequence_bacteria')

        # Check that protein_pairs is not empty
        if self.protein_pairs.empty:
            raise ValueError('protein_pairs is empty')

        if self.log:
            # Write number of proteins not null in a txt file
            with open(self.log, 'a') as f:
                f.write(f'get_input' + '_' * 70 + '\n')
                f.write(f"Number of phage proteins not null: {self.protein_pairs['sequence_phage'].notnull().sum()}\n")
                f.write(f"Number of bacteria proteins not null: {self.protein_pairs['sequence_k12'].notnull().sum()}\n\n")

        # Sort protein pairs by length of 'sequence_phage' and 'sequence_k12' columns
        # It reduces the number of padding residues needed
        self.protein_pairs.sort_values(by=['sequence_phage', 'sequence_k12'], key=lambda x: x.str.len(), ascending=False, inplace=True)

        # Store IDs and sequences in a dict
        self.input = {
            'phage': dict(),
            'bacteria': dict()
        }
        self.input['phage']['seqID'] = self.protein_pairs['seqID_phage'].tolist()
        self.input['phage']['sequence'] = self.protein_pairs['sequence_phage'].tolist()
        self.input['bacteria']['seqID'] = self.protein_pairs['seqID_k12'].tolist()
        self.input['bacteria']['sequence'] = self.protein_pairs['sequence_k12'].tolist()

        if self.debug:
            # Write number of proteins (seqID and sequence) in a txt file
            with open(self.debug, 'a') as f:
                f.write(f'get_input' + '_' * 70 + '\n')
                f.write(f'Number of phage seqID: {len(self.input["phage"]["seqID"])}\n')
                f.write(f'Number of phage sequence: {len(self.input["phage"]["sequence"])}\n')
                f.write(f'Number of bacteria seqID: {len(self.input["bacteria"]["seqID"])}\n')
                f.write(f'Number of bacteria sequence: {len(self.input["bacteria"]["sequence"])}\n\n')

    def init_embedded_proteins(self):
        if self.log:
            with open(self.log, 'a') as f:
                f.write(f'init_embedded_proteins' + '_' * 70 + '\n')

        # Initialize embedded_proteins
        self.embedded_proteins = {
            'phage': dict(),
            'bacteria': dict(),
            'paired': dict()
        }

        if self.actions['per_residue']:
            self.embedded_proteins['phage']['residue_embs'] = []
            self.embedded_proteins['bacteria']['residue_embs'] = []
            self.embedded_proteins['paired']['residue_embs'] = []

            if self.log:
                with open(self.log, 'a') as f:
                    f.write(f'embedded_proteins["phage"]["residue_embs"] initialized\n')
                    f.write(f'embedded_proteins["bacteria"]["residue_embs"] initialized\n')
                    f.write(f'embedded_proteins["paired"] initialized\n')
        if self.actions['per_protein']:
            self.embedded_proteins['phage']['protein_embs'] = []
            self.embedded_proteins['bacteria']['protein_embs'] = []
            self.embedded_proteins['paired']['protein_embs'] = []

            if self.log:
                with open(self.log, 'a') as f:
                    f.write(f'embedded_proteins["phage"]["protein_embs"] initialized\n')
                    f.write(f'embedded_proteins["bacteria"]["protein_embs"] initialized\n')
                    f.write(f'embedded_proteins["paired"] initialized\n')
        if self.log:
            with open(self.log , 'a') as f:
                f.write('\n')

    def update_actions(self, actions: dict):
        # Check that actions is a dict
        if not isinstance(actions, dict):
            raise TypeError('actions must be a dict')

        # Check that actions has the correct values
        if not all(isinstance(value, bool) for value in actions.values()):
            raise ValueError('actions values must be boolean')

        # Update actions
        self.actions.update(actions)

        if self.log:
            with open(self.log, 'a') as f:
                f.write('update_actions' + '_' * 70 + '\n')
                f.write(f'actions updated: {self.actions}\n\n')

    def update_embed_batch_size(self, embed_batch_size: int):
        # Check that embed_batch_size is an int
        if not isinstance(embed_batch_size, int):
            raise TypeError('embed_batch_size must be an int')

        # Check that embed_batch_size is positive
        if embed_batch_size <= 0:
            raise ValueError('embed_batch_size must be positive')

        # Update embed_batch_size
        self.embed_batch_size = embed_batch_size

        if self.log:
            with open(self.log, 'a') as f:
                f.write('update_embed_batch_size' + '_' * 70 + '\n')
                f.write(f'embed_batch_size updated: {self.embed_batch_size}\n\n')
    
    def get_embedder(self):
        # Set device and check if available
        if self.models_config['device'] == 'cuda:0' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available')
            
        self.device = torch.device(self.models_config['device'])
        
        if self.models_config['embedder'] == 't5_xl_u50':
            # Load T5 tokenizer and model if self.tokenizer and self.embedder are not defined
            if not hasattr(self, 'tokenizer') or not hasattr(self, 'embedder'):
                self.tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
                self.embedder = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc').to(self.device)
                # only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
                self.embedder.full() if self.device=='cpu' else self.embedder.half()
                self.embedder.eval() # set model to eval mode, we don't want to train it

    def embed_pairs(self, path=None, debug=False):
        # Check that input has been loaded
        if not self.input:
            raise ValueError('input has not been loaded')

        if self.log:
            with open(self.log, 'a') as f:
                f.write(f'embed_pairs' + '_' * 70 + '\n')

        # If specified path exists, load embedded_proteins from it
        if os.path.exists(path):
            self.embedded_proteins = torch.load(path)

            # Initialize embedded_proteins if it is empty
            if 'phage' not in self.embedded_proteins:
                self.embedded_proteins['phage'] = dict()
                if 'protein_embs' not in self.embedded_proteins['phage']:
                    self.embedded_proteins['phage']['protein_embs'] = []
            if 'bacteria' not in self.embedded_proteins:
                self.embedded_proteins['bacteria'] = dict()
                if 'protein_embs' not in self.embedded_proteins['bacteria']:
                    self.embedded_proteins['bacteria']['protein_embs'] = []
            if 'paired' not in self.embedded_proteins:
                self.embedded_proteins['paired'] = dict()
                if 'protein_embs' not in self.embedded_proteins['paired']:
                    self.embedded_proteins['paired']['protein_embs'] = []

            if self.log:
                with open(self.log, 'a') as f:
                    f.write(f'embedded_proteins loaded from {path}\n')
                    f.write(f'Number of phage protein_embs: {len(self.embedded_proteins["phage"]["protein_embs"])}\n')
                    f.write(f'Number of bacteria protein_embs: {len(self.embedded_proteins["bacteria"]["protein_embs"])}\n')
                    f.write(f'Number of paired protein_embs: {len(self.embedded_proteins["paired"]["protein_embs"])}\n')
                    f.write(f'Number of expected proteins: {self.n_pairs}\n')

        # If there is a mismatch in the number of proteins, and they were not concatenated
        if len(self.embedded_proteins['phage']['protein_embs']) != self.n_pairs:
            if len(self.embedded_proteins['paired']['protein_embs']) != self.n_pairs:
                if os.path.exists(path) and self.log:
                    with open(self.log, 'a') as f:
                        f.write('Mismatch in the number of proteins and they were not concatenated\n')
                        f.write(f'Re-computing embeddings\n')

                start = time.time()

                self.embed('phage', debug)
                self.embed('bacteria', debug)

                end = time.time()

                if self.log:
                    with open(self.log, 'a') as f:
                        f.write(f'Embedding time: {end - start} seconds\n\n')
                        f.write(f'Number of phage protein_embs: {len(self.embedded_proteins["phage"]["protein_embs"])}\n')
                        f.write(f'Number of bacteria protein_embs: {len(self.embedded_proteins["bacteria"]["protein_embs"])}\n')

                # Save embedded_proteins in a pt file
                if path:   
                    torch.save(self.embedded_proteins, path)

                if self.log:
                    with open(self.log, 'a') as f:
                        f.write(f'embedded_proteins saved in {path}\n')

            elif self.log:
                with open(self.log, 'a') as f:
                    f.write('Proteins concatenated already\n')

        elif self.log:
            with open(self.log, 'a') as f:
                f.write('Proteins embedded already\n')

        if self.log:
            with open(self.log, 'a') as f:
                f.write('\n')

    def embed(self, organism: str, debug=False):
        # Check that organism is a valid organism
        if organism not in self.input.keys():
            raise ValueError('organism must be either "phage" or "bacteria"')
        
        # Check that tokenizer and embedder have been loaded
        if not hasattr(self, 'tokenizer') or not hasattr(self, 'embedder'):
            start = time.time()
            self.get_embedder()
            end = time.time()
            print(f'Tokenizer and embedder loading time: {end - start} seconds')
        
        if debug:
            # Write organism in a txt file
            with open('PairingPredictor_debug.txt', 'a') as f:
                f.write(f'embed_{organism}_____________________________________________________\n')

        seq_dict = self.input[organism]
        batch = list()
        MAX_INPUT_LEN = 300
        for i, (id, seq) in enumerate(zip(seq_dict['seqID'], seq_dict['sequence']), 1):
            # Format and append sequence to batch
            seq_len = len(seq)
            # Replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
            if seq_len <= MAX_INPUT_LEN:
                seq = " ".join(list(re.sub(r"[UZOB]", "X", seq)))
                batch.append((id, seq, seq_len))

            # Embed batch if it is full, if it is the last batch or if current sequence is too long
            if i % self.embed_batch_size == 0 or i == len(seq_dict['seqID']) or seq_len > MAX_INPUT_LEN:
                # Embed batch before the long sequence
                if debug:
                    # Write number of proteins in batch in a txt file
                    with open('PairingPredictor_debug.txt', 'a') as f:
                        f.write(f'Number of proteins in batch: {len(batch)}\n')
                if batch:
                    ids, seqs, seq_lens = zip(*batch)
                    batch = list()

                    # add_special_tokens adds extra token at the end of each sequence
                    token_encoding = self.tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
                    input_ids      = torch.tensor(token_encoding['input_ids']).to(self.device)
                    attention_mask = torch.tensor(token_encoding['attention_mask']).to(self.device)

                    if debug:
                        # Write number of input_ids and attention_mask in a txt file
                        with open('PairingPredictor_debug.txt', 'a') as f:
                            f.write(f'Number of input_ids: {len(input_ids)}\n')
                            f.write(f'Number of attention_mask: {len(attention_mask)}\n')
                            f.write(f'Minimum sequence length in input_ids: {min([len(seq) for seq in input_ids])}\n')
                            f.write(f'Maximum sequence length in input_ids: {max([len(seq) for seq in input_ids])}\n')
                            f.write(f'First input_ids: {input_ids[0]}\n')
                            f.write(f'First attention_mask: {attention_mask[0]}\n')
                    try:
                        with torch.no_grad():
                            # returns: ( batch-size x max_seq_len_in_minibatch x embedding_dim )
                            embedding_repr = self.embedder(input_ids, attention_mask)
                            if debug:
                                # Write len of sequence in a txt file
                                with open('PairingPredictor_debug.txt', 'a') as f:
                                    f.write(f'Embedding successful: Len of sequence: {seq_len}\n')
                    except RuntimeError:
                        print("RuntimeError during embedding for {} (L={})".format(id, seq_len))
                        if debug:
                            # Write error in a txt file
                            with open('PairingPredictor_debug.txt', 'a') as f:
                                f.write(f'                      RuntimeError during embedding for {id} (L={seq_len})\n')
                        continue

                    if debug:
                        # Write number of embedded proteins in a txt file
                        with open('PairingPredictor_debug.txt', 'a') as f:
                            f.write(f'Number of embedding_repr: {len(embedding_repr.last_hidden_state)}\n')
                            f.write(f'Len ids: {len(ids)}\n')

                    for batch_idx in range(len(ids)): # for each protein in the current mini-batch
                        s_len = seq_lens[batch_idx]
                        # slice off padding --> batch-size x seq_len x embedding_dim  
                        emb = embedding_repr.last_hidden_state[batch_idx,:s_len]
                        if self.actions['per_residue']: # store per-residue embeddings (Lx1024)
                            self.embedded_proteins[organism]["residue_embs"].append(emb.detach().cpu().numpy().squeeze())
                        if self.actions['per_protein']: # apply average-pooling to derive per-protein embeddings (1024-d)
                            protein_emb = emb.mean(dim=0)
                            self.embedded_proteins[organism]["protein_embs"].append(protein_emb.detach().cpu().numpy().squeeze())

                    if debug:
                        # Write number of embedded proteins in a txt file
                        with open('PairingPredictor_debug.txt', 'a') as f:
                            f.write(f'Number of embedded proteins: {len(self.embedded_proteins[organism]["protein_embs"])}\n')

                if seq_len > MAX_INPUT_LEN:
                    # Embed long sequence which was not added to batch
                    chunks = [seq[j:j+MAX_INPUT_LEN] for j in range(0, len(seq), MAX_INPUT_LEN)]
                    chunks = [" ".join(list(re.sub(r"[UZOB]", "X", chunk))) for chunk in chunks]
                    token_encoding = self.tokenizer.batch_encode_plus(chunks, add_special_tokens=True, padding="longest")
                    input_ids      = torch.tensor(token_encoding['input_ids']).to(self.device)
                    attention_mask = torch.tensor(token_encoding['attention_mask']).to(self.device)

                    try:
                        with torch.no_grad():
                            # returns: ( chunks-size x max_input_len x embedding_dim )
                            embedding_repr = self.embedder(input_ids, attention_mask)
                    except RuntimeError:
                        print("RuntimeError during embedding for {} (L={})".format(id, seq_len))

                    # Concatenate chunks
                    embedding_repr = embedding_repr.last_hidden_state
                    embedding_repr_list = [embedding_repr[x] for x in range(embedding_repr.size(0))]
                    emb = torch.cat(embedding_repr_list, dim=0)
                    if self.actions['per_residue']:
                        self.embedded_proteins[organism]["residue_embs"].append(emb.detach().cpu().numpy().squeeze())
                    if self.actions['per_protein']:
                        protein_emb = emb.mean(dim=0)
                        self.embedded_proteins[organism]["protein_embs"].append(protein_emb.detach().cpu().numpy().squeeze())

        if debug:
            # Write number of embedded proteins in a txt file
            with open('PairingPredictor_debug.txt', 'a') as f:
                f.write('\n')
                f.write(f'Total number of embedded proteins: {len(self.embedded_proteins[organism]["protein_embs"])}\n\n')

    def concatenate_embeddings(self, path=None, debug=False, separator=300000):
        # Check that embedded_proteins has been loaded
        if 'phage' not in self.embedded_proteins or 'bacteria' not in self.embedded_proteins or 'paired' not in self.embedded_proteins:
            raise ValueError('embedded_proteins has not been initialised')

        if self.log:
            with open(self.log, 'a') as f:
                f.write(f'concatenate_embeddings' + '_' * 70 + '\n')

        # If length of self.embedded_proteins['paired'] = self.n_pairs, then concatenation has already been done
        if len(self.embedded_proteins['paired']['protein_embs']) != self.n_pairs:
            # Concatenate phage and bacteria per residue and per protein embeddings 
            # depending on the actions
            start = time.time()

            if self.actions['per_residue']:
                self.concatenate('residue_embs')
            if self.actions['per_protein']:
                self.concatenate('protein_embs', separator, debug)

            end = time.time()
            if self.log:
                with open(self.log, 'a') as f:
                    f.write(f'Concatenation time: {end - start} seconds\n')
                    f.write(f'Number of concatenated protein_embs: {len(self.embedded_proteins["paired"]["protein_embs"])}\n')

             # Save the concatenated embeddings
            torch.save(self.embedded_proteins['paired'], path)

            if self.log:
                with open(self.log, 'a') as f:
                    f.write(f'embedded_proteins["paired"] saved in {path}\n\n')

        else:
            if self.log:
                with open(self.log, 'a') as f:
                    f.write(f'Concatenation has already been done\n\n')

    def concatenate(self, embedding_type: str, separator=300000, debug=False):
        # Check that embedding_type is a valid embedding_type
        if embedding_type not in self.embedded_proteins['phage'].keys():
            raise ValueError('embedding_type must be either "residue_embs" or "protein_embs"')
        
        # Check that phage and bacteria embeddings have the same number of elements
        n_phage = len(self.embedded_proteins['phage'][embedding_type])
        n_bacteria = len(self.embedded_proteins['bacteria'][embedding_type])
        if n_phage != n_bacteria:
            raise ValueError(f'phage and bacteria {embedding_type} have different number of elements: {n_phage} and {n_bacteria}')
        
        # Only flattened_residue_embs should be concatenated, not the 2D residue_embs
        if embedding_type == 'residue_embs':
            embedding_type = 'flattened_residue_embs'

        # Concatenate phage and bacteria embeddings
        # Note: separator set to 300000 because it is out of the range of T5 embeddings values
        #       and it should help the model distinguish the two proteins
        if debug:
            # Write embedding_type in a txt file
            with open('PairingPredictor_debug.txt', 'a') as f:
                f.write(f'concatenate_{embedding_type}_____________________________________________________\n')
        
        separator = np.array([separator])  # Convert separator to a 1-dimensional array
        self.embedded_proteins['paired'][embedding_type] = []
        for i in range(len(self.embedded_proteins['phage'][embedding_type])):
            phage = self.embedded_proteins['phage'][embedding_type][i]
            bacteria = self.embedded_proteins['bacteria'][embedding_type][i]
            if debug:
                # Write phage, separator and bacteria in a txt file
                with open('PairingPredictor_debug.txt', 'a') as f:
                    f.write(f'phage: {phage}\n')
                    f.write(f'phage type: {type(phage)}\n')
                    f.write(f'separator: {separator}\n')
                    f.write(f'separator type: {type(separator)}\n')
                    f.write(f'bacteria: {bacteria}\n')
                    f.write(f'bacteria type: {type(bacteria)}\n')
            self.embedded_proteins['paired'][embedding_type].append(np.concatenate((phage, separator, bacteria)))

            # Raise error if there is a length mismatch
            if len(self.embedded_proteins['paired'][embedding_type][i]) != len(phage) + len(separator) + len(bacteria):
                raise ValueError(f'Length mismatch in concatenated {embedding_type} for self.input["phage"]["seqID"][{i}] and self.input["bacteria"]["seqID"][{i}], position {i}')                    
