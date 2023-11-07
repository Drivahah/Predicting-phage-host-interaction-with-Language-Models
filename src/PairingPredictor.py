import pandas as pd
import numpy as np
import torch
import re
import time
import os
from transformers import T5Tokenizer, T5EncoderModel



class PairingPredictor():
    def __init__(self, protein_pairs, debug=False):
        # Create debug file stating time
        if debug:
            with open('PairingPredictor_debug.txt', 'w') as f:
                f.write(f'Start time: {time.ctime()}\n\n')

        # Parameters
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
        self.get_input(protein_pairs, debug) # Get input data
        self.init_embedded_proteins() # Initialize embedded_proteins
        self.output = []
        self.n_pairs = len(protein_pairs)


    def get_input(self, protein_pairs: pd.DataFrame, debug=False):
        # Check that protein_pairs is a DataFrame
        if not isinstance(protein_pairs, pd.DataFrame):
            raise TypeError('protein_pairs must be a DataFrame')
        
        # Check that protein_pairs has the correct columns
        if not all(col in protein_pairs.columns for col in ['seqID_phage', 'sequence_phage', 'seqID_k12', 'sequence_k12']):
            raise ValueError('protein_pairs must have the following columns: seqID_phage, sequence_phage, seqID_bacteria, sequence_bacteria')

        # Sort protein pairs by length of 'sequence_phage' and 'sequence_k12' columns
        # It reduces the number of padding residues needed
        protein_pairs.sort_values(by=['sequence_phage', 'sequence_k12'], key=lambda x: x.str.len(), ascending=False, inplace=True)

        if debug:
            # Write number of proteins in a txt file
            with open('PairingPredictor_debug.txt', 'a') as f:
                f.write('get_input_____________________________________________________\n')
                f.write(f"Number of phage proteins: {len(protein_pairs['sequence_phage'])}\n")
                f.write(f"Number of bacteria proteins: {len(protein_pairs['sequence_k12'])}\n")

        # Store IDs and sequences in a dict
        self.input = {
            'phage': dict(),
            'bacteria': dict()
        }
        self.input['phage']['seqID'] = protein_pairs['seqID_phage'].tolist()
        self.input['phage']['sequence'] = protein_pairs['sequence_phage'].tolist()
        self.input['bacteria']['seqID'] = protein_pairs['seqID_k12'].tolist()
        self.input['bacteria']['sequence'] = protein_pairs['sequence_k12'].tolist()

        if debug:
            # Write number of proteins (seqID and sequence) in a txt file
            with open('PairingPredictor_debug.txt', 'a') as f:
                f.write(f'Number of phage seqID: {len(self.input["phage"]["seqID"])}\n')
                f.write(f'Number of phage sequence: {len(self.input["phage"]["sequence"])}\n')
                f.write(f'Number of bacteria seqID: {len(self.input["bacteria"]["seqID"])}\n')
                f.write(f'Number of bacteria sequence: {len(self.input["bacteria"]["sequence"])}\n\n')

    def init_embedded_proteins(self):
        # Initialize embedded_proteins
        self.embedded_proteins = {
            'phage': dict(),
            'bacteria': dict()
        }
        if self.actions['per_residue']:
            self.embedded_proteins['phage']['residue_embs'] = []
            self.embedded_proteins['bacteria']['residue_embs'] = []
            self.embedded_proteins['paired'] = dict()
        if self.actions['per_protein']:
            self.embedded_proteins['phage']['protein_embs'] = []
            self.embedded_proteins['bacteria']['protein_embs'] = []
            self.embedded_proteins['paired'] = dict()

    def update_actions(self, actions: dict, overwrite_embedded_proteins=False):
        # Check that actions is a dict
        if not isinstance(actions, dict):
            raise TypeError('actions must be a dict')

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
                self.embedder = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc').to(self.device)
                # only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
                self.embedder.full() if self.device=='cpu' else self.embedder.half()
                self.embedder.eval() # set model to eval mode, we don't want to train it

    def embed_pairs(self, path=None, debug=False):
        # Check that input has been loaded
        if not self.input:
            raise ValueError('input has not been loaded')

        # If specified path exists, load embedded_proteins from it
        if os.path.exists(path):
            self.embedded_proteins = torch.load(path)
            if debug:
                # State that embedded_proteins has been loaded from path
                with open('PairingPredictor_debug.txt', 'a') as f:
                    f.write(f'Embedded_proteins loaded from {path}\n')
        if len(self.embedded_proteins['phage']['protein_embs']) != self.n_pairs:
            if len(self.embedded_proteins['paired']) != self.n_pairs:
                start = time.time()

                self.embed('phage', debug)
                self.embed('bacteria', debug)

                end = time.time()
                print(f'Embedding time: {end - start} seconds')

                # Save embedded_proteins in a pt file
                if path:   
                    torch.save(self.embedded_proteins, path)

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

    def concatenate_embeddings(self, path=None, debug=False):
        # Check that embedded_proteins has been loaded
        if not self.embedded_proteins['phage'] or not self.embedded_proteins['bacteria']:
            raise ValueError('embedded_proteins has not been loaded')

        # If length of self.embedded_proteins['paired'] = self.n_pairs, then concatenation has already been done
        if len(self.embedded_proteins['paired']) != self.n_pairs:
            # Concatenate phage and bacteria per residue and per protein embeddings 
            # depending on the actions
            start = time.time()

            # if self.actions['per_residue']:
            #     self.concatenate('residue_embs')
            if self.actions['per_protein']:
                self.concatenate('protein_embs', debug=debug)

            end = time.time()
            print(f'Concatenation time: {end - start} seconds')
             # Save the concatenated embeddings
            torch.save(self.embedded_proteins, path)

    def concatenate(self, embedding_type: str, separator = 300000, debug=False, overwrite=False):
        # Check that embedding_type is a valid embedding_type
        if embedding_type not in self.embedded_proteins['phage'].keys():
            raise ValueError('embedding_type must be either "residue_embs" or "protein_embs"')
        
        # If embedded_proteins['paired'] exists already, check if embedding_type is already concatenated
        if 'paired' in self.embedded_proteins.keys():
            if embedding_type in self.embedded_proteins['paired'].keys():
                if not overwrite:
                    raise ValueError(f'{embedding_type} has already been concatenated. Set overwrite to True to overwrite it')

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

    def save_log(self, file_path: str):
        # Save parameters and embedded proteins in a txt file
        with open(file_path, 'w') as f:
            f.write(f'actions: {self.actions}\n\n')
            f.write(f'embed_batch_size: {self.embed_batch_size}\n\n')
            f.write(f'models_config: {self.models_config}\n\n')
            
            # Truncate lists in input and embedded_proteins at 5 elements
            for organism in self.input.keys():
                for key in self.input[organism].keys():
                    f.write(f'{organism}_{key}:\n{self.input[organism][key][:5]}')
                    f.write('\n\n')
            for organism in self.embedded_proteins.keys():
                for key in self.embedded_proteins[organism].keys():
                    f.write(f'{organism}_{key}:\n{self.embedded_proteins[organism][key][:5]}')
                    f.write('\n\n')
                    
