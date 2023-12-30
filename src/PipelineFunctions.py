import pandas as pd
import torch
import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from transformers import T5Tokenizer, T5EncoderModel, XLNetTokenizer, XLNetModel
import logging
from transformers import AdamW

# create a logger object
logger = logging.getLogger('Pipeline')
file_handler = logging.FileHandler('log.txt', 'a')
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def flatten(lst):
    """Flatten a nested list into a single list"""
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

def load_data(df_path, quick, debug=False):
    # Set logger level
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.info(f'LOADING DATA FROM {df_path}')

    # Check if the df_path exists
    if not os.path.exists(df_path):
        logger.error(f'{df_path} does not exist')
        raise FileNotFoundError(f'{df_path} does not exist')
    
    # If quick is True, load the test data instead
    if quick:
        df_dir = os.path.join('..', 'data', 'interim')
        df = pd.read_pickle(os.path.join(df_dir, 'test_df.pkl'))
        logging.debug(f'Loaded test_df.pkl from {df_dir}')
    else:
        df = pd.read_pickle(df_path)

        # Sort by length to reduce padding
        df.sort_values(by=['sequence_phage', 'sequence_k12'], key=lambda x: x.str.len(), ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True) # Make indexes contiguous
        logging.debug(f'Loaded {df_path}')

    # Return X, y columns as numpy arrays
    return df[['sequence_phage', 'sequence_k12']].values, df['pair'].values

# TODO remove fine_tune if I am not using it______________________________________________________________________________________________________________________
# define the custom embedder classes
class BaseEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name, device='cuda:0', fine_tune=False, num_epochs=1, num_steps=0, learning_rate=1e-3, org='phage', debug=False):
        # Set logger level
        if debug:
            logger.setLevel(logging.DEBUG)
        else:  
            logger.setLevel(logging.INFO)

        # Set device and check if available
        if device == 'cuda:0' and not torch.cuda.is_available():
            logger.error(f'Error while initialising {model_name}: CUDA is not available')
            raise RuntimeError(f'Error while initialising {model_name}: CUDA is not available')
            
        self.model_name = model_name
        self.device = device
        self.fine_tune = fine_tune
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.org = org
        self.load_model_and_tokenizer()

        # only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
        self.model.full() if self.device=='cpu' else self.model.half()
        self.model.eval()

    def load_model_and_tokenizer(self):
        raise NotImplementedError
    
    def fit(self, X):
        self.device = torch.device(self.device)

        if self.fine_tune:
            logger.debug(f'Fine-tuning {self.model_name} on {self.org} data')

            # Set model to train mode and define optimizer
            self.model.train()
            optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)

            # Convert X to a list if it is not already a list
            if not isinstance(X, list):
                X = X.tolist()
            logger.debug(f'X is a {type(X)}, shoulde be a list\nX shape:\n{len(X)}\nX[:3]:\n{X[:3]}')

            # Flatten X to be a list of strings
            X = [item[0] for item in X]
            logger.debug(f'X is now a {type(X)}, should be a list of strings\nX[:3]:\n{X[:3]}')

            # Define batch size
            if self.num_steps == 0:
                self.num_steps = len(X)
            batch_size = len(X) // self.num_steps

            # Fine-tune the model
            for epoch in range(self.num_epochs):    # Each epoch is one pass over the entire dataset
                logger.debug(f'EPOCH {epoch+1}/{self.num_epochs}')
                for step in range(self.num_steps):
                    logger.debug(f'STEP {step+1}/{self.num_steps}')

                    # Get the batch and encode it
                    batch = X[step*batch_size:(step+1)*batch_size]
                    logger.debug(f'batch:\n{batch}')
                    token_encoding = self.tokenizer.batch_encode_plus(batch, add_special_tokens=True, padding="longest")
                    input_ids = torch.tensor(token_encoding['input_ids']).to(self.device)
                    logger.debug(f'input_ids:\n{input_ids}')
                    attention_mask = torch.tensor(token_encoding['attention_mask']).to(self.device)
                    logger.debug(f'attention_mask:\n{attention_mask}')
                    embeddings = self.model(input_ids, attention_mask=attention_mask)
                    logger.debug(f'embeddings = model(input_ids, attention_mask):\n{embeddings}')
                    loss = embeddings.loss
                    logger.debug(f'loss = embeddings.loss: {loss}')

                    # Backpropagate loss and update weights
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            # Set model back to eval mode
            self.model.eval()
            logger.debug(f'Finished fine-tuning {self.model_name} on {self.org} data')

        return self

    def transform(self, X, batch_size=1):
        self.device = torch.device(self.device)
        logger.debug(f'Transforming {self.org} data with {self.model_name}')

        # # Convert X to a list if it is not already a flat list
        # if not isinstance(X, list) or any(isinstance(i, list) for i in X):
        #     logger.debug(f'X is a {type(X)}, converting to flat list')
        #     X = flatten(X)
        #     X = [str(item) for item in X]
        # else:
        #     logger.debug(f'X is already a flat list')
        # logger.debug(f'X is a {type(X)}, should be a flat list of strings\nX length: {len(X)}\nX[:3]:\n{X[:3]}')

        # Convert X to a list, introduce spaces between letters, and replace special aminoacids with X
        X = X.tolist()
        sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]
        logger.debug(f'Replaced special aminoacids with X\nX[:3]:\n{X[:3]}')

        # Get the batch and encode it
        embeddings_list = []
        logger.debug(f'batch size: {batch_size}')
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            logger.debug(f'batch index {i}-{i+batch_size}:\n{batch}')
            token_encoding = self.tokenizer.batch_encode_plus(batch, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(token_encoding['input_ids']).to(self.device)
            logger.debug(f'input_ids index {i}-{i+batch_size}:\n{input_ids}')
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(self.device)
            logger.debug(f'attention_mask index {i}-{i+batch_size}:\n{attention_mask}')
            with torch.no_grad():  # No need to calculate gradients
                embeddings = self.model(input_ids, attention_mask)
                logger.debug(f'embeddings = model(input_ids, attention_mask) {i+1}:\n{embeddings}')
                for batch_index in range(len(batch)):
                    emb = embeddings.last_hidden_state[batch_index, :len(batch[batch_index])]
                    logger.debug(f'emb = embeddings.last_hidden_state[batch_index, :len(batch[batch_index])]:\n{emb}')
                    emb = emb.mean(dim=0).detach().cpu().numpy().squeeze() # Average the embeddings over the sequence length
                    logger.debug(f'emb = emb.mean(dim=0).detach().cpu().numpy().squeeze():\n{emb}')
                    emb = emb.reshape(1, -1) # Wrap the 1D array into a 2D array
                    logger.debug(f'emb = emb.reshape(1, -1):\n{emb}')
                    embeddings_list.append(emb)
                    logger.debug(f'embeddings_list.append(emb):\n{embeddings_list}')

        # concatenate the list to an array
        embeddings_array = np.concatenate(embeddings_list, axis=0)
        logger.debug(f'embeddings_array = np.concatenate(embeddings_list, axis=0):\n{embeddings_array}\nFinished transforming {self.org} data with {self.model_name}')
        # logger.debug(f'embeddings_array.shape: {embeddings_array.shape}')

        return embeddings_array

class ProtT5Embedder(BaseEmbedder):
    def load_model_and_tokenizer(self):
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, do_lower_case=False)
        self.model = T5EncoderModel.from_pretrained(self.model_name).to(self.device)

class ProtXLNetEmbedder(BaseEmbedder):
    def load_model_and_tokenizer(self):
        self.tokenizer = XLNetTokenizer.from_pretrained(self.model_name, do_lower_case=False)
        self.model = XLNetModel.from_pretrained(self.model_name).to(self.device)

class SequentialEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, embedder_phage, embedder_bacteria):
        self.embedder_phage = embedder_phage
        self.embedder_bacteria = embedder_bacteria

    def fit(self, X, y=None):
        logger.debug(f'Fitting SequentialEmbedder')
        self.embedder_phage.fit(X['sequence_phage'], y)
        self.embedder_bacteria.fit(X['sequence_k12'], y)
        logger.debug(f'Finished fitting SequentialEmbedder')
        return self

    def transform(self, X, batch_size=1):
        logger.debug(f'Transforming SequentialEmbedder')
        embeddings_phage = self.embedder_phage.transform(X[:, 0], batch_size=batch_size)
        embeddings_bacteria = self.embedder_bacteria.transform(X[:, 1], batch_size=batch_size)
        output = np.concatenate([embeddings_phage, embeddings_bacteria], axis=1)
        # logger.debug(f'SequentialEmbedder output.shape: {output.shape}')
        logger.debug(f'SequentialEmbedder output[:3]:\n{output[:3]}\nFinished transforming SequentialEmbedder')
        return output
    