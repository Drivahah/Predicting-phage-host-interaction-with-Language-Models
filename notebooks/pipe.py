# import the libraries
import os
import pandas as pd
import torch
import argparse
import numpy as np
# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer, AutoModel
# from tensorflow.keras import layers
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN
from transformers import T5Tokenizer, T5EncoderModel, XLNetTokenizer, XLNetModel
# from torch.optim.lr_scheduler import StepLR
# from torch.utils.data import TensorDataset, DataLoader
import logging
# import ast

# Set workiiing directory to file location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Print start in a file
with open('A.txt', 'a') as f:
    print('Start', file=f)

# parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--embedder', type=str, choices=['prott5', 'protxlnet'], default='prott5', help='Embedder to use')
parser.add_argument('--fine_tune', action='store_true', help='Whether to fine-tune the embedder')
parser.add_argument('--estimator', type=str, choices=['logreg', 'rf', 'transformer'], default='logreg', help='Final estimator to use')
parser.add_argument('--load_embedder', action='store_true', help='Whether to load pre-trained parameters for the embedder')
parser.add_argument('--load_estimator', action='store_true', help='Whether to load pre-trained parameters for the final estimator')
parser.add_argument('--oversampling', type=str, choices=['smote', 'adasyn', 'none'], default='none', help='Oversampling technique to use for imbalanced data')
parser.add_argument('--grid_search', action='store_true', help='Whether to perform grid search for the pipeline')
parser.add_argument('--param_grid', type=str, default='{}', help='Parameter grid for the grid search as a string')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for fine-tuning the embedder')
parser.add_argument('--steps', type=int, default=10, help='Number of steps to train the embedder before freezing the parameters')
parser.add_argument('--lr_embedder', type=float, default=0.01, help='Learning rate for the embedder')
parser.add_argument('--lr_fine_tuned', type=float, default=0.001, help='Learning rate for the fine-tuned model')
parser.add_argument('--batch_size', type=int, default=3, help='Batch size for embedding the data')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for the models')
parser.add_argument('--debug', action='store_true', help='Whether to enable debug logging')
parser.add_argument('--quick', action='store_true', help='Run only on the first batch')
args = parser.parse_args()

# Print args parsed in a file
with open('A.txt', 'a') as f:
    print(args, file=f)

def load_data(df_path):
    if not os.path.exists(df_path):
            raise FileNotFoundError(f'{df_path} does not exist')
    df = pd.read_pickle(df_path)
    # Sort by length of 'sequence_phage' and 'sequence_k12' columns
    # It reduces the number of padding residues needed
    df.sort_values(by=['sequence_phage', 'sequence_k12'], key=lambda x: x.str.len(), ascending=False, inplace=True)

    # Return X, y columns as numpy arrays
    return df['sequence_phage'].values, df['pair'].values


# define the custom embedder classes
class BaseEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name, device='cuda:0', fine_tune=False, num_epochs=1, num_steps=100, learning_rate=1e-3):
        # # Set device and check if available
        # if device == 'cuda:0' and not torch.cuda.is_available():
        #     raise RuntimeError('CUDA is not available')
            
        # self.device_str = device # Fix a bug when sklearn tries to clone the model
        # self.device = torch.device(device)
        self.fine_tune = fine_tune
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.model_name = model_name
        # Load tokenizer and model
        self.load_model_and_tokenizer()
        # only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
        self.model.full() if self.device=='cpu' else self.model.half()
        self.model.eval() # set model to eval mode, we don't want to train it
        
    def load_model_and_tokenizer(self):
        raise NotImplementedError
    
    def fit(self, X, y=None):
        self.device = torch.device(device)

        if self.fine_tune:
            self.model.train() # set model to training mode
            optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
            for epoch in range(self.num_epochs):
                for step in range(self.num_steps):
                    # get the batch
                    batch = X[step]
                    # encode the batch
                    token_encoding = self.tokenizer.batch_encode_plus(batch, add_special_tokens=True, padding="longest")
                    input_ids = torch.tensor(token_encoding['input_ids']).to(self.device)
                    attention_mask = torch.tensor(token_encoding['attention_mask']).to(self.device)
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
            self.model.eval() # set model back to eval mode
        return self

    def transform(self, X, batch_size=3):
        self.device = torch.device(device)
        # initialize an empty list to store the embeddings
        embeddings_list = []
        # loop over the batches
        for i in range(0, len(X), batch_size):
            # get the batch
            batch = X[i:i+batch_size]
            # encode the batch
            token_encoding = self.tokenizer.batch_encode_plus(batch, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(token_encoding['input_ids']).to(self.device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(self.device)
            with torch.no_grad():
                embeddings = self.model(input_ids, attention_mask=attention_mask).last_hidden_state.mean(dim=1).cpu().numpy()
            # append the embeddings to the list
            embeddings_list.append(embeddings)
        # concatenate the list to an array
        embeddings_array = np.concatenate(embeddings_list, axis=0)
        return embeddings_array

class ProtT5Embedder(BaseEmbedder):
    def load_model_and_tokenizer(self):
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, do_lower_case=False)
        self.model = T5EncoderModel.from_pretrained(self.model_name).to(self.device)

class ProtXLNetEmbedder(BaseEmbedder):
    def load_model_and_tokenizer(self):
        self.tokenizer = XLNetTokenizer.from_pretrained(self.model_name, do_lower_case=False)
        self.model = XLNetModel.from_pretrained(self.model_name).to(self.device)


# Print class defined in a file
with open('A.txt', 'a') as f:
    print('Classes defined', file=f)

# create the pipeline
if args.embedder == 'prott5':
    embedder = ProtT5Embedder('Rostlab/prot_t5_xl_half_uniref50-enc', fine_tune=args.fine_tune, device=args.device)
elif args.embedder == 'protxlnet':
    embedder = ProtXLNetEmbedder('Rostlab/prot_xlnet', fine_tune=args.fine_tune, device=args.device)

if args.load_embedder:
    # load pre-trained parameters for the embedder
    embedder.load_state_dict(torch.load('embedder.pth'))
    embedder.eval()

if args.estimator == 'logreg':
    estimator = LogisticRegression()
elif args.estimator == 'rf':
    estimator = RandomForestClassifier()
# elif args.estimator == 'transformer':
#     # use a transformer model as the final estimator
#     estimator = TransformerClassifier(device=args.device)

if args.load_estimator:
    # load pre-trained parameters for the final estimator
    estimator = joblib.load(args.estimator + '.pkl')

if args.oversampling == 'smote':
    # use SMOTE for imbalanced data
    pipe = ImbPipeline([
        ('embedder', embedder),
        ('smote', SMOTE()),
        ('estimator', estimator)
    ])
elif args.oversampling == 'adasyn':
    # use ADASYN for imbalanced data
    pipe = ImbPipeline([
        ('embedder', embedder),
        ('adasyn', ADASYN()),
        ('estimator', estimator)
    ])
else:
    # do not use any oversampling technique
    pipe = Pipeline([
        ('embedder', embedder),
        ('estimator', estimator)
    ])

# Print pipeline defined in a file
with open('A.txt', 'a') as f:
    print('Pipeline defined', file=f)

# create a logger object
logger = logging.getLogger('pipeline')
# set the level of logging
if args.debug:
    # enable debug logging if specified in the command line
    logger.setLevel(logging.DEBUG)
else:
    # otherwise use info logging
    logger.setLevel(logging.INFO)
# create a file handler object
file_handler = logging.FileHandler('log.txt', 'a')
# set the format of logging
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add the formatter to the file handler
file_handler.setFormatter(formatter)
# add the file handler to the logger
logger.addHandler(file_handler)
# log the pipeline options
logger.info(f'Pipeline options: embedder={args.embedder}, fine_tune={args.fine_tune}, estimator={args.estimator}, load_embedder={args.load_embedder}, load_estimator={args.load_estimator}, oversampling={args.oversampling}, grid_search={args.grid_search}, param_grid={args.param_grid}, epochs={args.epochs}, steps={args.steps}, lr_embedder={args.lr_embedder}, lr_fine_tuned={args.lr_fine_tuned}, batch_size={args.batch_size}')

# load the data
INPUT_FOLDER = os.path.join('..', 'data', 'interim')
DATA_PATH = os.path.join(INPUT_FOLDER, '2_model_df.pkl')
X, y = load_data(DATA_PATH)
# log the data shape
logger.info(f'Data shape: X={X.shape}, y={y.shape}')

# perform nested cross-validation
param_grid = eval(args.param_grid) # convert the string to a dictionary
if args.grid_search:
    # perform grid search for the pipeline
    logger.debug('Performing grid search for the pipeline')
    grid = GridSearchCV(pipe, param_grid, cv=5)
    scores = cross_val_score(grid, X, y, cv=5)
    # log the best parameters and score from the grid search
    logger.debug(f'Best parameters: {grid.best_params_}')
    logger.debug(f'Best score: {grid.best_score_}')
else:
    # use the default parameters for the pipeline
    logger.debug('Using the default parameters for the pipeline')
    scores = cross_val_score(pipe, X, y, cv=5)

# log the nested cross-validation scores and the mean score
logger.info(f'Nested cross-validation scores: {scores}')
logger.info(f'Mean score: {scores.mean()}')

# Print end in a file
with open('A.txt', 'a') as f:
    print('End', file=f)