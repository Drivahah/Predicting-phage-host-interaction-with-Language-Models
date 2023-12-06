# import the libraries
import os
import pandas as pd
import torch
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN
from transformers import T5Tokenizer, T5EncoderModel, XLNetTokenizer, XLNetModel
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
from transformer_classifier import TransformerClassifier
import logging

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
class ProtT5Embedder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='Rostlab/prot_t5_xl_half_uniref50-enc', device='cuda:0'):
        # Set device and check if available
        if device == 'cuda:0' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available')
            
        self.device = torch.device(device)
        # load the model
        self.model = SentenceTransformer(model_name).to(self.device)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, batch_size=32):
        # initialize an empty list to store the embeddings
        embeddings_list = []
        # loop over the batches
        for i in range(0, len(X), batch_size):
            # get the batch
            batch = X[i:i+batch_size]
            # encode the batch
            embeddings = self.model.encode(batch)
            # append the embeddings to the list
            embeddings_list.append(embeddings)
            # if quick run, break after the first batch
            if args.quick:
                break
        # concatenate the list to an array
        embeddings_array = np.concatenate(embeddings_list, axis=0)
        return embeddings_array
    
    def get_output_shape(self):
        return self.model.get_sentence_embedding_dimension()

class ProtXLNetEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='Rostlab/prot_xlnet', device='cuda:0'):
        # Set device and check if available
        if device == 'cuda:0' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available')
            
        self.device = torch.device(device)
        # load the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, batch_size=32):
        # initialize an empty list to store the embeddings
        embeddings_list = []
        # loop over the batches
        for i in range(0, len(X), batch_size):
            # get the batch
            batch = X[i:i+batch_size]
            # tokenize the batch
            inputs = self.tokenizer(batch, return_tensors='pt', padding=True).to(self.device)
            # get the last hidden state of the model
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
            # average the embeddings along the sequence dimension
            embeddings = embeddings.mean(dim=1).detach().numpy()
            # append the embeddings to the list
            embeddings_list.append(embeddings)
            # if quick run, break after the first batch
            if args.quick:
                break
        # concatenate the list to an array
        embeddings_array = np.concatenate(embeddings_list, axis=0)
        return embeddings_array
    
    def get_output_shape(self):
        return self.model.config.hidden_size

# define the fine-tuned embedder class
class FineTunedEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, embedder, units=64, device='cuda:0', epochs=10, steps=10, lr_embedder=0.01, lr_fine_tuned=0.001):
        # Set device and check if available
        if device == 'cuda:0' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available')
            
        self.device = torch.device(device)
        # set the embedder, the units, the epochs, and the steps
        self.embedder = embedder
        self.units = units
        self.epochs = epochs
        self.steps = steps
        # create a loss function
        self.loss_function = torch.nn.MSELoss()
        # create a fine-tuned model with a dense layer
        embeddings_shape = self.embedder.get_output_shape()
        input_layer = Input(shape=(embeddings_shape[1],))
        output_layer = layers.Dense(self.units, activation='relu')(input_layer)
        self.fine_tuned_model = Model(input_layer, output_layer)
        self.fine_tuned_model.to(self.device)
        # create an optimizer for the embedder
        self.optimizer_embedder = torch.optim.Adam(self.embedder.parameters(), lr=lr_embedder)

        # create an optimizer for the fine_tuned_model
        self.optimizer_fine_tuned = torch.optim.Adam(self.fine_tuned_model.parameters(), lr=lr_fine_tuned)

        # create a scheduler that changes the learning rate of the embedder after the steps
        self.scheduler_embedder = StepLR(self.optimizer_embedder, step_size=self.steps, gamma=0, last_epoch=-1)
    
    def fit(self, X, y=None):
        # # encode the input data using the embedder
        # embeddings = self.embedder.transform(X)
         # Assuming X and y are PyTorch tensors
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        # loop over the epochs
        for epoch in range(self.epochs):
            # loop over the batches
            for inputs, targets in dataloader:
                # encode the inputs using the embedder
                embeddings = self.embedder(inputs)
                # get the predictions of the fine-tuned model
                predictions = self.fine_tuned_model(embeddings)
                # compute the loss
                loss = self.loss_function(predictions, targets)
                # zero the gradients
                self.optimizer_embedder.zero_grad()
                # backpropagate the loss
                loss.backward()
                # update the parameters
                self.optimizer_embedder.step()
                # update the learning rate
                self.scheduler_embedder.step()
        return self
    
    def transform(self, X, batch_size=32):
        # encode the input data using the embedder
        embeddings = self.embedder.transform(X, batch_size)
        # get the predictions of the fine-tuned model
        predictions = self.fine_tuned_model(embeddings)
        return predictions

# Print class defined in a file
with open('A.txt', 'a') as f:
    print('Classes defined', file=f)

# create the pipeline
if args.embedder == 'prott5':
    embedder = ProtT5Embedder(device=args.device)
elif args.embedder == 'protxlnet':
    embedder = ProtXLNetEmbedder(device=args.device)

if args.fine_tune:
    # add a fine-tuning step to the embedder
    embedder = FineTunedEmbedder(embedder, epochs=args.epochs, steps=args.steps, lr_embedder=args.lr_embedder, lr_fine_tuned=args.lr_fine_tuned, device=args.device)

if args.load_embedder:
    # load pre-trained parameters for the embedder
    embedder.load_state_dict(torch.load('embedder.pth'))
    embedder.eval()

if args.estimator == 'logreg':
    estimator = LogisticRegression()
elif args.estimator == 'rf':
    estimator = RandomForestClassifier()
elif args.estimator == 'transformer':
    # use a transformer model as the final estimator
    estimator = TransformerClassifier(device=args.device)

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