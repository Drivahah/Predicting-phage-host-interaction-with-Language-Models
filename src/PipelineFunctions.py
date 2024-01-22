import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from transformers import T5Tokenizer, T5EncoderModel, XLNetTokenizer, XLNetModel
import logging
from transformers import AdamW
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import BaseSearchCV
import matplotlib.pyplot as plt


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

def flatten_data(X):
        output = X.reshape(X.shape[0], -1)  # flatten data so each dimension of the arrays is a feature, not the two columns
        logger.debug(f'Flattened data X[:3]\n{output[:3]}')
        return output


# TODO remove fine_tune if I am not using it______________________________________________________________________________________________________________________
# define the custom embedder classes
class BaseEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name, device='cuda:0', fine_tune=False, num_epochs=1, num_steps=0, learning_rate=1e-3, org='phage', debug=False, prot=True):
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
        self.prot = prot
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
        X = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in X]
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
                    if self.prot:
                        emb = emb.mean(dim=0).detach().cpu().numpy().squeeze() # Average the embeddings over the sequence length
                        logger.debug(f'emb = emb.mean(dim=0).detach().cpu().numpy().squeeze():\n{emb}')
                        emb = emb.reshape(1, -1) # Wrap the 1D array into a 2D array
                        logger.debug(f'emb = emb.reshape(1, -1):\n{emb}')
                    else:
                        emb = emb.detach().cpu().numpy().squeeze()
                        logger.debug(f'emb = emb.detach().cpu().numpy().squeeze():\n{emb}')
                    embeddings_list.append(emb)
                    logger.debug(f'embeddings_list.append(emb):\n{embeddings_list}')

        # concatenate the list to an array
        axis = 0 if self.prot else 0
        if self.prot:
            embeddings_array = np.concatenate(embeddings_list, axis=axis)
        else:
            embeddings_array = embeddings_list
        logger.debug(f'embeddings_array = embeddings_list:\n{embeddings_array}')
        #     logger.debug(f'embeddings_array = np.stack(embeddings_list):\n{embeddings_array}\nFinished transforming {self.org} data with {self.model_name}')
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
    def __init__(self, embedder_phage, embedder_bacteria, prot=True):
        self.embedder_phage = embedder_phage
        self.embedder_bacteria = embedder_bacteria
        self.prot = prot

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
        if self.prot:
            output = np.concatenate([embeddings_phage, embeddings_bacteria], axis=1)
        else:
            output = [np.vstack((arr1, arr2)) for arr1, arr2 in zip(embeddings_phage, embeddings_bacteria)]
        logger.debug(f'Output type: {type(output)}')
        logger.debug(f'SequentialEmbedder output[:3]:\n{output[:3]}\nFinished transforming SequentialEmbedder')
        return output
    
def calculate_metrics(estimator, X, y):
    y_pred = estimator.predict(X)
    
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        # Compute ROC AUC only if there are both classes present in y_true
        'roc_auc': roc_auc_score(y, estimator.predict_proba(X)[:, 1]) if len(set(y)) > 1 else None
    }
    return metrics

class CustomRandomForestClassifier(RandomForestClassifier):
    def __init__(self, n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None):
        super().__init__(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start, class_weight=class_weight, ccp_alpha=ccp_alpha, max_samples=max_samples)
        self.metrics = []

    def fit(self, X, y):
        super().fit(X, y)
        self.metrics.append(calculate_metrics(self, X, y))
        return self

# Define a simple function to plot the metrics
def plot_metrics(metrics, metric_name, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot([m['train_' + metric_name] for m in metrics], label=f'Train {metric_name}')
    plt.plot([m['test_' + metric_name] for m in metrics], label=f'Test {metric_name}')
    plt.xlabel('Training Portion')
    plt.ylabel(metric_name.capitalize())
    plt.title(f'Train vs Test {metric_name.capitalize()} Over Time')
    plt.legend()
    plt.savefig(save_path)
    plt.close()  # Close the plot to avoid displaying it inline if not desired  

# Define the attention module 
class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(input_dim, 1))

    def forward(self, x):
        e = torch.tanh(torch.matmul(x, self.attention_weights))
        a = F.softmax(e, dim=1)
        output = x * a
        # return torch.sum(output, axis=1)  # Sum over the sequence dimension
        return output

class SelfAttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttentionLayer, self).__init__()
        self.W_q = nn.Parameter(torch.empty(input_dim, input_dim))
        self.W_k = nn.Parameter(torch.empty(input_dim, input_dim))
        self.W_v = nn.Parameter(torch.empty(input_dim, input_dim))

        # Apply Xavier initialization
        init.xavier_uniform_(self.W_q)
        init.xavier_uniform_(self.W_k)
        init.xavier_uniform_(self.W_v)

    def forward(self, x):
        Q = torch.matmul(x, self.W_q)
        K = torch.matmul(x, self.W_k)
        V = torch.matmul(x, self.W_v)

        e = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(Q.size(-1)).float())
        a = F.softmax(e, dim=-1)

        output = torch.matmul(a, V)
        return output

# Now define the overall Neural Network including the attention layer
class AttentionNetwork(nn.Module):
    def __init__(self, input_dim, self_attention=False):
        super(AttentionNetwork, self).__init__()
        if self_attention:
            self.attention = SelfAttentionLayer(input_dim)
        else:
            self.attention = AttentionLayer(input_dim)
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        attention_out = self.attention(x)
        context_vector = torch.sum(attention_out, axis=1)  # Sum over the sequence dimension
        out = torch.sigmoid(self.fc(context_vector))
        return out

class SklearnCompatibleAttentionClassifier(BaseSearchCV, ClassifierMixin):
    def __init__(self, model, model_dir, lr=0.01, batch_size=3, epochs=20, scorings=None):
        self.model = model
        self.model_dir = model_dir
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.loss_file = os.path.join(self.model_dir, 'loss.npy')
        self.scaler = StandardScaler()
        self.scorings = scorings

    def fit(self, X, y):
        self.model.train()
        X_normalized = self.scaler.fit_transform(X)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        self.classes_ = torch.unique(y_tensor)
        dataset = TensorDataset(torch.tensor(X_normalized, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        loss_values = []
        for epoch in range(self.epochs):
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.optimizer.step()
                loss_values.append(loss.item())

        np.save(self.loss_file, np.array(loss_values))
        
        return self

    def predict(self, X):
        self.model.eval()
        inputs = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)
        return torch.round(outputs).cpu().numpy()

    def predict_proba(self, X):
        self.model.eval()
        inputs = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)
        return torch.cat((1 - outputs, outputs), axis=1).cpu().numpy()

    def score(self, X, y):
        predictions = self.predict(X)
        accuracy = (predictions == y).mean()
        return accuracy

    def _run_search(self, evaluate_candidates):
        results = super()._run_search(evaluate_candidates)
        if self.scorings is not None:
            for scorer_name in self.scorings:
                scorer = self.scorers_[scorer_name]
                results['mean_test_' + scorer_name] = scorer._score_func(self, self.cv_results_)
        return results
    
class ShapeLogger(BaseEstimator, TransformerMixin):
    def __init__(self, previous_step_name):
        self.previous_step_name = previous_step_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.info(f"FINISHED {self.previous_step_name.upper()}\n{X[:3]}")
        return X