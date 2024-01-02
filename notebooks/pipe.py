import os
import argparse
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
import joblib
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN
import logging
from datetime import datetime

# Set workiiing directory to file location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Import custom modules
import sys
sys.path.append("../src")
from PipelineFunctions import load_data, flatten_data, ProtT5Embedder, ProtXLNetEmbedder, SequentialEmbedder

# parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--embedder', type=str, choices=['prott5', 'protxlnet'], default='prott5', help='Embedder to use')
parser.add_argument('--fine_tune', action='store_true', help='Whether to fine-tune the embedder')
parser.add_argument('--oversampling', type=str, choices=['smote', 'adasyn', 'none'], default='none', help='Oversampling technique to use for imbalanced data')
parser.add_argument('--estimator', type=str, choices=['logreg', 'rf', 'transformer'], default='logreg', help='Final estimator to use')
parser.add_argument('--train', action='store_true', help='Whether to train the whole model')
parser.add_argument('--grid_search', action='store_true', help='Whether to perform grid search for the pipeline')
parser.add_argument('--param_grid', type=str, default='{}', help='Parameter grid for the grid search as a string')
parser.add_argument('--predict', type=str, default=None, help='Specify the pre-trained model to use to predict on the given data')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for fine-tuning the embedder')
parser.add_argument('--steps', type=int, default=10, help='Number of steps to train the embedder before freezing the parameters')
parser.add_argument('--batch_size', type=int, default=3, help='Batch size for embedding the data')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for the models')
parser.add_argument('--debug', action='store_true', help='Whether to enable debug logging')
parser.add_argument('--quick', action='store_true', help='Run only on the first batch')
parser.add_argument('--SP', type=str, choices=['parallel', 'sequential'], default='sequential', help='Whether to embed the sequences in parallel or sequentially')
args = parser.parse_args()

# Define log file and add an empty line
LOG_FILENAME = 'log.txt'
with open(LOG_FILENAME, 'a') as f:
    f.write('\n')

# Create a logger object
logger = logging.getLogger('Pipeline')
if args.debug:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(LOG_FILENAME, 'a')
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.info('NEW RUN')

# Log the pipeline options
logger.info(f'Pipeline args: \
    embedder={args.embedder}, \
    fine_tune={args.fine_tune}, \
    estimator={args.estimator}, \
    train={args.train}, \
    grid_search={args.grid_search}, \
    param_grid={args.param_grid}, \
    predict={args.predict}, \
    epochs={args.epochs}, \
    steps={args.steps}, \
    batch_size={args.batch_size}, \
    device={args.device}, \
    debug={args.debug}, \
    quick={args.quick}, \
    SP={args.SP}')

# PIPELINE
# Define the embedder
if args.embedder == 'prott5':
    embedder_phage = ProtT5Embedder('Rostlab/prot_t5_xl_half_uniref50-enc', fine_tune=args.fine_tune, device=args.device, debug=args.debug)
    embedder_bacteria = ProtT5Embedder('Rostlab/prot_t5_xl_half_uniref50-enc', fine_tune=args.fine_tune, device=args.device, org='bacteria', debug=args.debug)
elif args.embedder == 'protxlnet':
    embedder_phage = ProtXLNetEmbedder('Rostlab/prot_xlnet', fine_tune=args.fine_tune, device=args.device, debug=args.debug)
    embedder_bacteria = ProtXLNetEmbedder('Rostlab/prot_xlnet', fine_tune=args.fine_tune, device=args.device, org='bacteria', debug=args.debug)

# Sequential or parallel embedder for phage and bacteria
if args.SP == 'sequential':
    pair_embedder = SequentialEmbedder(embedder_phage, embedder_bacteria)
elif args.SP == 'parallel':
    column_indices = {'sequence_phage': 0, 'sequence_k12': 1}
    pair_embedder = ColumnTransformer(transformers=[
        ('embedder_phage', embedder_phage, [column_indices['sequence_phage']]),
        ('embedder_bacteria', embedder_bacteria, [column_indices['sequence_k12']]),
    ], remainder='drop')  # drop any columns not specified in transformers

# Define oversampling technique
if args.oversampling == 'smote':
    pipe1 = ImbPipeline([('pair_embedder', pair_embedder),
                         ('oversampling', SMOTE())])
elif args.oversampling == 'adasyn':
    pipe1 = ImbPipeline([('pair_embedder', pair_embedder),
                         ('oversampling', ADASYN())])
else:
    pipe1 = Pipeline([('pair_embedder', pair_embedder)])

# Define the final estimator
if args.estimator == 'logreg':
    estimator = LogisticRegression()
    pipe2 = Pipeline([('estimator', estimator)])
elif args.estimator == 'rf':
    estimator = RandomForestClassifier()
    pipe2 = Pipeline([('flatten', FunctionTransformer(flatten_data)), 
                      ('estimator', estimator)])


# Load data
INPUT_FOLDER = os.path.join('..', 'data', 'interim')
DATA_PATH = os.path.join(INPUT_FOLDER, '2_model_df.pkl')
X, y = load_data(DATA_PATH, args.quick, args.debug)
logger.info(f'LOAD DATA\nData shape: X={X.shape}, y={y.shape}')
logger.debug(f'X[:5]:\n {X[:5]}\ny[:5]:\n {y[:5]}')

scoring = ('accuracy', 'precision', 'recall', 'f1', 'roc_auc')
refit = 'f1'

# (Nested) cross-validation
param_grid = eval(args.param_grid) # convert the string to a dictionary
splits = {'inner': 3, 'outer': 3}
MODELS_DIR = os.path.join('..', 'models')
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)
if args.train:
    logger.info('TRAINING THE WHOLE MODEL')
    logger.debug('Splits: ' + str(splits))

    best_outer_score = float('-inf')  # Initialize with a very small value for maximization
    best_model = None
    best_params = None

    # Outer cross-validation
    outer_scores = []
    outer_cv = StratifiedKFold(n_splits=splits['outer'], shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(outer_cv.split(X, y)):
        logger.debug(f'Outer fold {fold+1}')
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        logger.debug(f'X_train[:5]:\n {X_train[:5]}\ny_train[:5]:\n {y_train[:5]}\nX_test[:5]:\n {X_test[:5]}\ny_test[:5]:\n {y_test[:5]}')

        # Embed and resample the training data
        X_train = pipe1.named_steps['pair_embedder'].transform(X_train, batch_size=args.batch_size)
        logger.debug(f'FINISHED EMBEDDING:\nData shape after embedding: X_train={X_train.shape}, y_train={y_train.shape}\nX_train[:5]:\n {X_train[:5]}\ny_train[:5]:\n {y_train[:5]}')
        X_train, y_train = pipe1.named_steps['oversampling'].fit_resample(X_train, y_train)
        logger.debug(f'FINISHED RESAMPLING\nX_train[:5]:\n {X_train[:5]}\ny_train[:5]:\n {y_train[:5]}')
        logger.debug(f'Data shape after resampling: X_train={X_train.shape}, y_train={y_train.shape}')

        # Embed the test data
        X_test = pipe1.named_steps['pair_embedder'].transform(X_test, batch_size=args.batch_size)
        logger.debug(f'FINISHED EMBEDDING:\nData shape after embedding: X_test={X_test.shape}, y_test={y_test.shape}\nX_test[:5]:\n {X_test[:5]}\ny_test[:5]:\n {y_test[:5]}')

        # Grid search and inner CV
        if args.grid_search:
            '''
            Inside grid.fit, all the hyperparameters combinations are tested.
            For each combination, a stratified CV is performed on the training set. 
            Stratified means that the proportion of the classes is preserved in each fold.
            The combination with the best socre (folds average) is selected.
            The selected combination of hyperparameters is then used to fit the pipeline on the whole training set.
            This model is evaluated on the test set (aka 1 fold of the outer CV).
            '''
            logger.debug('PERFORMING GRID SEARCH')
            inner_cv = StratifiedKFold(n_splits=splits['inner'], shuffle=True, random_state=42)
            grid = GridSearchCV(pipe2, param_grid, cv=inner_cv, scoring=scoring, refit=refit, verbose=3, n_jobs=-1)
            grid.fit(X_train, y_train)
            score = grid.score(X_test, y_test)
            logger.debug(f'Best parameters: {grid.best_params_}')
        # Fit the pipeline on the training data without grid search and inner CV
        else:
            pipe2.fit(X_train, y_train)
            score = pipe2.score(X_test, y_test)

        outer_scores.append(score)
        logger.debug(f'Score: {score}')

        # Update best model
        if score > best_outer_score:
            best_outer_score = score
            best_model = grid.best_estimator_
            if args.grid_search:
                best_params = grid.best_params_

    # Save the best model
    if best_model is not None:
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        model_name = f'Model_{timestamp}.pkl'
        joblib.dump(best_model, os.path.join(MODELS_DIR, model_name))
        if args.grid_search:
            logger.debug(f'Best parameters across all outer folds: {best_params}')
            # Save the best parameters to a file
            with open('best_parameters.txt', 'a') as f:
                f.write(f'Model: {model_name}\n')
                f.write(f'Embedder: {args.embedder}\n')
                f.write(f'Fine-tuning: {"Yes" if args.fine_tune else "No"}\n')
                if args.fine_tune:
                    f.write('Epochs and steps for fine-tuning: ' + str(args.epochs) + ' ' + str(args.steps) + '\n')
                f.write(f'Oversampling: {args.oversampling}\n')
                f.write(f'Estimator: {args.estimator}\n')
                f.write(f'Hyperparameters: {best_params}\n')
                f.write(f'Outer score: {best_outer_score}\n')
                f.write('Outer splits: ' + str(splits['outer']) + '\n')
                if args.grid_search:
                    f.write(f'Inner splits: {splits["inner"]}\n')
                f.write('\n')
        logger.debug(f'Best score across all outer folds: {best_outer_score}')
        logger.debug(f'Best model saved as: {model_name}')
    else:
        logger.warning('No best model found.')

    # Log the results of nested cross-validation
    logger.debug(f'Nested cross-validation scores: {outer_scores}\nMean score: {np.mean(outer_scores)}')

PREDICTIONS_DIR = os.path.join('..', 'data', 'predictions')
if not os.path.exists(PREDICTIONS_DIR):
    os.makedirs(PREDICTIONS_DIR)
if args.predict:
    loaded_model = joblib.load(os.path.join(MODELS_DIR, args.predict))
    predictions = loaded_model.predict(X)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    predictions_file = f'predictions_{args.predict}_{timestamp}.txt'
    np.savetxt(os.path.join(PREDICTIONS_DIR, predictions_file), predictions, fmt='%s')
