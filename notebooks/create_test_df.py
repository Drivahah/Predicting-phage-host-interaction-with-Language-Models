# Create a pandas df with short test proteic sequences

import pandas as pd
import numpy as np
import random
import string
import os

# Create a list of random sequences
def random_seq(length):
    return ''.join(random.choice(string.ascii_uppercase) for i in range(length))

# Make a df with three columns, sequence_phage, sequence_k12, and pair
# It has 100 rows
# The sequence_phage column has random sequences of length between 5 and 25
# The sequence_k12 column has random sequences of length between 5 and 25
# The pair column has random 1 and 0

df = pd.DataFrame({
    'sequence_phage': [random_seq(random.randint(5,25)) for i in range(100)],
    'sequence_k12': [random_seq(random.randint(5,25)) for i in range(100)],
    'pair': np.random.randint(2, size=100)
})

# Define savedir data/interim using os
savedir = os.path.join('data', 'interim')

# Save as pickle
df.to_pickle(os.path.join(savedir, 'test_df.pkl'))