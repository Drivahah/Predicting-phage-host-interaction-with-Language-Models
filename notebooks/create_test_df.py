# Create a pandas df with short test proteic sequences

import pandas as pd
import numpy as np
import random
import string
import os

# Create a list of random sequences
def random_seq(length):
    return ''.join(random.choice(string.ascii_uppercase) for i in range(length))

# Make a df with two columns, phage and k12
# It has 100 rows
# The phage column has random sequences of length between 5 and 25
# The k12 column has random sequences of length between 5 and 25

df = pd.DataFrame({'phage': [random_seq(random.randint(5,25)) for i in range(100)],
                     'k12': [random_seq(random.randint(5,25)) for i in range(100)]})

# Define savedir data/interim using os
savedir = os.path.join('data', 'interim')

# Save as pickle
df.to_pickle(os.path.join(savedir, 'test_df.pkl'))