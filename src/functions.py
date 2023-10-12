import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def PhaNNS_thresholds(conf_thresholds: np.arange, tail_fiber_thresholds: np.arange, percentage=False, protein_NN=None):
    """
    This function takes two lists of thresholds and returns a heatmap of the number or percentage of survivors for each combination of thresholds.
    The function takes the following arguments:
    - conf_thresholds: np.arange
    - tail_fiber_thresholds: np.arange
    - percentage: bool
    """
    # Create a table to store the results
    df = pd.DataFrame(index=conf_thresholds, columns=tail_fiber_thresholds)

    # Compute the number or percentage of survivors for each combination of thresholds
    for conf in conf_thresholds:
        for tf in tail_fiber_thresholds:
            survivors = protein_NN[(protein_NN['confidence'] > conf) & (protein_NN['tail_fiber'] > tf)]
            if percentage:
                df.loc[conf, tf] = len(survivors) / len(protein_NN) * 100
            else:
                df.loc[conf, tf] = len(survivors)

    # Convert the data type of the DataFrame to float
    df = df.apply(pd.to_numeric)

    # Plot the results as a heatmap for percentage
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(df, cmap='YlGnBu')

    # Add labels and ticks
    ax.set_xticks(np.arange(len(tail_fiber_thresholds)))
    ax.set_yticks(np.arange(len(conf_thresholds)))
    ax.set_xticklabels(tail_fiber_thresholds)
    ax.set_yticklabels(conf_thresholds)
    ax.set_xlabel('Tail Fiber Threshold')
    ax.set_ylabel('Confidence Threshold')
    if percentage:
        ax.set_title('Percentage of Survivors')
    else:
        ax.set_title('Number of Survivors')
    plt.colorbar(im)
    plt.show()

    display(df)


def phageRBP_thresholds(thresholds: np.arange, protein_NN=None):
    """
    This function takes a list of thresholds and returns a table of the number and percentage of survivors for each threshold.
    The function takes the following arguments:
    - thresholds: np.arange
    """
    # Create a table to store the results
    df = pd.DataFrame(index=thresholds, columns=['survivors', 'percentage'])

    # Compute the number and percentage of survivors for each threshold which have a prediction of 1
    for threshold in thresholds:
        survivors = protein_NN.loc[(protein_NN['PhageRBPdetect_score'] > threshold) & (protein_NN['PhageRBPdetect_prediction'] == 1)]
        df.loc[threshold, 'survivors'] = len(survivors)
        df.loc[threshold, 'percentage'] = len(survivors) / len(protein_NN) * 100

    # Convert the data type of the DataFrame to float
    df = df.apply(pd.to_numeric)

    display(df)


def NN_thresholds(thresholds: dict, protein_NN=None):
    """
    This function takes a dictionary of thresholds and returns a subset of protein_NN with the survivors.
    The dictionary must contain the following keys:
    - phanns_tf
    - phanns_conf
    - phagerbp

    The dictionary must contain the following values:
    - float
    - float
    - float

    The function returns a subset of protein_NN.
    """

    # Subset protein_NN to relevant columns
    df = protein_NN[['seqID_phage', 'tail_fiber', 'confidence', 'PhageRBPdetect_prediction', 'PhageRBPdetect_score', 'ESM_based_fiber_prediction']]

    # Subset to the thresholds
    df = df.loc[(df['tail_fiber'] > thresholds['phanns_tf']) &
                (df['confidence'] > thresholds['phanns_conf']) & 
                (df['PhageRBPdetect_score'] > thresholds['phagerbp']) &
                (df['ESM_based_fiber_prediction'] == 1)]
    
    # Return the subset
    return df


def random_concatenation(df_left, df_right, n_entries=3000, exclude_df=None):
    """
    This function takes two dataframes and returns a dataframe with n random pairings of the two dataframes without duplicates.
    The function takes the following arguments:
    - df_left: dataframe
    - df_right: dataframe
    - n_entries: int
    """
    # Select n random entries from each dataframe with replacement
    df_left_sample = df_left.sample(n_entries, replace=True)
    df_right_sample = df_right.sample(n_entries, replace=True)
    
    # Concatenate the selected entries into a new dataframe
    pairings = pd.concat([df_left_sample.reset_index(drop=True), df_right_sample.reset_index(drop=True)], axis=1)

    # Add the exclude_df to the pairings
    if exclude_df is not None:
        n_rows_exclude = len(exclude_df)
        pairings = pd.concat([exclude_df, pairings], axis=0)

    # Drop duplicates
    while len(pairings) != len(pairings.drop_duplicates()):
        new_samples = len(pairings) - len(pairings.drop_duplicates())
        pairings.drop_duplicates(inplace=True)
        df_left_sample = df_left.sample(new_samples, replace=True)
        df_right_sample = df_right.sample(new_samples, replace=True)
        new_pairings = pd.concat([df_left_sample.reset_index(drop=True), df_right_sample.reset_index(drop=True)], axis=1)
        pairings = pd.concat([pairings, new_pairings], axis=0)

    # Remove the exclude_df from the pairings
    if exclude_df is not None:
        pairings = pairings.iloc[n_rows_exclude:, :]

    # Reset index
    pairings.reset_index(drop=True, inplace=True)

    # Return the resulting dataframe
    return pairings