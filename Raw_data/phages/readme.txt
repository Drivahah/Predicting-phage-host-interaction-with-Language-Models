This dataset contains valuable information regarding the phages mentioned in the Maffei et al. paper. It encompasses a wide range of data that can be instrumental for further research and analysis. Below is a breakdown of the contents included in this dataset:

    Phage Proteomes: basel_proteome.fasta
        The dataset includes the full proteome of each phage mentioned in the Maffei et al. paper.
        This covers a total of 248 phages that have been sequenced and annotated.
        Specifically, this includes 69 phages from the Basel collection.

    Protein Labeling (File: phage_proteins_label.csv):

        Here, we've employed three distinctive NN-based techniques for protein labeling:
            PhaNNS
            PhageRBPdetect
            ESM-based method developed by Yumeng

        Interpreting the Labels:

            Columns 1-11 present the label scores for PhaNNs, with values approaching 10 signifying high confidence.
            Column 12 provides PhaNNs' confidence level.
            Column 13 indicates PhageRBPdetect predictions: 1 denotes an RBP prediction, while 0 signifies otherwise.
            Column 14 offers PhageRBPdetect scores, with values close to 1 signifying strong confidence.
            Column 15 presents the ESM-based label.
            Column 16 features 1 if the label in Column 15 is "tail_fiber."

    Supplementary Table (File: Basel_receptors.csv):

        This supplementary table is a comprehensive repository of information concerning outer membrane receptors associated with each of the 69 Basel phages.
        The data is extracted from the Maffei et al. paper's S1 data.
