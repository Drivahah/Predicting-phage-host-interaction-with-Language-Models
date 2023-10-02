import pandas as pd

class Header():
    def __init__(self, header='', organism='phage'):
        self.header_str = str(header)
        self.header_dict = dict()

        organism = organism.lower()

        if self.header_str != '':
            self.read_header()
            self.make_df()
    
    def read_header(self):
        if organism == 'phage':
            self.id, self.tags = self.header_str.split(' ', 1)
            self.tags = self.tags.split('] [')
            self.tags = [tag.strip('[]') for tag in self.tags]
            
            # Populate header_dict
            self.header_dict['seqID'] = self.id
            for tag in self.tags:
                tag = tag.split('=')
                self.header_dict[tag[0]] = tag[1]

        elif organism == 'k12':
            # Define a function that takes a string as an argument and returns a dictionary
            def parse_string(string):
                # Split the string by spaces
                parts = string.split()
                # Initialize an empty dictionary
                header_dict = {}
                # Loop through the parts of the string
                for part in parts:
                    # If the part contains an equal sign, split it by the equal sign and assign the key and value to the dictionary
                    if "=" in part:
                        key, value = part.split("=")
                        header_dict[key] = value
                    # Else, check if the part is the first one, which is the id
                    elif part == parts[0]:
                        header_dict["id"] = part
                    # Else, assume the part is the name and append it to the dictionary with the key "name"
                    else:
                        header_dict["name"] = header_dict.get("name", "") + part + " "
                # Return the dictionary
                return header_dict
            
            # Adapt previous function to the method
            self.header_dict = parse_string(self.header_str)

        else:
            raise ValueError(f'Invalid organism: {organism}')

    def make_df(self):
        self.df = pd.DataFrame(self.header_dict, index=[0])


class Entry():
    def __init__(self, entry, organism):
        self.entry_str = str(entry)
        self.sequence = str()

        self.read_entry()
        self.make_df()

    def read_entry(self):
        self.entry_list = self.entry_str.split('\n', 1)
        self.header = Header(self.entry_list[0], organism)
        try:
            self.sequence = self.entry_list[1].replace('\n', '')
        except IndexError:
            raise IndexError(f'No sequence found in entry: {self.header.id}\nHeader: {self.header.header_str}\nSequence: {self.sequence}')
        


        #  Check that the sequence contains only valid aminoacids
        FASTA_AA = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ*-'  #https://en.wikipedia.org/wiki/FASTA_format
        for aa in self.sequence:
            if aa not in FASTA_AA:
                raise ValueError(f'Invalid aminoacid in sequence: -{aa}-')
            
    def make_df(self):
        self.df = pd.DataFrame(self.header.header_dict, index=[0])
        self.df['sequence'] = self.sequence

class fasta():
    def __init__(self, file, organism):
        self.file_path = file
        self.entries = list()

        self.read_fasta()
        self.make_df()

    def read_fasta(self):
        with open(self.file_path, 'r') as f:
            self.fasta_list = f.read().split('\n>')
        self.fasta_list[0] = self.fasta_list[0].strip('>')
        
        for entry in self.fasta_list:
            self.entries.append(Entry(entry, organism))

    def make_df(self):
        self.df = pd.concat([entry.df for entry in self.entries])
        self.df.reset_index(drop=True, inplace=True)



