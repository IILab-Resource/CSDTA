from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from SmilesEnumerator import SmilesEnumerator
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import MolStandardize
from typing import List
import joblib
from transformers import TrainingArguments
from sklearn import preprocessing

# /home/zhuyan/DeepDTAF-master/DeepDTAF_add_self_pkt/data/zy_training_pkt_2.csv
# /home/zhuyan/DeepDTAF-master/DeepDTAF_add_self_pkt/data/zy_validation_pkt_2.csv
# /home/zhuyan/DeepDTAF-master/DeepDTAF_add_self_pkt/data/zy_test_pkt_2.csv

residue_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']

aliphatic_residues_table = ['A', 'I', 'L', 'M', 'V']
aromatic_residues_table = ['F', 'W', 'Y']
polar_neutral_residues_table = ['C', 'N', 'Q', 'S', 'T']
acidic_charged_residues_table = ['D', 'E']
basic_charged_residues_table = ['H', 'K', 'R']

weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}

hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}

pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}

pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}


def normalize_dict(dic):
    max_ = dic[max(dic, key=dic.get)]
    min_ = dic[min(dic, key=dic.get)]
    interval = float(max_) - float(min_)
    
    for key in dic.keys():
        dic[key] = (dic[key] - min_) / interval
        
    dic['X'] = (max_ + min_) / 2.0 # For unknown 
    return dic

weight_table = normalize_dict(weight_table)
pka_table = normalize_dict(pka_table)
pkb_table = normalize_dict(pkb_table)
pkx_table = normalize_dict(pkx_table)
pl_table = normalize_dict(pl_table)
hydrophobic_ph2_table = normalize_dict(hydrophobic_ph2_table)
hydrophobic_ph7_table = normalize_dict(hydrophobic_ph7_table)
NUM_RES_PROPERTIES = 12

def residue_features(residue):
    numeric_properties = [weight_table[residue],
                          hydrophobic_ph2_table[residue], 
                          hydrophobic_ph7_table[residue],
                          pl_table[residue],
                          pka_table[residue], 
                          pkb_table[residue], 
                          pkx_table[residue]]
    binary_properties = [1 if residue in acidic_charged_residues_table else 0,
                         1 if residue in basic_charged_residues_table else 0, 
                         1 if residue in aromatic_residues_table else 0,
                         1 if residue in aliphatic_residues_table else 0, 
                         1 if residue in polar_neutral_residues_table else 0]
    return np.array(binary_properties + numeric_properties)

def extract_residue_features(protein_seq, MAX_SEQ_LEN):
    protein_property = np.zeros((MAX_SEQ_LEN, NUM_RES_PROPERTIES), dtype=np.float32)
    for i, ch in enumerate(protein_seq[:MAX_SEQ_LEN]):
        protein_property[i,] = residue_features(protein_seq[i]) # Get the chemical properties
    return protein_property

CHAR_SMI_SET = {"(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
                "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
                "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
                "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
                "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
                "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
                "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
                "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64}



CHAR_SMI_SET_LEN = len(CHAR_SMI_SET)
PT_FEATURE_SIZE = 40
CHARPROTSET = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
				"U": 19, "T": 20, "W": 21, 
				"V": 22, "Y": 23, "X": 24, 
				"Z": 25 }

CHARPROTLEN = 25



def label_sequence(line, MAX_SEQ_LEN):
    
    X = np.zeros(MAX_SEQ_LEN, dtype=np.int)

    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = CHARPROTSET[ch]

    return X

def label_smiles(line, max_smi_len):
    X = np.zeros(max_smi_len, dtype=np.int)
    for i, ch in enumerate(line[:max_smi_len]):
        X[i] = CHAR_SMI_SET[ch] 

    return X

class MyDataset(Dataset):
    def __init__(self, data_path, phase, max_seq_len, max_smi_len):
        data_path = Path(data_path)

        affinity = {}
        affinity_df = pd.read_csv(data_path / 'affinity_data.csv')
        for _, row in affinity_df.iterrows():
            affinity[row[0]] = row[1]
        self.affinity = affinity


        ligands_df = pd.read_csv(data_path / f"{phase}_smi_can.csv")
        ligands = {i["pdbid"]: i["can_smiles"] for _, i in ligands_df.iterrows()}
        self.smi = ligands
        self.max_smi_len = max_smi_len
        self.pdbids = ligands_df['pdbid'].values

        # seq_path = data_path / phase / 'global'
        # self.seq_path = sorted(list(seq_path.glob('*')))
        self.max_seq_len = max_seq_len

        prot_df = pd.read_csv(data_path / f"{phase}_seq_.csv")
        prots = {i["id"]: i["seq"] for _, i in prot_df.iterrows()}
        self.prots = prots
        if phase == 'test105' or phase == 'test71' :
            pkt = pd.read_csv(data_path / f"zy_{phase}_pkt.csv")
        else:
            pkt = pd.read_csv(data_path / f"zy_{phase}_pkt_2.csv")#kt_2_select_norm_.csv")#f"zy_training_pkt_2_select_norm_.csv")zy_{phase}_pkt_2_norm_end.csv#, index_col=0.values[:]
            disabled_features = ['F43', 'F44', 'F87','F88', 'F131', 'F132']
            pkt = pkt.drop(disabled_features, axis=1)
        
        a = pkt['pdbid'].apply(lambda x:x[:4]).tolist()
        pkt = pkt.drop(labels='pdbid',axis=1)
        # print(pkt.head())
        # scaler = preprocessing.MinMaxScaler()
        # pkt = pd.DataFrame(scaler.fit_transform(pkt.values.astype(np.float32)), columns=pkt.columns, index=pkt.index)
        # print(pkt.head())
        pkt['pdbid']=a  # 在列索引为2的位置插入一列,列名为:city，刚插入时不会有值，整列都是NaN
        pkt.set_index(['pdbid'], inplace = True)
         
        self.pkt = pkt
        
		
        comp900 = pd.read_csv(f"data/comp900_{phase}.csv")#kt_2_select_norm_.csv")#f"zy_training_pkt_2_select_norm_.csv")zy_{phase}_pkt_2_norm_end.csv#, index_col=0.values[:]
        # a = comp900['pdbid'].apply(lambda x:x[:4]).tolist()
        # comp900 = comp900.drop(labels='pdbid',axis=1)
        # scaler = preprocessing.MinMaxScaler()
        # comp900 = pd.DataFrame(scaler.fit_transform(comp900.values), columns=comp900.columns, index=comp900.index)
        # comp900['pdbid']=a 
        comp900.set_index(['pdbid'], inplace = True)
        self.comp900 = comp900
        self.length = len(self.smi)


    def __getitem__(self, idx):
        pdbid = self.pdbids[idx]
        # seq = self.seq_path[idx]

        # _seq_tensor = pd.read_csv(seq, index_col=0).drop(['idx'], axis=1).values[:self.max_seq_len]
        # seq_tensor = np.zeros((self.max_seq_len, PT_FEATURE_SIZE))
        # seq_tensor[:len(_seq_tensor)] = _seq_tensor
        
        aug_smile =   self.smi[pdbid]
        protseq = self.prots[pdbid]

        return (
                # seq_tensor.astype(np.float32),
                np.array(self.pkt.loc[pdbid], dtype=np.float32),
                np.array(self.comp900.loc[pdbid], dtype=np.float32),
                label_smiles(aug_smile, self.max_smi_len),
                label_sequence(protseq,self.max_seq_len),
                # extract_residue_features(protseq,self.max_seq_len ),
                np.array(self.affinity[pdbid], dtype=np.float32))

    def __len__(self):
        return self.length



class PretrainedDataset(Dataset):
    def __init__(self, data_path, phase, max_seq_len, max_smi_len, prots, drugs):
        data_path = Path(data_path)

        affinity = {}
        affinity_df = pd.read_csv(data_path / 'affinity_data.csv')
        for _, row in affinity_df.iterrows():
            affinity[row[0]] = row[1]
        self.affinity = affinity


        ligands_df = pd.read_csv(data_path / f"{phase}_smi_can.csv")
        ligands = {i["pdbid"]: i["can_smiles"] for _, i in ligands_df.iterrows()}
        self.smi = ligands
        self.max_smi_len = max_smi_len
        self.pdbids = ligands_df['pdbid'].values


        self.max_seq_len = max_seq_len

        prot_df = pd.read_csv(data_path / f"{phase}_seq_.csv")
        # prots = {i["id"]: i["seq"] for _, i in prot_df.iterrows()}
        # self.prots = prots
        if phase == 'test105' or phase == 'test71' :
            pkt = pd.read_csv(data_path / f"zy_{phase}_pkt.csv")
            disabled_features = ['F43', 'F44', 'F45', 'F46', 'F47','F48', 'F49', 'F50','F51','F52','F53','F54','F55', 'F56', 'F57','F58', 'F59', 'F60','F61','F62','F63','F64','F65'
            , 'F66', 'F67','F68', 'F69', 'F70','F71','F72','F73','F74','F75', 'F76', 'F77','F78', 'F79', 'F80','F81','F82','F83','F84','F85', 'F86', 'F87','F88', 'F89', 'F90','F91','F92','F93','F94','F95'
            , 'F96', 'F97','F98', 'F99', 'F100','F101','F102','F103','F104','F105', 'F106', 'F107','F108', 'F109', 'F110','F111','F112','F113','F114','F115'
            , 'F116', 'F117','F118', 'F119', 'F120','F121','F122','F123','F124','F125', 'F126']
            pkt = pkt.drop(disabled_features, axis=1)
        else:
            pkt = pd.read_csv(data_path / f"zy_{phase}_pkt_2.csv")#kt_2_select_norm_.csv")#f"zy_training_pkt_2_select_norm_.csv")zy_{phase}_pkt_2_norm_end.csv#, index_col=0.values[:]
            # disabled_features = ['F43', 'F44', 'F87','F88', 'F131', 'F132']
            disabled_features = ['F43', 'F44', 'F45', 'F46', 'F47','F48', 'F49', 'F50','F51','F52','F53','F54','F55', 'F56', 'F57','F58', 'F59', 'F60','F61','F62','F63','F64','F65'
            , 'F66', 'F67','F68', 'F69', 'F70','F71','F72','F73','F74','F75', 'F76', 'F77','F78', 'F79', 'F80','F81','F82','F83','F84','F85', 'F86', 'F87','F88', 'F89', 'F90','F91','F92','F93','F94','F95'
            , 'F96', 'F97','F98', 'F99', 'F100','F101','F102','F103','F104','F105', 'F106', 'F107','F108', 'F109', 'F110','F111','F112','F113','F114','F115'
            , 'F116', 'F117','F118', 'F119', 'F120','F121','F122','F123','F124','F125', 'F126', 'F127','F128', 'F129', 'F130','F131','F132']
   
            pkt = pkt.drop(disabled_features, axis=1)
        
        a = pkt['pdbid'].apply(lambda x:x[:4]).tolist()
        pkt = pkt.drop(labels='pdbid',axis=1)
        
        pkt['pdbid']=a  # 在列索引为2的位置插入一列,列名为:city，刚插入时不会有值，整列都是NaN
        pkt.set_index(['pdbid'], inplace = True)
         
        self.pkt = pkt
        
		
        comp900 = pd.read_csv(f"data/comp900_{phase}.csv")#kt_2_select_norm_.csv")#f"zy_training_pkt_2_select_norm_.csv")zy_{phase}_pkt_2_norm_end.csv#, index_col=0.values[:]
        comp900.set_index(['pdbid'], inplace = True)
        self.comp900 = comp900
        self.length = len(self.smi)

        self.prots = prots 
        self.drugs = drugs


    def __getitem__(self, idx):
        pdbid = self.pdbids[idx]
        aug_smile =   self.smi[pdbid]
        protseq = self.prots[pdbid]
        protseq = np.zeros((self.max_seq_len, 1024), dtype=np.float32)
        protseq[:self.prots[pdbid].shape[0]] = self.prots[pdbid]
        smile_emb = np.zeros((self.max_smi_len, 768), dtype=np.float32)
        drug_bert_emb = self.drugs[pdbid]
        if drug_bert_emb.shape[0] >  self.max_smi_len:
            smile_emb[:self.max_smi_len] = drug_bert_emb[:self.max_smi_len]
        else:
            smile_emb[:drug_bert_emb.shape[0]] = drug_bert_emb


        return (
                # seq_tensor.astype(np.float32),
                np.array(self.pkt.loc[pdbid], dtype=np.float32),
                np.array(self.comp900.loc[pdbid], dtype=np.float32),
                smile_emb,
                protseq,
                # label_smiles(aug_smile, self.max_smi_len),
                # label_sequence(protseq,self.max_seq_len),
                # extract_residue_features(protseq,self.max_seq_len ),
                np.array(self.affinity[pdbid], dtype=np.float32))

    def __len__(self):
        return self.length
    



class SeqDataset(Dataset):
    def __init__(self, data_path, phase, max_seq_len, max_smi_len):
        data_path = Path(data_path)

        affinity = {}
        affinity_df = pd.read_csv(data_path / 'affinity_data.csv')
        for _, row in affinity_df.iterrows():
            affinity[row[0]] = row[1]
        self.affinity = affinity


        ligands_df = pd.read_csv(data_path / f"{phase}_smi_can.csv")
        ligands = {i["pdbid"]: i["smiles"] for _, i in ligands_df.iterrows()}
        self.smi = ligands
        self.max_smi_len = max_smi_len

        prot_df = pd.read_csv(data_path / f"{phase}_seq_.csv")
        prots = {i["id"]: i["seq"] for _, i in prot_df.iterrows()}
        self.prots = prots
        self.max_seq_len = max_seq_len


        self.pdbids = ligands_df['pdbid'].values

        self.length = len(self.smi)


    def __getitem__(self, idx):
        pdb_id = self.pdbids[idx]
        smile =   self.smi[pdb_id]
        protseq = self.prots[pdb_id]
        


        return ( label_sequence(protseq,self.max_seq_len),
                label_smiles(smile, self.max_smi_len),
                # extract_residue_features(protseq,self.max_seq_len ),
                np.array(self.affinity[pdb_id], dtype=np.float32))
 

    def __len__(self):
        return self.length



# class SeqEmbDataset(Dataset):
#     def __init__(self, data_path, prots,phase, max_seq_len, max_smi_len):
#         data_path = Path(data_path)

#         affinity = {}
#         affinity_df = pd.read_csv(data_path / 'affinity_data.csv')
#         for _, row in affinity_df.iterrows():
#             affinity[row[0]] = row[1]
#         self.affinity = affinity


#         ligands_df = pd.read_csv(data_path / f"{phase}_smi_can.csv")
#         ligands = {i["pdbid"]: i["smiles"] for _, i in ligands_df.iterrows()}
#         ligands_can = {i["pdbid"]: i["smiles"] for _, i in ligands_df.iterrows()}
#         self.smi = ligands
#         self.max_smi_len = max_smi_len
#         self.smi_can = ligands_can
        
#         self.prots = prots
#         # self.drugs = drugs
#         self.max_seq_len = max_seq_len
#         self.pdbids = ligands_df['pdbid'].values

#         self.length = len(self.smi)
#         self.phase = phase


#     def __getitem__(self, idx):
#         pdb_id = self.pdbids[idx]
#         # if self.phase == 'training':
#         #     if np.random.uniform() > 0.5:
#         #         smile =   self.smi[pdb_id]
#         #     else:
#         #         smile =   self.smi_can[pdb_id]

#         # else:
#         smile =   self.smi[pdb_id]
#         protseq = np.zeros((self.max_seq_len, 1024), dtype=np.float32)
#         protseq[:self.prots[pdb_id].shape[0]] = self.prots[pdb_id]
#         # smile_emb = np.zeros((self.max_smi_len, 768), dtype=np.float32)
#         # drug_bert_emb = self.drugs[pdb_id]
#         # if drug_bert_emb.shape[0] >  self.max_smi_len:
#         #     smile_emb[:self.max_smi_len] = drug_bert_emb[:self.max_smi_len]
#         # else:
#         #     smile_emb[:drug_bert_emb.shape[0]] = drug_bert_emb

#         return ( protseq,
#                 # smile_emb,
#                 label_smiles(smile, self.max_smi_len),
#                 np.array(self.affinity[pdb_id], dtype=np.float32))
 

#     def __len__(self):
#         return self.length




class SeqEmbDataset(Dataset):
    def __init__(self, data_path, prots, drugs ,phase, max_seq_len, max_smi_len):
        data_path = Path(data_path)

        affinity = {}
        affinity_df = pd.read_csv(data_path / 'affinity_data.csv')
        for _, row in affinity_df.iterrows():
            affinity[row[0]] = row[1]
        self.affinity = affinity


        ligands_df = pd.read_csv(data_path / f"{phase}_smi_can.csv")
        ligands = {i["pdbid"]: i["smiles"] for _, i in ligands_df.iterrows()}
        ligands_can = {i["pdbid"]: i["smiles"] for _, i in ligands_df.iterrows()}
        self.smi = ligands
        self.max_smi_len = max_smi_len
        self.smi_can = ligands_can
        
        self.prots = prots
        self.drugs = drugs
        self.max_seq_len = max_seq_len
        self.pdbids = ligands_df['pdbid'].values

        self.length = len(self.smi)
        self.phase = phase


    def __getitem__(self, idx):
        pdb_id = self.pdbids[idx]
        
        smile =   self.smi[pdb_id]
        protseq = np.zeros((self.max_seq_len, 1024), dtype=np.float32)
        protseq[:self.prots[pdb_id].shape[0]] = self.prots[pdb_id]
        smile_emb = np.zeros((self.max_smi_len, 384), dtype=np.float32)
        drug_bert_emb = self.drugs[pdb_id]
        if drug_bert_emb.shape[0] >  self.max_smi_len:
            smile_emb[:self.max_smi_len] = drug_bert_emb[:self.max_smi_len]
        else:
            smile_emb[:drug_bert_emb.shape[0]] = drug_bert_emb

        return ( protseq,
                smile_emb,
                # label_smiles(smile, self.max_smi_len),
                np.array(self.affinity[pdb_id], dtype=np.float32))
 

    def __len__(self):
        return self.length

# data_path = '/home/zhuyan/pdbbind_v2016_refined/data/'
# phase = 'training'
# max_seq_len = 1000  
# max_smi_len = 150
# data = SeqEmbDataset(data_path, phase, max_seq_len, max_smi_len)
# print(next(iter(data)))
# print(data.max_smi_len)
# X = new_label_smiles('O=C(c1nnc(o1)C(C)(C)C)[C@@H]1CCC[NH2+]1', 100)
# print(X)