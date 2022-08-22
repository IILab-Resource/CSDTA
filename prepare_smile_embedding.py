from tqdm import tqdm 
import pandas as pd 
import joblib
# from dgllife.utils import smiles_to_bigraph
from dgl import save_graphs, load_graphs
from joblib import Parallel, delayed, cpu_count
from functools import partial
from subprocess import PIPE, run
from transformers import AutoTokenizer, AutoModel

def out(command):
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    return result.stdout
chemical_tokenizer = AutoTokenizer.from_pretrained('DeepChem/ChemBERTa-77M-MLM')
chemberta = AutoModel.from_pretrained('DeepChem/ChemBERTa-77M-MLM')


drug_embedding = {}
for phase_name in [ 'training', 'validation','test', 'test105', 'test71']:
    smi_df = pd.read_csv( 'data/' + phase_name+ '_smi_can.csv')
    smiles = {i["pdbid"]: i["can_smiles"] for _, i in smi_df.iterrows()}
    can_s = []
    for pid, s in tqdm(smiles.items()):
        tokens = chemical_tokenizer(s, return_tensors='pt')
        output = chemberta(**tokens)
        embedding = output.last_hidden_state.detach().numpy()[0,1:-1]
        drug_embedding[pid] = embedding

joblib.dump(drug_embedding, 'data/drug_emb_all.job')       
# smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True)
# for phase_name in [ 'training', 'validation','test', 'test105', 'test71']:
#     smi_df = pd.read_csv( 'data/' + phase_name+ '_smi.csv')
#     smiles = {i["pdbid"]: i["smiles"] for _, i in smi_df.iterrows()}
#     can_s = []
#     for pid, s in tqdm(smiles.items()):
#         can_s.append(out('obabel -:"' + s +'"  -ocan' ).replace('"',' ').strip())

#     smi_df['can_smiles'] = can_s
#     smi_df.to_csv('data/' + phase_name+ '_smi_can.csv')
        # print(can_s)
        # print('---------')
        # smiles_to_graph(can_s, node_featurizer=args['node_featurizer'], edge_featurizer=args['edge_featurizer'])
        
            # prot_bert_embedding[pid] = embeddings['full'][0]
# joblib.dump(prot_bert_embedding, 'data/prot_emb_all.job')
# # print(prot_bert_embedding[pid].shape)
# # joblib.dump(seen_sequences, 'data/seq2pid.job')
