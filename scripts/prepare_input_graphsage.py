from DeepLinkPrediction.InteractionNetwork import UndirectedInteractionNetwork
from networkx.readwrite import json_graph
import pandas as pd
import networkx as nx
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--ppi_scaffold', required=True, type=str)
parser.add_argument('--disease', required=True, type=str)
parser.add_argument('--screening', required=True, type=str)
args = parser.parse_args()

BASE_PATH ='/'.join(os.getcwd().split('/')[:-1])

ppi_scaffold = args.ppi_scaffold
disease = args.disease
screening = '' if args.screening == 'rnai' else '_crispr'


heterogeneous_network = pd.read_csv(BASE_PATH+f"/heterogeneous_networks/"
                                    f"{ppi_scaffold}_{disease.replace(' ','_')}_dependencies{screening}.csv")
heterogeneous_network_obj = UndirectedInteractionNetwork(heterogeneous_network)

G = heterogeneous_network_obj.getnxGraph(return_names=False)
node_attributes = {n: {"test": False, "val": False, "id": n} for n in heterogeneous_network_obj.nodes}
nx.set_node_attributes(G, values=node_attributes)

d = json_graph.node_link_data(G)

save_fp = BASE_PATH+f"/GraphSAGE_input{screening}/{ppi_scaffold}/{disease.replace(' ','_')}"
if not os.path.isdir(save_fp):
    os.makedirs(save_fp)


with open(save_fp+f"/input-G.json", 'w') as fp:
    json.dump(d, fp)



with open(save_fp+f"/input-id_map.json", 'w') as fp:
    json.dump({k: k for k in heterogeneous_network_obj.int2gene.keys()}, fp)

class_map = {k: [1] for k in heterogeneous_network_obj.int2gene.keys()}

with open(save_fp+f"/input-class_map.json", 'w') as fp:
    json.dump(class_map, fp)
