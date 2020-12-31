"""This program creates the Propagated SPOKE Entry Vectors using GeneLab fold change data
Usage:
  make_omop_psevs.py [-b=N] [-w=N] [-i=NAME] [-d=NAME] [-v=NAME] [-s=NAME] [-o=NAME]
  make_omop_psevs.py -h | --help

Options:
  -h --help     Show this screen.
  -b=N     Prob of random jump [default: 0.1].
  -w=N     Number of workers [default: 12].
  -i=NAME     Input filename [default: GLDS-4_array_differential_expression.tsv].
  -d=NAME     Input directory [default: GeneLab_for_SPOKE/].
  -v=NAME     Version directory [default: V2/].
  -s=NAME     SPOKE directory [default: spoke_v_2/].
  -o=NAME     Output filename [default: GLDS-4_ranks_and_rank_by_type_for_meta].
"""
from docopt import docopt

arguments = docopt(__doc__)

import numpy as np
from collections import Counter
import os
import pandas as pd
import io
import math
import time
import multiprocessing as mp
import itertools 
import requests
import re
from scipy import stats


# Gene lab import directory
input_directory=arguments['-d']
# spoke directory 
spoke_directory = arguments['-s']
# pre computed psev directory
gene_psev_directory=spoke_directory+'gene_psevs/'
# save directory
version=arguments['-v']
# fc file
diff_file = arguments['-i']
# output filename
output_filename = arguments['-o']
# probability of random jump
probability_random_jump=float(arguments['-b'])
# number of jobs
n_jobs=int(arguments['-w'])

save_str = '_'.join(str(probability_random_jump).split('.'))

gene_indices_df, exp_df = pd.DataFrame(), pd.DataFrame()

samples, avg_rank, std_rank = [],[],[]

def load_gene_indices_df(node_list):
	gene_indices_df = pd.read_csv(gene_psev_directory+'gene_group_%s.tsv' % 0, sep='\t', header=0, index_col=False)
	gene_indices_df.loc[:,'Round'] = 0
	gene_indices_df.loc[:,'round_index'] = np.arange(len(gene_indices_df))
	for group_index in np.arange(1, 20):
		df = pd.read_csv(gene_psev_directory+'gene_group_%s.tsv' % group_index, sep='\t', header=0, index_col=False)
		df.loc[:,'Round'] = group_index
		df.loc[:,'round_index'] = np.arange(len(df))
		gene_indices_df = pd.concat((gene_indices_df, df), axis=0)
	gene_indices_df.loc[:,'Node'] = [node_list[i] for i in gene_indices_df.node_2_index.values]
	return gene_indices_df

def get_order_rank_vector(test_vector, ground_truth_order):
	temp = test_vector.argsort()
	ranks = np.empty(len(test_vector), int)
	ranks[temp] = np.arange(len(test_vector))
	ranks = ranks[ground_truth_order]
	return ranks

def get_rank_by_type_vector(vector, type_list):
	rank_list = np.zeros(len(type_list))
	for node_type in set(type_list):
		node_type_index = np.arange(len(type_list))[type_list==node_type]
		rank_list[node_type_index]=len(node_type_index)-get_order_rank_vector(vector[node_type_index], np.arange(len(node_type_index))) # low is top
	return rank_list

def get_rank_matrix(matrix, type_list, rank_by_type):
	if rank_by_type==True:
		return np.array([get_rank_by_type_vector(row, type_list) for i, row in enumerate(matrix)])
	else:
		return np.array([get_order_rank_vector(row, np.arange(len(row))) for i, row in enumerate(matrix)])

def get_mean_in_par(current_round):
	current_gene_indices = gene_indices_df[gene_indices_df.Round==current_round].round_index.values
	return np.sum(np.load(gene_psev_directory+'raw_psev_%s_gene_group_%s_sparse.npy' % (save_str, current_round), allow_pickle=False)[current_gene_indices], axis=0)

def get_std_in_par(current_round):
	current_gene_indices = gene_indices_df[gene_indices_df.Round==current_round].round_index.values
	return np.sum((np.load(gene_psev_directory+'raw_psev_%s_gene_group_%s_sparse.npy' % (save_str, current_round), allow_pickle=False)[current_gene_indices]-avg_rank)**2, axis=0)

def get_mean_zscore_rank_exp(current_round):
	current_gene_indices = gene_indices_df[(gene_indices_df.Round==current_round)&(gene_indices_df.seen==True)].round_index.values
	psev_matrix = get_rank_matrix(np.nan_to_num((np.load(gene_psev_directory+'raw_psev_%s_gene_group_%s_sparse.npy' % (save_str, current_round), allow_pickle=False)[current_gene_indices] - avg_rank)/std_rank), [], False)
	sample_psev = np.dot(exp_df[samples][(exp_df.Round==current_round)&(exp_df.round_index.isin(current_gene_indices))].values.T, psev_matrix)
	del psev_matrix
	return sample_psev

def get_any_func_in_par(func, input_vals, n_jobs=4):
	p = mp.Pool(n_jobs)
	output = p.map(func, input_vals)
	p.close()
	p.join()
	return output

def get_mouse_to_human_entrez():
	ortholog_url = 'http://www.informatics.jax.org/downloads/reports/HOM_AllOrganism.rpt'
	s=requests.get(ortholog_url).content
	ortholog_df=pd.read_csv(io.StringIO(s.decode('utf-8')), sep='\t', header=0, index_col=False)
	cols = ['Symbol','EntrezGene ID']
	mouse_to_human = pd.merge(ortholog_df[['HomoloGene ID']+cols][(ortholog_df['Common Organism Name']=='mouse, laboratory')].rename(index=str, columns=dict(zip(cols, ['Mouse_'+col for col in cols]))),
		ortholog_df[['HomoloGene ID']+cols][(ortholog_df['Common Organism Name']=='human')].rename(index=str, columns=dict(zip(cols, ['Human_'+col for col in cols]))), on='HomoloGene ID')
	return mouse_to_human

def filter_and_merge_results(exp_df):
	# get mean for genes seen more than once
	exp_df = exp_df.groupby('Node').apply(np.mean).drop(['Node'],axis=1).reset_index()	
	# get max fc
	exp_df.loc[:,'max_fc'] = np.max(exp_df[[col for col in exp_df.columns.values if 'Log2fc_' in col]].values,axis=1)
	# remove if all fcs 0
	exp_df = exp_df[exp_df.max_fc!=0]
	return exp_df

<<<<<<< HEAD
def get_mapped_counts_and_diff_exp_dfs(version, mouse_to_human, spoke_genes, node_list):
=======
def get_mapped_counts_and_diff_exp_dfs(version, accession, mouse_to_human, spoke_genes, node_list):
	# get filename
	diff_file = [f for f in listdir(input_directory+version) if ('differential_expression' in f) & (accession in f)][0]
>>>>>>> dd91528dc5b35449df1bf14023a1e43df359f62a
	# load exp df
	exp_df = pd.read_csv(input_directory+version+diff_file, sep='\t', header=0, index_col=False)
	exp_df.loc[:,'ENTREZID'] = exp_df.ENTREZID.values.astype(int)
	# make map from ensembl to human entrez
	map_from_ensembl = pd.merge(exp_df[['ENSEMBL', 'ENTREZID']], mouse_to_human[['Mouse_EntrezGene ID', 'Human_EntrezGene ID']].rename(index=str, columns={'Mouse_EntrezGene ID':'ENTREZID'}), on='ENTREZID')
	map_from_ensembl.loc[:,'Human_EntrezGene ID'] = map_from_ensembl['Human_EntrezGene ID'].values.astype(str)
	# filter mapping so only spoke genes
	map_from_ensembl = map_from_ensembl[map_from_ensembl['Human_EntrezGene ID'].isin(spoke_genes)].rename(index=str, columns={'Human_EntrezGene ID':'Node'})
	# filter express so only spoke genes
	exp_df = pd.merge(map_from_ensembl, exp_df, on=['ENSEMBL', 'ENTREZID'])
	# drop other name cols and exp counts
	exp_df = exp_df[[col for col in exp_df.columns.values if re.match(r'Node|P.value_|Adj.p.value_|Log2fc_|LRT.p.value', col)!=None]].drop_duplicates()
	# filter df
	exp_df = filter_and_merge_results(exp_df)
	# load gene indices for rounds
	gene_indices_df = load_gene_indices_df(node_list)
	# merge w/ rounds
	exp_df = pd.merge(gene_indices_df, exp_df, on='Node').sort_values('node_2_index')
	# add to gene indices if seen
	gene_indices_df.loc[:,'seen'] = np.in1d(gene_indices_df.node_2_index.values, exp_df.node_2_index.unique())
	# normalize by seen genes
	gene_indices_df = gene_indices_df[gene_indices_df.seen==True]
	# n genes
	total_genes = len(gene_indices_df)
	return exp_df, gene_indices_df, total_genes

def check_sign(exp_df, groups, space_ground_basal, group_type_1, group_type_2):
	group_1_list = [groups[i] for i, group_type in enumerate(space_ground_basal) if group_type_1 == group_type]
	group_2_list = [groups[i] for i, group_type in enumerate(space_ground_basal) if group_type_2 == group_type]
	group_1_over_2 = ['Log2fc_(%s)v(%s)' % tuples for tuples in itertools.product(group_1_list,group_2_list)]
	exp_df.loc[:,'%s_over_%s_same_sign' % (group_type_1, group_type_2)] = (np.sum(exp_df[group_1_over_2].values>0, axis=1) == len(group_1_over_2)) | (np.sum(exp_df[group_1_over_2].values<0, axis=1) == len(group_1_over_2))
	exp_df.loc[:,'%s_over_%s_pos' % (group_type_1, group_type_2)] = np.sum(exp_df[group_1_over_2].values>0, axis=1) == len(group_1_over_2)
	return exp_df

def filter_same_direction(exp_df, gene_indices_df, samples):
	groups = np.array(list(set(np.ravel([s[8:-1].split(')v(') for s in samples]))))
	space_ground_basal = np.array([re.findall(r'SPACE|GROUND|BASAL', group, re.IGNORECASE)[0] for group in groups])
	control_groups = np.setdiff1d(space_ground_basal, ['Space'])
	for control_group in control_groups:
		exp_df = check_sign(exp_df, groups, space_ground_basal, 'Space', control_group)
	# check same dir within control
	exp_df.loc[:,'%s_over_%s_same_sign' % ('Space', 'Ground_Basal')] = np.all(exp_df[['Space_over_%s_same_sign' % control_group for control_group in control_groups]].values==True, axis=1)
	# check same dir between control
	exp_df.loc[:,'%s_over_%s_same_sign' % ('Space', 'Ground_Basal')] = (exp_df.Space_over_Ground_Basal_same_sign.values==True)&(np.all(exp_df[['Space_over_%s_pos' % control_group for control_group in control_groups]].values==True, axis=1)|np.all(exp_df[['Space_over_%s_pos' % control_group for control_group in control_groups]].values==False, axis=1))
	# filter genes
	gene_indices_df = gene_indices_df[gene_indices_df.node_2_index.isin(exp_df[exp_df.Space_over_Ground_Basal_same_sign==True].node_2_index.values)]
	total_genes=len(gene_indices_df)
	return exp_df, gene_indices_df, total_genes




# load spoke info
node_info_df = pd.read_csv(spoke_directory+'node_info.tsv', sep='\t', header=0, index_col=False)

# load mouse to human
mouse_to_human = get_mouse_to_human_entrez()
# get fc
exp_df, gene_indices_df, total_genes = get_mapped_counts_and_diff_exp_dfs(version, mouse_to_human, node_info_df[node_info_df.Node_Type=='Gene'].Node.values, node_info_df.Node.values)

samples = np.array([col for col in exp_df.columns.values if 'Log2fc_' in col])
# filter fc fc
exp_df, gene_indices_df, total_genes = filter_same_direction(exp_df, gene_indices_df, samples)
# z score
avg_rank=np.sum(np.array(get_any_func_in_par(get_mean_in_par, gene_indices_df.Round.unique(), n_jobs)), axis=0, dtype=float)/float(total_genes)
# get std rank of entire pop
std_rank = np.sqrt(np.sum(np.array(get_any_func_in_par(get_std_in_par, gene_indices_df.Round.unique(), n_jobs)), axis=0, dtype=float)/float(total_genes))
# get results per sample
sample_psev = np.array(get_any_func_in_par(get_mean_zscore_rank_exp, gene_indices_df[gene_indices_df.seen==True].Round.unique(), n_jobs))
sample_psev=np.sum(sample_psev, axis=0)/float(total_genes)
# rank nodes
sample_psev = get_rank_matrix(sample_psev, [], False)
result_df = pd.concat((node_info_df, pd.DataFrame(sample_psev.T, columns=samples)),axis=1)
for sample in samples:
	result_df.loc[:,'Rank_by_type_%s' % sample] = get_rank_by_type_vector(result_df[sample].values, node_info_df.Node_Type.values)

#
result_df.to_csv(input_directory+version+output_filename+'%s.tsv' % save_str, sep='\t', header=True, index=False)





