import numpy as np
from collections import Counter
import os
import pandas as pd
import io
import requests
import math
import time
import multiprocessing as mp
import itertools 
from itertools import product
from scipy.stats import zscore
import requests
import io
from os import listdir
import re
from scipy import stats


pos_neg_or_same_dir = 'neg'
pos_neg_or_same_dir = 'same_dir'
#pos_neg_or_same_dir='pos'
#pos_neg_or_same_dir='abs_exp'
#pos_neg_or_same_dir='abs_dot'


input_directory='GeneLab_for_SPOKE/'
#gene_psev_directory='/wynton/scratch/canelson/gene_psevs/'
gene_psev_directory='spoke_v_2/gene_psevs/'
#omop_directory='omop/mapping/qb3/omop_build_1/'
omop_directory='omop_build_1/'
spoke_directory = 'spoke_v_2/'
probability_random_jump = 0.1
#probability_random_jump = 0.33

use_seen_genes_for_z = True
n_jobs=12

save_str = '_'.join(str(probability_random_jump).split('.'))

gene_indices_df, exp_df = pd.DataFrame(), pd.DataFrame()

samples, avg_rank, std_rank = [],[],[]

def get_spoke_genes(spoke_directory):
	# load node list
	node_list = np.load(spoke_directory+'spoke_node_list.npy', allow_pickle=False).astype(str).astype(str)
	# load node name list
	node_name_list = np.load(spoke_directory+'node_name_list.npy', allow_pickle=False)
	#node_name_list = np.array([name.decode('UTF-8') for name in node_name_list], dtype=str)
	# load spoke node list
	node_type_list = np.load(spoke_directory+'node_type_list.npy', allow_pickle=False).astype(str)
	# node info
	node_info_df = pd.DataFrame(np.array([node_list, node_name_list, node_type_list]).T, columns=['Node', 'Node_Name', 'Node_Type'])
	node_info_df.loc[:,'Node_Index'] = np.arange(len(node_info_df))
	return node_info_df, node_list, node_name_list, node_type_list

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
	t1=time.time()
	output = p.map(func, input_vals)
	print(time.time() - t1)
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

def filter_and_merge_results(exp_df, p_val_thresh):
	# add coll w/ # of tests that pass p value thresh 
	exp_df.loc[:,'n_pass_p_thresh'] = np.sum(exp_df[[col for col in exp_df.columns.values if 'P.value_' in col]].values<=p_val_thresh, axis=1)
	# remove if 0 sig
	#exp_df = exp_df[exp_df.n_pass_p_thresh>0]
	# get mean for genes seen more than once
	exp_df = exp_df.groupby('Node').apply(np.mean).drop(['Node'],axis=1).reset_index()	
	# get max fc
	exp_df.loc[:,'max_fc'] = np.max(exp_df[[col for col in exp_df.columns.values if 'Log2fc_' in col]].values,axis=1)
	# remove if all fcs 0
	exp_df = exp_df[exp_df.max_fc!=0]
	return exp_df

def get_mapped_counts_and_diff_exp_dfs(version, accession, ercc, mouse_to_human, spoke_genes, p_val_thresh, node_list):
	# get filenames
	diff_files = np.array([f for f in listdir(input_directory+version+'%s/' % accession) if '%sdifferential_expression' % ercc in f])
	diff_files = diff_files[np.array([len(f) for f in diff_files])==min([len(f) for f in diff_files])][0]
	print(diff_files)
	# load exp df
	exp_df = pd.read_csv(input_directory+version+'%s/%s' % (accession, diff_files), sep='\t', header=0, index_col=False)
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
	exp_df = filter_and_merge_results(exp_df, p_val_thresh)
	# load gene indices for rounds
	gene_indices_df = load_gene_indices_df(node_list)
	# merge w/ rounds
	exp_df = pd.merge(gene_indices_df, exp_df, on='Node').sort_values('node_2_index')
	# add to gene indices if seen
	gene_indices_df.loc[:,'seen'] = np.in1d(gene_indices_df.node_2_index.values, exp_df.node_2_index.unique())
	if use_seen_genes_for_z == True:
		gene_indices_df = gene_indices_df[gene_indices_df.seen==True]
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
	print space_ground_basal
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

def get_t_stat_and_p(i):
	return stats.ttest_ind(group_1_vals[:,i], group_2_vals[:,i])

def add_mean_stats_to_df(group_type_1, node_info_df, matrix, compare_groups, space_groups, df_type):
	df = pd.concat((node_info_df, pd.DataFrame(matrix.T, columns=compare_groups)), axis=1)
	df.loc[:,'mean_controls'] = np.mean(matrix[np.arange(len(compare_groups))[np.in1d(compare_groups, np.setdiff1d(compare_groups, space_groups))]], axis=0)
	df.loc[:,'mean_ground_space'] = np.mean(df[['Basal_Space', 'Ground_Space']].values,axis=1)
	df.loc[:,'mean_ground_basals'] = np.mean(df[['Basal_Basal', 'Ground_Ground', 'Basal_Ground', 'Ground_Basal']].values,axis=1)
	df.to_csv(input_directory+version+'meta_compare_%s_%s%s_%s.tsv'%(group_type_1, df_type, save_str, pos_neg_or_same_dir), sep='\t', header=True, index=False)
	return df

p_val_thresh = 0.05
version='V2/'
ercc='' # 'ERCCnorm_'
#accession='GLDS-288'
# load spoke info
node_info_df, node_list, node_name_list, node_type_list = get_spoke_genes(spoke_directory)

mouse_to_human = get_mouse_to_human_entrez()

all_results_df = pd.DataFrame(np.array([node_list, node_name_list]).T, columns=['Node', 'Node_Name'])
all_gene_exp = node_info_df[node_info_df.Node_Type=='Gene']
all_p_val_exp = node_info_df[node_info_df.Node_Type=='Gene']

all_samples = []
all_studies = []
genes_seen = []
for a in [4, 288, 289, 244, 245, 246]:
	ercc = ''
	if a in [244, 245, 246]:
		#ercc='ERCCnorm_'
		ercc=''
	exp_df, gene_indices_df, total_genes = get_mapped_counts_and_diff_exp_dfs(version, 'GLDS-%s' % a, ercc, mouse_to_human, node_list[node_type_list=='Gene'], p_val_thresh, node_list)
	samples = np.array([col for col in exp_df.columns.values if 'Log2fc_' in col])
	if pos_neg_or_same_dir[:8]=='same_dir':
		print pos_neg_or_same_dir
		exp_df, gene_indices_df, total_genes = filter_same_direction(exp_df, gene_indices_df, samples)
	elif (pos_neg_or_same_dir=='pos') or (pos_neg_or_same_dir=='neg'):
		for sample in samples:
			if pos_neg_or_same_dir=='pos':
				exp_df.loc[:,sample] = np.max(np.array([exp_df[sample].values, np.zeros(exp_df[sample].values.shape)]), axis=0)
			else:
				exp_df.loc[:,sample] = -np.min(np.array([exp_df[sample].values, np.zeros(exp_df[sample].values.shape)]), axis=0)
	all_gene_exp = pd.merge(all_gene_exp, exp_df[np.concatenate((['Node'], samples))], on='Node', how='left')
	all_p_val_exp = pd.merge(all_p_val_exp, exp_df[np.concatenate((['Node'], [sample.replace('Log2fc_','P.value_') for sample in samples]))], on='Node', how='left')
	genes_seen.append(gene_indices_df[gene_indices_df.seen==True].Node.values)
'''
	#
	avg_rank=np.sum(np.array(get_any_func_in_par(get_mean_in_par, gene_indices_df.Round.unique(), n_jobs)), axis=0, dtype=float)/float(total_genes)
	# get std rank of entire pop
	std_rank = np.sqrt(np.sum(np.array(get_any_func_in_par(get_std_in_par, gene_indices_df.Round.unique(), n_jobs)), axis=0, dtype=float)/float(total_genes))
	if 'abs_exp' in pos_neg_or_same_dir:
		print(exp_df.head())
		for sample in samples:
			exp_df.loc[:,sample]=np.abs(exp_df[sample].values)
	# get results per sample
	#sample_psev = np.sum(np.array(get_any_func_in_par(get_mean_zscore_rank_exp, gene_indices_df[gene_indices_df.seen==True].Round.unique(), n_jobs)), axis=0)/float(total_genes)
	sample_psev = np.array(get_any_func_in_par(get_mean_zscore_rank_exp, gene_indices_df[gene_indices_df.seen==True].Round.unique(), n_jobs))
	print sample_psev.shape
	sample_psev=np.sum(sample_psev, axis=0)/float(total_genes)
	print sample_psev.shape
	if 'abs_dot' in pos_neg_or_same_dir:
		sample_psev = np.abs(sample_psev)
	sample_psev = get_rank_matrix(sample_psev, [], False)
	result_df = pd.concat((node_info_df, pd.DataFrame(sample_psev.T, columns=samples)),axis=1)
	for sample in samples:
		result_df.loc[:,'Rank_by_type_%s' % sample] = get_rank_by_type_vector(result_df[sample].values, node_type_list)
		#result_df.loc[:,'Rank_%s' % sample] = len(node_list)-get_order_rank_vector(result_df[sample].values, np.arange(len(node_list)))
	#
	all_results_df = pd.merge(all_results_df, result_df, on=['Node', 'Node_Name'])
	all_samples.append(samples)
	all_studies.append(np.repeat(a, len(samples)))
	for sample in samples:
		s = ['Log2fc_(MHU-2 & Space Flight & 1G by centrifugation)v(MHU-2 & Ground Control & 1G)', 'Log2fc_(MHU-2 & Space Flight & uG)v(MHU-2 & Ground Control & 1G)', 'Log2fc_(MHU-1 & Space Flight & uG)v(MHU-1 & Ground Control & 1G)', 'Log2fc_(MHU-1 & Space Flight & 1G by centrifugation)v(MHU-1 & Ground Control & 1G)', 'Log2fc_(Space Flight)v(Ground Control)', 'Log2fc_(Ground Control & ~30 & On Earth & Upon euthanasia)v(Basal Control & 1 & On Earth & Upon euthanasia)', 'Log2fc_(Ground Control & ~60 & On Earth & Carcass)v(Basal Control & 1 & On Earth & Carcass)', 'Log2fc_(Space Flight & ~60 & On ISS & Carcass)v(Ground Control & ~60 & On Earth & Carcass)', 'Log2fc_(Space Flight & ~30 & On Earth & Upon euthanasia)v(Basal Control & 1 & On Earth & Upon euthanasia)', 'Log2fc_(Space Flight & ~30 & On Earth & Upon euthanasia)v(Ground Control & ~30 & On Earth & Upon euthanasia)', 'Log2fc_(Space Flight & ~30 & On Earth & Upon euthanasia)v(Basal Control & 1 & On Earth & Upon euthanasia)', 'Log2fc_(Space Flight & ~30 & On Earth & Upon euthanasia)v(Ground Control & ~30 & On Earth & Upon euthanasia)', 'Log2fc_(Space Flight & ~60 & On ISS & Carcass)v(Ground Control & ~60 & On Earth & Carcass)', 'Log2fc_(Space Flight & ~60 & On ISS & Carcass)v(Basal Control & 1 & On Earth & Carcass)', 'Log2fc_(Space Flight & ~60 & On ISS & Carcass)v(Basal Control & 1 & On Earth & Carcass)', 'Log2fc_(Space Flight & ~30 & On Earth & Upon euthanasia)v(Basal Control & 1 & On Earth & Upon euthanasia)', 'Log2fc_(Ground Control & ~30 & On Earth & Upon euthanasia)v(Basal Control & 1 & On Earth & Upon euthanasia)', 'Log2fc_(Space Flight & ~60 & On ISS & Carcass)v(Ground Control & ~60 & On Earth & Carcass)', 'Log2fc_(Space Flight & ~60 & On ISS & Carcass)v(Basal Control & 1 & On Earth & Carcass)', 'Log2fc_(Ground Control & ~30 & On Earth & Upon euthanasia)v(Basal Control & 1 & On Earth & Upon euthanasia)', 'Log2fc_(Space Flight & ~30 & On Earth & Upon euthanasia)v(Ground Control & ~30 & On Earth & Upon euthanasia)', 'Log2fc_(Ground Control & ~60 & On Earth & Carcass)v(Basal Control & 1 & On Earth & Carcass)', 'Log2fc_(Ground Control & ~60 & On Earth & Carcass)v(Basal Control & 1 & On Earth & Carcass)', 'Log2fc_(Space Flight & uG)v(Ground Control & 1G)', 'Log2fc_(Space Flight & 1G by centrifugation)v(Ground Control & 1G)']
		if sample in s:
			l = np.ravel([result_df[[sample, 'Rank_by_type_'+sample]][result_df.Node_Name==name].values.astype(str)[0] for name in ['Space Motion Sickness', 'space motion sickness', 'Jet Lag Syndrome']])
			print '\t'.join(np.concatenate(([sample, str(a)],l)))


'''

pos_neg_or_same_dir = pos_neg_or_same_dir+'_no_ercc'
all_p_val_exp.to_csv(input_directory+version+'mean_gene_p_val_exp_in_meta_count%s_%s.tsv' % (save_str, pos_neg_or_same_dir), sep='\t', header=True, index=False)



'''
all_genes = node_info_df[node_info_df.Node_Type=='Gene'].Node.values
seen_genes_df = pd.DataFrame(np.array([all_genes]+[np.in1d(all_genes, genes) for genes in genes_seen]).T, columns=np.array(['Node', 4, 288, 289, 244, 245, 246], dtype=str))
seen_genes_df.to_csv(input_directory+version+'seen_genes_df_%s_%s.tsv' % (save_str, pos_neg_or_same_dir), sep='\t', header=True, index=False)




############ save meta info ############
all_results_df.to_csv(input_directory+version+'ranks_and_rank_by_type_for_meta%s_%s.tsv' % (save_str, pos_neg_or_same_dir), sep='\t', header=True, index=False)
genes_seen_count_df = pd.merge(node_info_df,pd.DataFrame(np.array(Counter(np.concatenate(genes_seen)).items()), columns=['Node', 'Count']), on='Node')
genes_seen_count_df.to_csv(input_directory+version+'genes_seen_in_meta_count%s_%s.tsv' % (save_str, pos_neg_or_same_dir), sep='\t', header=True, index=False)
all_gene_exp.to_csv(input_directory+version+'mean_gene_log2_exp_in_meta_count%s_%s.tsv' % (save_str, pos_neg_or_same_dir), sep='\t', header=True, index=False)
#
all_samples = np.concatenate(all_samples)
all_sample_cols = np.array([col for col in all_results_df.columns.values if ('Log2fc_' in col) & ('Rank_' not in col)])
all_sample_types = np.array(['_'.join(re.findall(r'SPACE|GROUND|BASAL', sample, re.IGNORECASE)) for sample in all_sample_cols])
meta_df = pd.DataFrame(np.array([all_samples, all_sample_cols, all_sample_types, np.concatenate(all_studies)]).T, columns=['Sample_Name', 'Sample_Col_Name', 'Sample_Type', 'Study'])
meta_df.to_csv(input_directory+version+'sample_meta_info%s_%s.tsv' % (save_str, pos_neg_or_same_dir), sep='\t', header=True, index=False)

############ save t stat p vals ############

space_groups = ['Space_Basal', 'Space_Ground', 'Both_Space']
for group_type_1 in space_groups:
	if group_type_1 == 'Both_Space':
		compare_groups = list(set(all_sample_types)-set(['Space_Basal', 'Space_Ground']))
		group_1_vals, group_2_vals = all_results_df[all_sample_cols[np.in1d(all_sample_types, ['Space_Basal', 'Space_Ground'])]].values.T, []
	else:
		compare_groups = list(set(all_sample_types)-set([group_type_1]))
		group_1_vals, group_2_vals = all_results_df[all_sample_cols[all_sample_types==group_type_1]].values.T, []
	all_p_vals, all_t_stats = np.zeros((len(compare_groups), len(node_list))),np.zeros((len(compare_groups), len(node_list)))
	for index_2, group_type_2 in enumerate(compare_groups):
		group_2_vals = all_results_df[all_sample_cols[all_sample_types==group_type_2]].values.T
		t_stat, p_val = np.array(get_any_func_in_par(get_t_stat_and_p, np.arange(len(node_list)), n_jobs=10)).T
		all_p_vals[index_2] = p_val
		all_t_stats[index_2] = t_stat
	p_val_df = add_mean_stats_to_df(group_type_1, node_info_df, all_p_vals, compare_groups, space_groups, 'p_value')
	t_stat_df = add_mean_stats_to_df(group_type_1, node_info_df, all_t_stats, compare_groups, space_groups, 't_stat')

'''

