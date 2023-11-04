

import pandas as pd
import numpy as np
import os
from docopt import docopt
from sklearn.metrics.pairwise import cosine_similarity
import scipy.spatial as sp

# for testing
from log_colors import colors

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

class SCAG():
    def __init__(self,interaction=4):
        self.interaction = interaction
    
    def ceil(self,n):
        return int(-1 * n // 1 * -1)

    def floor(self,n):
        return int(n // 1)

    def randomize(self,mat):
        '''
            a ramdom algorithm to convert each data to an integer with a specific probability. 
        '''
        [m,n] = mat.shape
        # m - bulk number
        # n - cell type number
        for i in range(m):
            for j in range(n):
                # loop through each entry
                tmp = mat.iloc[i,j]
                if tmp!=0:
                    #print(tmp)
                    c = self.floor(tmp)
                    # if entry is integer, pass
                    if c == tmp:
                        continue
                    else:  
                        d = self.ceil(tmp)
                        #print(c,d)
                        new = np.random.choice([c,d], p=[d-tmp,tmp-c])
                        mat.iloc[i,j] = new
        return mat
    
    def filtering_gene(self,ori_bulk,sc_ref):

        '''
            filter the bulk and single_cell data by only keeping the genes expressed in both bulk and sc.
        '''

        # cell x genes
        # 1. remove unexpressed genes
        filtered_sc = sc_ref[sc_ref.apply(np.sum,axis=1)!=0]
        filtered_bulk = ori_bulk[ori_bulk.apply(np.sum,axis=1)!=0]
        bulk_gene = set(filtered_bulk.index)
        sc_gene = set(filtered_sc.index)
        shared_genes = bulk_gene.intersection(sc_gene)
        filtered_sc = filtered_sc[filtered_sc.index.isin(shared_genes)]
        filtered_bulk = filtered_bulk[filtered_bulk.index.isin(shared_genes)]
        shared_genes=list(shared_genes)
        return filtered_bulk, filtered_sc, shared_genes
    
    def pear(self,D,D_re):
        tmp = np.corrcoef(D.flatten(order='C'), D_re.flatten(order='C'))
        return tmp[0,1] 
    
    def calculate_graph(self,dataframe):
        matrix = dataframe.values.T
        return cosine_similarity(matrix)
    

    def calculate_distance(self,matrix1,matrix2):
        return (1 - sp.distance.cdist(matrix1, matrix2, 'cosine'))
    
    def norm_center(self,df):
        a = df.apply(lambda x: (x)/np.sum(x) , axis=0)
        return a.apply(lambda x: (x - np.mean(x)) , axis=0)

    def init_solution(self,rna_matrix, sc_ref, cell_num, fraction_martix, num):

        '''
            params:
                rna_matrix: The original bulk matrix.
                sc_ref: The sc_ref.
                cell_num: The ramdomized fraction data.
                fraction_martix: The original fraction data.
                num: The cell_number. # TODO: ?
        '''

        print(colors.OKCYAN, "rna matrix\n", rna_matrix, colors.ENDC)

        # initial solution
        picked_index = {}
        correlations = []
        sample_list = rna_matrix.columns.tolist()
        gene_num = rna_matrix.shape[0]
        init_mapping_exp = pd.DataFrame()
        new_bulk_exp = self.norm_center(rna_matrix) 
        new_sc_exp = self.norm_center(sc_ref) # INFO: [gene_num, sc_sample_num]

        for sample_id in range(len(sample_list)):
            # maximize correlation
            sample = sample_list[sample_id]
            G = np.array(new_bulk_exp[sample]).reshape(1,gene_num)
            O = np.array(rna_matrix[sample]).reshape(1,gene_num)
            cor_st_sc = np.dot(G,new_sc_exp) # INFO: [1, sc_sample_num]
            
            # TODO: refactor
            cor_st_sc_tp = pd.concat([pd.DataFrame(cor_st_sc.T).reset_index(drop=True), pd.DataFrame(list(sc_ref.columns.T))], axis=1)
            # print(colors.OKBLUE, "cor_st_sc_tp\n", cor_st_sc_tp, colors.ENDC)
            # redundant change !!
            cor_st_sc_tp.columns = ['cor', 'tp']
            col = list(cor_st_sc_tp.columns)
            B = cor_st_sc_tp.sort_values(by = col[::-1],ascending = False)
            B["cell_type"] = [B.split("__")[1] for B in B["tp"]]
            #B = pd.DataFrame(B, index = exp.index )
            a = cell_num.drop(['sample_id'], axis=1)
            a = a.iloc[sample_id]
            #b=a.loc[:, (a != 0).all(axis=0)]
            #b=b.drop(['sample_id'], axis=1)
            b = (pd.DataFrame(a[a!=0])).fillna(0)
            picked_index[sample] = []
            
            print(colors.OKCYAN, "cor_st_sc_tp\n", cor_st_sc_tp, colors.ENDC)
            print(colors.OKBLUE, "B\n", B, colors.ENDC)
            for index, row in b.iterrows():
                tmp = B[B.iloc[:,2] == index].iloc[0:int(row.iloc[0]),1].tolist()
                #tmp = B[B.iloc[:,1] == index].iloc[0:int(row.iloc[0])].index.tolist()
                picked_index[sample].extend(tmp)
            #break
            print(colors.WARNING, "sc_ref[picked_index[sample]]\n", sc_ref[picked_index[sample]], colors.ENDC)
            exp_tmp = sc_ref[picked_index[sample]].apply(lambda x: x.sum(), axis=1)/num # INFO: get the sum of the rows and then divided by num
            # print(colors.OKGREEN, "exp_tmp\n", exp_tmp, colors.ENDC)
            init_mapping_exp[sample] = exp_tmp
            correlations.append(self.pear(O.T,exp_tmp.values))      
        return picked_index, correlations, init_mapping_exp


    
    def fit(self,sc_ref,ori_bulk,ori_frc_matrix,cell_number,outdir,times):

        '''
            params
                sc_ref: The sc_ref.
                ori_bulk: The original bulk data.
                ori_frc_matrix: The fraction calculated by the graph model.
                cell_number: ? # TODO: ?
                outdir: The folder path to save the outputs.
                times: The iteration times.
        '''

        #frc_matrix = ori_frc_matrix.reset_index()
        #copy fraction results
        frc_matrix = ori_frc_matrix.reset_index()
        del frc_matrix[frc_matrix.columns[0]]
        del sc_ref[sc_ref.columns[0]]
        sample_idex = ori_frc_matrix.index

        # delete the cell type whose number<0.5
        frc_matrix[frc_matrix<(1/(2*cell_number))]=0
        num = frc_matrix * cell_number # NOTE: fraction data times cell_number
        num = self.randomize(num)
        num.insert(0, 'sample_id', sample_idex, allow_duplicates=False)
        num.to_csv(outdir + '/cell_type_num_each_sample.csv', index = False, header = True, sep = ',')

        filter = False
        #filtering gene
        if filter:
            if times==0:
                print('- %d genes in spatial data, %d genes in single-cell data.'%(ori_bulk.shape[0],sc_ref.shape[0]))
                ori_bulk, sc_ref, shared_gene = self.filtering_gene(ori_bulk, sc_ref)
                print('- %d shared and expressed genes has been kept.'%(ori_bulk.shape[0]))
            else:
                shared_gene=list(ori_bulk.index)

        shared_gene=list(ori_bulk.index)
        init_solution,init_correlation,init_mapping_exp = self.init_solution(ori_bulk,sc_ref,num,frc_matrix,cell_number)
        print(colors.RED, times,'initial solution:',"min correlation", min(init_correlation),"average correlation",np.mean(init_correlation),"max correlation", max(init_correlation), colors.ENDC)
        init_mapping_exp.insert(0,'GeneSymbol',shared_gene)
        init_mapping_exp=init_mapping_exp.set_index('GeneSymbol')    
        init_mapping_exp.to_csv(outdir + '/init_mapping_exp.csv', index = True, header= True, sep = ',')
        ori_cosine = self.calculate_graph(ori_bulk)
        init_cosine = self.calculate_graph(init_mapping_exp)
        
        np.savetxt(outdir + '/init_graph.csv', init_cosine, delimiter=',')
        np.savetxt(outdir + '/ori_graph.csv', ori_cosine, delimiter=',')


        return init_mapping_exp,ori_bulk

