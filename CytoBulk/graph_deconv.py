import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import multiprocessing as mp
import json
import warnings
import sys

from tqdm import tqdm
from typing import Callable, Generator
import torch.distributed as dist
from scipy.stats import pearsonr


from utils import check_paths
from draw_plots import draw_deconv_eval_plots


class Const:
    """
    Some constants used in the class.
    """
    MODE_TRAINING = "training"
    MODE_PREDICTION = "prediction"
    SAMPLE_COL = "Unnamed: 0"
    # GENE_SYMBOL_COL = "Unnamed: 0"
    GENE_SYMBOL_COL = "GeneSymbol"
    BATCH_SIZE = 64
    LEARNING_RATE = 0.005
    EPOCH_NUM = 40
    SEED = 20230602

    CHEB_MODE = 0


def configure_device(use_gpu):
    if use_gpu:
        if torch.cuda.is_available(): return "cuda"
        if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def naive_parallel(
        func: Callable, 
        args: Generator, 
        cpu_limit = mp.cpu_count()
):
    print(f"Try to allocate {cpu_limit}, ", end='')
    cpu_limit = min(cpu_limit, mp.cpu_count())
    print(f"{cpu_limit} cpu(s) are currently available.")

    with mp.Pool(processes=cpu_limit) as pool:
        ret = pool.starmap_async(func, args)
        print(ret.get())


class LinearModel(torch.nn.Module):
    def __init__(self, t):
        super(LinearModel, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(t,64),
            nn.BatchNorm1d(64),
            torch.nn.Linear(64,1),
            torch.nn.Sigmoid()
        )

    def forward(self, t):
        f = self.encoder(t)
        return f
    

class ChebConv(nn.Module):
    def __init__(self, in_c, out_c, K, device, bias=True, normalize=True):
        super(ChebConv, self).__init__()
        self.device = device
        self.normalize = normalize

        self.weight = nn.Parameter(torch.Tensor(K + 1, 1, in_c, out_c))
        # nn.init.xavier_normal_(self.weight)
        nn.init.orthogonal_(self.weight, gain=nn.init.calculate_gain('leaky_relu', 0.4))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_c))
            # nn.init.zeros_(self.bias)
            nn.init.orthogonal_(self.bias, gain=nn.init.calculate_gain('leaky_relu', 0.4))
        else:
            self.register_parameter("bias", None)

        self.K = K + 1

    def forward(self, inputs, graph, mode=Const.CHEB_MODE):
        L = ChebConv.get_laplacian(graph, self.normalize)
        
        L = L.cpu()
        lam, u = np.linalg.eig(L)
        lam = torch.FloatTensor(lam)
        lam = lam.to(self.device)
        lam= torch.diag(lam)
        u = torch.FloatTensor(u).to(self.device)
        lam = 2*((lam - torch.min(lam).to(self.device)) / (torch.max(lam).to(self.device) - torch.min(lam).to(self.device))) - torch.eye(lam.size(0)).to(self.device)
        
        mul_L = self.cheb_polynomial(lam).unsqueeze(1)

        if mode == 0:
            # print("chebnet mode 0")
            result = torch.matmul(inputs, mul_L)
        elif mode == 1:
            # print("chebnet mode 1")
            result = torch.matmul(inputs, mul_L)
            result = torch.matmul(result, u.t())
        elif mode == 2:
            # print("chebnet mode 2")
            result = torch.matmul(u, mul_L)
            result = torch.matmul(result, u.t())
            result = torch.matmul(inputs, result)

        result = torch.matmul(result, self.weight)
        result = torch.sum(result, dim=0) + self.bias

        temp=[]
        for i in range(result.size()[0]):
            if torch.min(result[i]) == torch.max(result[i]):
                temp.append(result[i])
            else:
                temp.append(2*((result[i] - torch.min(result[i])) / (torch.max(result[i]) - torch.min(result[i]))) - 1)
        temp = torch.stack(temp)

        return temp

    def cheb_polynomial(self, laplacian):
        N = laplacian.size(0)
        multi_order_laplacian = torch.zeros([self.K, N, N], device=laplacian.device, dtype=torch.float)
        multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=torch.float)

        if self.K == 1:
            return multi_order_laplacian
        else:
            multi_order_laplacian[1] = laplacian
            if self.K == 2:
                return multi_order_laplacian
            else:
                for k in range(2, self.K):
                    multi_order_laplacian[k] = 2 * torch.mm(laplacian, multi_order_laplacian[k-1]) - multi_order_laplacian[k-2]

        return multi_order_laplacian

    @staticmethod
    def get_laplacian(graph, normalize):
        if normalize:

            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L


class ChebNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c, K, device):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(in_c=in_c, out_c=hid_c, K=K, device=device)
        self.conv2 = ChebConv(in_c=hid_c, out_c=out_c, K=K, device=device)
        self.act = nn.ELU()

    def forward(self, graph, data):
        graph_data = graph
        flow_x = data

        B, N = flow_x.size(0), flow_x.size(1)

        flow_x = flow_x.view(B, 1, N)

        output_1 = self.act(self.conv1(flow_x, graph_data))
        output_2 = self.act(self.conv2(output_1, graph_data))

        return output_2


class InferDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        
    def __getitem__(self, index):
        x = self.data[index]
        return x
    
    def __len__(self):
        return len(self.data)

def get_G(cell_name, sel_gene, sc_folder):
        def __get_mat_YW(y_path, sel_gene):
            mat_Y_ori = pd.read_csv(y_path, sep='\t') # NOTE: hard coded as "\t"
            # mat_Y_ori = pd.read_csv(y_path)
            mat_Y_ori = mat_Y_ori[mat_Y_ori[Const.GENE_SYMBOL_COL].isin(sel_gene)]
            mar_scref_inter_gene = mat_Y_ori[Const.GENE_SYMBOL_COL].to_list()
            mat_Y = mat_Y_ori.drop(Const.GENE_SYMBOL_COL, axis=1)
            mat_Y = torch.FloatTensor(mat_Y.values)
            mat_W = mat_Y @ mat_Y.t()
            return mat_Y , mat_W, mar_scref_inter_gene

        sec_num = 1e-20
        mat_Y, mat_W, mar_scref_inter_gene = __get_mat_YW(
            f"{sc_folder}PBMC_30K_{cell_name}_scrna.txt", # NOTE: mind the path
            # f"{sc_folder}{cell_name}.csv",
            sel_gene
        )
        num = len(mat_W)
        mat_G = mat_W + sec_num*torch.eye(num) + sec_num*torch.ones(num, num)
        return mat_G, num, mar_scref_inter_gene

def select_gene(expression: pd.DataFrame, sel_gene: list):
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    ret_exp = pd.DataFrame(expression.iloc[:, 0])
    for gene in sel_gene:
        ret_exp[gene] = expression[gene] if gene in expression.columns else 0
    # print(ret_exp.iloc[:, 1:])
    return ret_exp.iloc[:, 1:]

def train_cell_loop_once(cell, 
                        marker,
                        expression,
                        fraction,
                        batch_size,
                        sc_folder,
                        out_dir,
                        device
                    ):
    print(f"Start training the model for {cell} cell type...")

    sel_gene = marker[cell]
    mat_G, num, mar_scref_inter_gene = get_G(cell, sel_gene, sc_folder)
    mat_G = mat_G.to(device)

    # input_bulk = expression.loc[:, expression.columns.isin(sel_gene)]

    input_bulk = select_gene(expression, mar_scref_inter_gene)
    # input_bulk = input_bulk.sample(n=batch_size*(input_bulk.shape[0] // batch_size), axis=0, random_state=Const.SEED)
    train_data = input_bulk.values
    # train_label = fraction[cell].sample(n=batch_size*(input_bulk.shape[0] // batch_size), axis=0, random_state=Const.SEED)
    # train_label = train_label.values
    train_label = fraction[cell].values
    # print(train_data.shape, train_label.shape)

    # train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(train_data), torch.FloatTensor(train_label))
    # train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)

    # valid_expression = pd.read_csv("./output/transfer_testing/SKCM_small/SKCM_GSE139249_part_expression.csv")
    # valid_fraction = pd.read_csv("./output/transfer_testing/SKCM_small/SKCM_GSE139249_part_fraction.csv")
    # valid_bulk = select_gene(valid_expression, mar_scref_inter_gene)
    # valid_data = valid_bulk.values
    # valid_label = valid_fraction[cell].values
    # valid_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(valid_data), torch.FloatTensor(valid_label))
    # valid_loader = torch.utils.data.DataLoader(dataset = valid_dataset, batch_size = 1, shuffle = False)

    full_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(train_data), torch.FloatTensor(train_label))
    train_size = int(batch_size * (0.85 * len(full_dataset) // batch_size))
    valid_size = len(full_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [train_size, valid_size])
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(dataset = valid_dataset, batch_size = 1, shuffle = False)
    print(train_size, valid_size)

    model_graph = ChebNet(num,num,num,2, device=device).to(device)
    model_graph_optim = torch.optim.Adam(model_graph.parameters(), lr=Const.LEARNING_RATE, weight_decay = 1e-8)
    model_graph_schelr = torch.optim.lr_scheduler.StepLR(model_graph_optim, 5, gamma=0.9)
    model_linear = LinearModel(num).to(device)
    model_linear_optim = torch.optim.Adam(model_linear.parameters(), lr=Const.LEARNING_RATE, weight_decay = 1e-8)
    model_linear_schelr = torch.optim.lr_scheduler.StepLR(model_linear_optim, 5, gamma=0.9)

    # TODO: flag
    # model_graph.load_state_dict(torch.load(f'./output/SKCM/model/graph_{cell}.pt', map_location='cpu'))
    # model_linear.load_state_dict(torch.load(f'./output/NSCLC/model/linear_{cell}.pt', map_location='cpu'))

    plot_info_dict = {"mse_loss": []}
    pre_pearson_r = -np.inf; graph_break = False
    best_graph = None; best_linear = None
    # for epo in tqdm(range(Const.EPOCH_NUM), leave=False):
    for epo in range(Const.EPOCH_NUM):
        model_graph.train(); model_linear.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            target = torch.reshape(target,(batch_size, 1))
            data = data.to(torch.float32).to(device)
            target = target.to(torch.float32).to(device)

            model_graph_optim.zero_grad()
            model_linear_optim.zero_grad()

            zlist=torch.reshape(model_graph(mat_G, data), (batch_size, -1))      
            output_frac = model_linear(zlist)
            # if batch_idx == 0: print(output_frac)
            # sys.exit(0)

            loss_f = ((output_frac-target)**2).sum() / 1 / batch_size
            plot_info_dict["mse_loss"].append(loss_f.data.cpu().detach().clone().numpy().item())

            # if batch_idx == 1:
            #     if epo>20 and loss_f<0.01:
            #         should_break = True; break

            loss_f.backward()

            if not graph_break: model_graph_optim.step()
            model_linear_optim.step()

        model_graph.eval(); model_linear.eval()
        with torch.no_grad():
            valid_cor_dict = {
                "frac_pred": [],
                "frac_truth": []
            }
            for batch_idx, (data, target) in enumerate(valid_loader):
                target = torch.reshape(target,(1, 1))
                data = data.to(torch.float32).to(device)
                target = target.to(torch.float32).to(device)

                zlist = torch.reshape(model_graph(mat_G, data), (1, -1))      
                output_frac = model_linear(zlist)
                # if batch_idx == 0: print(output_frac)

                loss_f = ((output_frac-target)**2).sum()
                valid_cor_dict["frac_pred"].append(output_frac.cpu().detach().clone().numpy().item())
                valid_cor_dict["frac_truth"].append(target.cpu().detach().clone().numpy().item())
            
            pearson_r, pearson_p = pearsonr(valid_cor_dict["frac_pred"], valid_cor_dict["frac_truth"])
            if pearson_r > pre_pearson_r:
                pre_pearson_r = pearson_r
                best_graph = model_graph.state_dict()
                best_linear = model_linear.state_dict()
            if loss_f <= 0.01 and pearson_r >= 0.85: graph_break = True
            print(f"epoch{epo}-pearsonR", pearson_r, pearson_p, loss_f)

        model_graph_schelr.step()
        model_linear_schelr.step()

        if (loss_f <= 0.009 and
            (pearson_r >= 0.95 or (epo > 10 and pearson_r >= 0.90) or (epo > 20 and pearson_r >= 0.85))): break    

    with open(f'{out_dir}/plot/train_plot_info_{cell}.json', 'w') as f: json.dump(plot_info_dict, f)
    print(f"Saving {cell} model...", end=" ")
    # TODO: path check
    torch.save(best_graph, f"{out_dir}/model/graph_{cell}.pt")
    torch.save(best_linear, f"{out_dir}/model/linear_{cell}.pt")
    print("Done.")

class GraphDeconv:
    def __init__(
            self,
            cell_num=200,
            mode=Const.MODE_PREDICTION,
            use_gpu=True
    ):
        """
            cell_num: int, the number of cell for each bulk sample.
            mode: string, prediction or training.
            use_gpu: bool, if `True`, the model will use CUDA or MPS, otherwise, it will only use CPU.
        """
        self.cell_num = cell_num
        self.mode = mode
        self.device = configure_device(use_gpu)

    def fit(
        self,
        out_dir,
        expression,
        marker=None,
        sc_folder=None,
        model_folder=None
    ):
        # TODO: test if this still work
        # tot_cell_list = marker.keys() 
        tot_cell_list = ({"NK cells":[],"B cells":[],"Monocytes":[],"T CD4 naive cells":[],"T CD4 effector cells":[],"T CD8 central memory cells":[],"T CD8 effector cells":[],"T CD8 effector memory cells":[],"T CD8 exhausted cells":[],"T CD8 naive cells":[],"cDC Cells":[],"Tregs":[]}).keys()

        final_ret = pd.DataFrame()
        for cell in tqdm(tot_cell_list, leave=False):
            sel_gene = marker[cell]
            mat_G, num, mar_scref_inter_gene = get_G(cell, sel_gene, sc_folder)
            mat_G = mat_G.to(self.device)

            input_bulk = select_gene(expression, mar_scref_inter_gene)
            test_data = input_bulk.values

            test_dataset = InferDataset(torch.FloatTensor(test_data))
            test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = 1, shuffle = False)

            model_graph = ChebNet(num,num,num,2, device=self.device).to(self.device)
            model_graph.load_state_dict(torch.load(f'{model_folder}/graph_{cell}.pt'))
            model_linear = LinearModel(num).to(self.device)
            model_linear.load_state_dict(torch.load(f'{model_folder}/linear_{cell}.pt'))
            model_graph.eval(); model_linear.eval()

            merged_ret = pd.DataFrame()
            for batch_idx, data in enumerate(test_loader):            
                data = data.to(torch.float32).to(self.device)

                zlist=torch.reshape(model_graph(mat_G, data), (1, -1))      
                output_frac = model_linear(zlist)

                partial_ret = (output_frac.cpu().detach().clone().numpy()).reshape((-1, 1))
                partial_ret = pd.DataFrame(partial_ret)
                merged_ret = pd.concat([merged_ret, partial_ret])

            final_ret = pd.concat([final_ret, merged_ret], axis=1)
        
        # for debuging
        final_ret.to_csv(f"{out_dir}/prediction_frac.csv", index=False, header=list(tot_cell_list))


    def test(
        self,
        out_dir,
        expression,
        fraction,
        marker=None,
        sc_folder=None,
        model_folder=None
    ):
        # tot_cell_list = marker.keys() 
        tot_cell_list = ({"B cells":[],"T CD4 naive cells":[],"T CD8 effector cells":[],"Macrophages M1":[],"Monocytes":[],"NK cells":[]}).keys()

        loss_cross_type = 0
        plot_info_dict = {}
        final_ret = pd.DataFrame()
        for cell_idx, cell in enumerate(tot_cell_list):
            sel_gene = marker[cell]
            mat_G, num, mar_scref_inter_gene = get_G(cell, sel_gene, sc_folder)
            mat_G = mat_G.to(self.device)

            print("\n\nmar_scref_inter_gene", len(mar_scref_inter_gene))
            print("simple filter", (expression.loc[:, expression.columns.isin(mar_scref_inter_gene)]).shape)
            # input_bulk = expression.loc[:, expression.columns.isin(mar_scref_inter_gene)]
            input_bulk = select_gene(expression, mar_scref_inter_gene)
            print("inputbulk", input_bulk.shape, "\n\n")
            test_data = input_bulk.values
            test_label = fraction[cell].values

            test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(test_data), torch.FloatTensor(test_label))
            test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = 1, shuffle = False)

            model_graph = ChebNet(num,num,num,2, device=self.device).to(self.device)
            model_graph.load_state_dict(torch.load(f'{model_folder}/graph_{cell}.pt'))
            model_linear = LinearModel(num).to(self.device)
            model_linear.load_state_dict(torch.load(f'{model_folder}/linear_{cell}.pt'))
            model_graph.eval(); model_linear.eval()

            loss_f_sum = 0
            plot_info_dict[cell] = {
                "mse_loss": [],
                "frac_pred": [],
                "frac_truth": []
            }
            merged_ret = pd.DataFrame()
            for batch_idx, (data, target) in tqdm(enumerate(test_loader), leave=False):
                target = torch.reshape(target,(1, 1))
            
                data = data.to(torch.float32).to(self.device)
                target = target.to(torch.float32).to(self.device)

                zlist=torch.reshape(model_graph(mat_G,data), (1, -1))      
                output_frac = model_linear(zlist)

                loss_f = ((output_frac-target)**2).sum(); loss_f_sum += loss_f
                plot_info_dict[cell]["mse_loss"].append(loss_f.data.cpu().detach().clone().numpy().item())
                plot_info_dict[cell]["frac_pred"].append(output_frac.cpu().detach().clone().numpy().item())
                plot_info_dict[cell]["frac_truth"].append(target.cpu().detach().clone().numpy().item())

                partial_ret = (output_frac.cpu().detach().clone().numpy()).reshape((-1, 1))
                partial_ret = pd.DataFrame(partial_ret)
                merged_ret = pd.concat([merged_ret, partial_ret])

            print(f"{cell_idx+1} Loss of the model for {cell} cell type:", loss_f_sum / len(test_loader))
            pearson = pearsonr(plot_info_dict[cell]["frac_pred"], plot_info_dict[cell]["frac_truth"])
            print(f"pearsonR {cell}: {pearson}")
            loss_cross_type += loss_f_sum / len(test_loader)

            final_ret = pd.concat([final_ret, merged_ret], axis=1)
        
        print(f'Average loss for deconvolution:', loss_cross_type / len(tot_cell_list))

        for c in tot_cell_list:
            with open(f'{out_dir}/plot/test_plot_info_{c}.json', 'w') as f: json.dump(plot_info_dict[c], f)
            # draw_deconv_eval_plots(plot_info_dict[c])

        # for debuging
        idx = []
        for i in range(final_ret.shape[0]): idx.append(f"Sample{i}")
        final_ret.index = idx
        final_ret.to_csv(f"{out_dir}/plot/test_ret.csv", header=list(tot_cell_list))


    def train(
        self,
        out_dir,
        expression=None,
        fraction=None,
        marker=None,
        sc_folder=None,
        batch_size = Const.BATCH_SIZE
    ):
        """
        out_dir: string, the directory for saving trained models.
        expression: string, needed if `mode` is `training`, the path of the bulk expression file.
        fraction: string, needed if `mode` is `training`, the path of the bulk fraction file.
        marker: string, needed if `mode` is `training`, the path of the gene marker file.
        sc_folder: string, needed if `mode` is `training`, the path of the folder containing single cell reference.
        """

        # checking
        if self.mode != Const.MODE_TRAINING:
            raise ValueError("This function can only be used under training mode.")
        
        check_paths(output_folder=out_dir)

        if expression.shape[0] != fraction.shape[0]:
            raise ValueError(f"Please check the input, the shape of the expression file {expression.shape} does not match the one of fraction {fraction.shape}.")
        

        # tot_cell_list = marker.keys() 
        tot_cell_list = ({"Monocytes":[],"NK cells":[]}).keys()

        if torch.backends.mps.is_available():
            for cell in tot_cell_list:
                train_cell_loop_once(cell, marker, expression, fraction, batch_size, sc_folder, out_dir, self.device)
        else:
            for cell in tot_cell_list:
                train_cell_loop_once(cell, marker, expression, fraction, batch_size, sc_folder, out_dir, self.device)
            # TODO: GPU parallel for CUDA
            # naive_parallel(train_cell_loop_once, [(cell, marker, expression, fraction, batch_size, sc_folder, out_dir, self.device) for cell in tot_cell_list])


# for quick testing
if __name__ == "__main__":
    print(f"chebnet mode {Const.CHEB_MODE}")
    deconv = GraphDeconv(mode="training")

    training_expression = pd.read_csv("./output/reftest/sti_data/CLL_GSE142744/CLL_GSE142744_expression_training.csv")
    training_fraction = pd.read_csv("./output/reftest/sti_data/CLL_GSE142744/CLL_GSE142744_fraction_training.csv")
    
    testing_expression = pd.read_csv("./output/reftest/sti_data/CLL_GSE142744/CLL_GSE142744_expression.csv")
    testing_fraction = pd.read_csv("./output/reftest/sti_data/CLL_GSE142744/CLL_GSE142744_fraction.csv")
    
    # marker = pd.read_csv("./output/gene_marker20230616.csv").to_dict('list')
    marker = pd.read_csv("./output/marker_gene_barplot_20230701.csv").to_dict('list')
    # marker = pd.read_csv("./output/pt/gene_marker20221228.csv").to_dict('list')
    out_dir = "./output"
    deconv.train(out_dir=out_dir,
                expression=training_expression,
                fraction=training_fraction,
                marker=marker,
                sc_folder=out_dir+"/PBMC_30K/refsc/"
                )
    deconv.test(out_dir=out_dir,
                expression=testing_expression,
                fraction=testing_fraction,
                marker=marker,
                sc_folder=out_dir+"/PBMC_30K/refsc/",
                model_folder=out_dir+"/model"
                )

    # deconv.test(out_dir=out_dir,
    #         expression=testing_expression,
    #         fraction=testing_fraction,
    #         marker=marker,
    #         sc_folder=out_dir+"/pt/single_new/",
    #         model_folder=out_dir+"/pt/models/pt/models"
    #         )


    # deconv.fit(out_dir=out_dir,
    #             expression=testing_expression,
    #             marker=marker,
    #             sc_folder=out_dir+"/NSCLC/refsc/",
    #             model_folder=out_dir+"/model"
    #             )

    # deconv.train(out_dir=out_dir,
    #             expression=training_expression,
    #             fraction=training_fraction,
    #             marker=marker,
    #             sc_folder=out_dir+"/transfer_testing/SKCM_small/full_dataset/ref_sc/",
    #             )
    # deconv.test(out_dir=out_dir,
    #             expression=testing_expression,
    #             fraction=testing_fraction,
    #             marker=marker,
    #             sc_folder=out_dir+"/transfer_testing/SKCM_small/full_dataset/ref_sc/",
    #             # model_folder=out_dir+"/transfer_testing/SKCM_small/model_small"
    #             model_folder=out_dir+"/transfer_testing/SKCM_small/model_full"
    #             )

    # deconv.train(out_dir=out_dir,
    #             expression=training_expression,
    #             fraction=training_fraction,
    #             marker=marker,
    #             sc_folder=out_dir+"/SKCM/refsc/",
    #             )
    # deconv.test(out_dir=out_dir,
    #             expression=testing_expression,
    #             fraction=testing_fraction,
    #             marker=marker,
    #             sc_folder=out_dir+"/SKCM/refsc/",
    #             model_folder=out_dir+"/transfer_testing/SKCM_small/model_transfer"
    #            )
