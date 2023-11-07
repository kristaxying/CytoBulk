import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import multiprocessing as mp
import json
import warnings
import sys
from tqdm import tqdm
from os.path import exists
from typing import Callable, Generator
import torch.distributed as dist
from scipy.stats import pearsonr
from ... import utils
from ... import get



class Const:
    """
    Some constants used in the class.
    """
    MODE_TRAINING = "training"
    MODE_PREDICTION = "prediction"
    SAMPLE_COL = "sample_name"
    GENE_SYMBOL_COL = "GeneSymbol"
    BATCH_SIZE = 64
    LEARNING_RATE = 0.005
    EPOCH_NUM = 25
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
    

class GraphConv(nn.Module):
    def __init__(self, in_c, out_c, K, device, bias=True, normalize=True):
        super(GraphConv, self).__init__()
        self.device = device
        self.normalize = normalize

        self.weight = nn.Parameter(torch.Tensor(K + 1, 1, in_c, out_c))
        nn.init.orthogonal_(self.weight, gain=nn.init.calculate_gain('leaky_relu', 0.4))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_c))
            # nn.init.zeros_(self.bias)
            nn.init.orthogonal_(self.bias, gain=nn.init.calculate_gain('leaky_relu', 0.4))
        else:
            self.register_parameter("bias", None)

        self.K = K + 1

    def forward(self, inputs, graph, mode=Const.CHEB_MODE):
        L = GraphConv.get_laplacian(graph, self.normalize)
        
        L = L.cpu()
        lam, u = np.linalg.eig(L)
        lam = torch.FloatTensor(lam)
        lam = lam.to(self.device)
        lam= torch.diag(lam)
        u = torch.FloatTensor(u).to(self.device)
        lam = 2*((lam - torch.min(lam).to(self.device)) / (torch.max(lam).to(self.device) - torch.min(lam).to(self.device))) - torch.eye(lam.size(0)).to(self.device)
        
        mul_L = self.cheb_polynomial(lam).unsqueeze(1)

        if mode == 0:
            result = torch.matmul(inputs, mul_L)
        elif mode == 1:
            result = torch.matmul(inputs, mul_L)
            result = torch.matmul(result, u.t())
        elif mode == 2:
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


class GraphNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c, K, device):
        super(GraphNet, self).__init__()
        self.conv1 = GraphConv(in_c=in_c, out_c=hid_c, K=K, device=device)
        self.conv2 = GraphConv(in_c=hid_c, out_c=out_c, K=K, device=device)
        self.act = nn.ELU()

    def forward(self, graph, data):
        graph_data = graph
        flow_x = data

        B, N = flow_x.size(0), flow_x.size(1)

        flow_x = flow_x.view(B, 1, N)

        output_1 = self.act(self.conv1(flow_x, graph_data))
        output_2 = self.act(self.conv2(output_1, graph_data))

        return output_2

def change_lr(optim, new_lr):
    for g in optim.param_groups:
        g['lr'] = new_lr

class InferDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        
    def __getitem__(self, index):
        x = self.data[index]
        return x
    
    def __len__(self):
        return len(self.data)

def get_G(cell_name, sel_gene, sc_adata,annotation_key):
        def _get_mat_YW(sc_df):
            mat_Y = torch.FloatTensor(sc_df.values)
            mat_W = mat_Y @ mat_Y.t()
            return mat_W

        sec_num = 1e-20
        sub_adata = sc_adata[sc_adata.obs[annotation_key]==cell_name,sel_gene].copy()
        sub_df = get.count_data(sub_adata)
        mat_W = _get_mat_YW(sub_df)
        num = len(mat_W)
        mat_G = mat_W + sec_num*torch.eye(num) + sec_num*torch.ones(num, num)
        return mat_G, num

def select_gene(expression: pd.DataFrame, sel_gene: list):
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    ret_exp = pd.DataFrame(expression.iloc[:, 0])
    for gene in sel_gene:
        ret_exp[gene] = expression[gene] if gene in expression.columns else 0
    return ret_exp.iloc[:, 1:]

def train_cell_loop_once(cell, 
                        marker,
                        expression,
                        fraction,
                        batch_size,
                        sc_adata,
                        out_dir,
                        device,
                        annotation_key):
    if exists(f'{out_dir}/graph_{cell}.pt') and exists(f'{out_dir}/linear_{cell}.pt'):
        print(f"skipping trainning model for {cell}")
    else:
        print(f"Start training the model for {cell} cell type...")
        sel_gene = marker[cell]
        mat_G, num = get_G(cell, sel_gene, sc_adata,annotation_key)
        mat_G = mat_G.to(device)
        input_bulk = expression[sel_gene]
        #select_gene(expression, sel_gene)

        train_data = input_bulk.values
        train_label = fraction[cell].values

        full_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(train_data), torch.FloatTensor(train_label))
        train_size = int(batch_size * (0.85 * len(full_dataset) // batch_size))
        valid_size = len(full_dataset) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [train_size, valid_size])
        train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
        valid_loader = torch.utils.data.DataLoader(dataset = valid_dataset, batch_size = 1, shuffle = False)

        model_graph = GraphNet(num,num,num,2, device=device).to(device)
        model_graph_optim = torch.optim.Adam(model_graph.parameters(), lr=Const.LEARNING_RATE, weight_decay = 1e-8)
        model_graph_schelr = torch.optim.lr_scheduler.StepLR(model_graph_optim, 5, gamma=0.9)
        model_linear = LinearModel(num).to(device)
        model_linear_optim = torch.optim.Adam(model_linear.parameters(), lr=Const.LEARNING_RATE, weight_decay = 1e-8)
        model_linear_schelr = torch.optim.lr_scheduler.StepLR(model_linear_optim, 5, gamma=0.9)

        plot_info_dict = {"mse_loss": []}
        pre_pearson_r = -np.inf
        pre_loss = np.inf
        model_r = pre_pearson_r
        model_loss = pre_loss
        best_graph = None; best_linear = None
        graph_break = False
        linear_stop = False
        # for epo in tqdm(range(Const.EPOCH_NUM), leave=False):
        for epo in range(Const.EPOCH_NUM):
            model_graph.train(); model_linear.train()
            for _, (data, target) in enumerate(train_loader):
                target = torch.reshape(target,(batch_size, 1))
                data = data.to(torch.float32).to(device)
                target = target.to(torch.float32).to(device)
                model_graph_optim.zero_grad()
                model_linear_optim.zero_grad()
                zlist=torch.reshape(model_graph(mat_G, data), (batch_size, -1))      
                output_frac = model_linear(zlist)
                loss_f = ((output_frac-target)**2).sum() / 1 / batch_size
                plot_info_dict["mse_loss"].append(loss_f.data.cpu().detach().clone().numpy().item())
                loss_f.backward()

                if not graph_break: model_graph_optim.step()
                model_linear_optim.step()

            model_graph.eval(); model_linear.eval()
            with torch.no_grad():
                valid_cor_dict = {
                    "frac_pred": [],
                    "frac_truth": []
                }
                for _, (data, target) in enumerate(valid_loader):
                    target = torch.reshape(target,(1, 1))
                    data = data.to(torch.float32).to(device)
                    target = target.to(torch.float32).to(device)

                    zlist = torch.reshape(model_graph(mat_G, data), (1, -1))      
                    output_frac = model_linear(zlist)
                    loss_f = ((output_frac-target)**2).sum()
                    valid_cor_dict["frac_pred"].append(output_frac.cpu().detach().clone().numpy().item())
                    valid_cor_dict["frac_truth"].append(target.cpu().detach().clone().numpy().item())
                
                pearson_r, pearson_p = pearsonr(valid_cor_dict["frac_pred"], valid_cor_dict["frac_truth"])
                if epo==0:
                    pre_loss = loss_f
                    pre_pearson_r = pearson_r
                    model_r = pearson_r
                    model_loss = loss_f
                    best_graph = model_graph.state_dict()
                    best_linear = model_linear.state_dict()
                else:
                    if pearson_r >= model_r and model_loss > loss_f-0.001:
                        model_loss = loss_f
                        model_r = pearson_r
                        best_graph = model_graph.state_dict()
                        best_linear = model_linear.state_dict()
                        #print(f"pearson_r = {model_r}...epo={epo}", end=" ")

            if not graph_break and epo>0:
                if pearson_r> 0.95 and epo<=20:
                    graph_break = True
                    change_lr(model_linear_optim, 0.0005)
                    print("stop graph training,linear training--0.0005")
                elif pearson_r> 0.9 and epo>5 and epo<=20:
                    graph_break = True
                    change_lr(model_linear_optim, 0.0015)
                    print("graph training--0.0005,linear training--0.0015")
                elif pearson_r> 0.85 and epo>10 and epo<=20:
                    change_lr(model_graph_optim, 0.0005)
                    change_lr(model_linear_optim, 0.0015)
                    print("graph training--0.0005,linear training--0.0015")
                elif epo>= 22:
                    graph_break = True
                    change_lr(model_linear_optim, 0.0005)
                    print("epo>15,graph training break,linear training--0.0005")
            elif graph_break and epo>22:
                if not linear_stop:
                    if loss_f < pre_loss and pre_loss-loss_f <0.001 and pre_loss-loss_f > 0.0001:
                        if loss_f < 0.001 and loss_f > 0.0005:
                            change_lr(model_linear_optim, 0.0005)
                            print("change linear training--0.0005")
                        elif loss_f <= 0.0005:
                            change_lr(model_linear_optim, 0.0001)
                            print("change linear training--0.0001")
                        elif loss_f < pre_loss and pre_loss-loss_f <= 0.0001:
                            linear_stop = True
                            print("stop linear training")
                if linear_stop and graph_break:
                    break

            print(f"epoch{epo}-pearsonR", pearson_r, pearson_p, loss_f)
            pre_pearson_r = pearson_r
            pre_loss = loss_f
            model_graph_schelr.step()
            model_linear_schelr.step()
        with open(f'{out_dir}/plot/train_plot_info_{cell}.json', 'w') as f: json.dump(plot_info_dict, f)
        print(f"Saving {cell} model...", end=" ")
        print(f"pearson_r = {model_r}...loss={model_loss}...", end=" ")
        # TODO: path check
        torch.save(best_graph, f"{out_dir}/graph_{cell}.pt")
        torch.save(best_linear, f"{out_dir}/linear_{cell}.pt")
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
        expression,
        marker=None,
        sc_adata=None,
        annotation_key = None,
        model_folder=None,
        out_dir='./'):
        
        utils.check_paths(output_folder=out_dir)

        device=self.device
        tot_cell_list = marker.keys()
        final_ret = pd.DataFrame()
        for cell in tqdm(tot_cell_list, leave=False):
            sel_gene = marker[cell]
            mat_G, num = get_G(cell, sel_gene, sc_adata,annotation_key)
            mat_G = mat_G.to(device)
            input_bulk = expression[sel_gene]
            test_data = input_bulk.values

            test_dataset = InferDataset(torch.FloatTensor(test_data))
            test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = 1, shuffle = False)

            model_graph = GraphNet(num,num,num,2, device=self.device).to(self.device)
            model_graph.load_state_dict(torch.load(f'{model_folder}/graph_{cell}.pt'))
            model_linear = LinearModel(num).to(self.device)
            model_linear.load_state_dict(torch.load(f'{model_folder}/linear_{cell}.pt'))
            model_graph.eval(); model_linear.eval()

            merged_ret = pd.DataFrame()
            for _, data in enumerate(test_loader):            
                data = data.to(torch.float32).to(self.device)

                zlist=torch.reshape(model_graph(mat_G, data), (1, -1))      
                output_frac = model_linear(zlist)

                partial_ret = (output_frac.cpu().detach().clone().numpy()).reshape((-1, 1))
                partial_ret = pd.DataFrame(partial_ret)
                merged_ret = pd.concat([merged_ret, partial_ret])

            final_ret = pd.concat([final_ret, merged_ret], axis=1)
        
        # for debuging
        final_ret.to_csv(f"{out_dir}/prediction_frac.csv", index=expression.index.tolist(), header=list(tot_cell_list))
        return final_ret


    def train(
        self,
        expression=None,
        fraction=None,
        marker=None,
        sc_adata=None,
        annotation_key = None,
        batch_size = Const.BATCH_SIZE,
        out_dir = "./"
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
        
        utils.check_paths(output_folder=out_dir)
        utils.check_paths(output_folder=out_dir+"./plot")

        if expression.shape[0] != fraction.shape[0]:
            raise ValueError(f"Please check the input, the shape of the expression file {expression.shape} \
                            does not match the one of fraction {fraction.shape}.")
        
        tot_cell_list = marker.keys()

        if torch.backends.mps.is_available():
            for cell in tot_cell_list:
                train_cell_loop_once(cell, marker, expression, fraction, batch_size, sc_adata, out_dir, self.device,annotation_key)
        else:
            for cell in tot_cell_list:
                train_cell_loop_once(cell, marker, expression, fraction, batch_size, sc_adata, out_dir, self.device,annotation_key)