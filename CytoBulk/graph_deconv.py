import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import multiprocessing as mp
from tqdm import tqdm
from typing import Callable, Generator
import torch.distributed as dist

from utils import check_paths


class Const:
    """
    Some constants used in the class.
    """
    MODE_TRAINING = "training"
    MODE_PREDICTION = "prediction"
    SAMPLE_COL = "Unnamed: 0"
    GENE_SYMBOL_COL = "GeneSymbol"
    BATCH_SIZE = 100
    LEARNING_RATE = 0.0005
    EPOCH_NUM = 60
    SEED = 20230602


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
            torch.nn.Linear(64,1),
            torch.nn.ReLU(),
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
        nn.init.xavier_normal_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_c))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

        self.K = K + 1

    def forward(self, inputs, graph):
        L = ChebConv.get_laplacian(graph, self.normalize)
        
        L = L.cpu()
        lam, _u = np.linalg.eig(L)
        lam = torch.FloatTensor(lam)
        lam = lam.to(self.device)
        lam= torch.diag(lam)
        lam = 2*((lam - torch.min(lam).to(self.device)) / (torch.max(lam).to(self.device) - torch.min(lam).to(self.device))) - torch.eye(lam.size(0)).to(self.device)
        
        mul_L = self.cheb_polynomial(lam).unsqueeze(1)
        result = torch.matmul(inputs, mul_L)
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
        self.act = nn.ReLU()

    def forward(self, graph,data):
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
            mat_Y_ori = pd.read_csv(y_path, sep="\t") # NOTE: hard coded as "\t"
            mat_Y_ori = mat_Y_ori[mat_Y_ori[Const.GENE_SYMBOL_COL].isin(sel_gene)] 
            mat_Y = mat_Y_ori.drop(Const.GENE_SYMBOL_COL,axis=1)
            mat_Y = torch.FloatTensor(mat_Y.values)
            mat_W = mat_Y @ mat_Y.t()
            return mat_Y , mat_W
        sec_num = 1e-20
        mat_Y, mat_W = __get_mat_YW(
            f"{sc_folder}{cell_name}_scrna.txt", # NOTE: mind the path
            sel_gene
        )
        num = len(mat_W)
        mat_G = mat_W + sec_num*torch.eye(num) + sec_num*torch.ones(num, num)
        return mat_G, num

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
    input_bulk = expression.loc[:, expression.columns.isin(sel_gene)]
    # print(input_bulk.shape)
    input_bulk = input_bulk.sample(n=batch_size*(input_bulk.shape[1] // batch_size), axis=1, random_state=Const.SEED)
    train_data = input_bulk.values

    train_label = fraction[cell].values

    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(train_data), torch.FloatTensor(train_label))
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = Const.BATCH_SIZE, shuffle = True)

    mat_G, num = get_G(cell, sel_gene, sc_folder)
    mat_G = mat_G.to(device)

    model_graph = ChebNet(num,num,num,2, device=device).to(device)
    model_graph_optim = torch.optim.Adam(model_graph.parameters(), lr=Const.LEARNING_RATE, weight_decay = 1e-8)
    model_linear = LinearModel(num).to(device)
    model_linear_optim = torch.optim.Adam(model_linear.parameters(), lr=Const.LEARNING_RATE, weight_decay = 1e-8)

    should_break = False
    for epo in tqdm(range(Const.EPOCH_NUM), leave=False):
        for batch_idx, (data, target) in enumerate(train_loader):
            target = torch.reshape(target,(Const.BATCH_SIZE, 1))
    
            data = data.to(torch.float32).to(device)
            target = target.to(torch.float32).to(device)
            model_graph_optim.zero_grad()
            model_linear_optim.zero_grad()

            zlist=torch.reshape(model_graph(mat_G,data), (Const.BATCH_SIZE, -1))      
            output_frac = model_linear(zlist)

            loss_f = ((output_frac-target)**2).sum() / 1 / Const.BATCH_SIZE

            if batch_idx == 1:
                if epo>20 and loss_f<0.01:
                    should_break = True; break
            loss_f.backward()

            model_graph_optim.step()
            model_linear_optim.step()

        if should_break:
            print(cell, epo, loss_f)
            break

    print(f"Saving {cell} model...", end=" ")
    torch.save(model_graph.state_dict(),f"{out_dir}/graph_{cell}.pt")
    torch.save(model_linear.state_dict(),f"{out_dir}/linear_{cell}.pt")
    print("Done.")

class GraphDeconv:
    def __init__(
            self,
            cell_num=200,
            mode=Const.MODE_PREDICTION,
            use_gpu=False
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
        tot_cell_list = marker.keys() 

        final_ret = pd.DataFrame()
        for cell in tqdm(tot_cell_list, leave=False):
            sel_gene = marker[cell]
            input_bulk = expression.loc[:, expression.columns.isin(sel_gene)]
            test_data = input_bulk.values

            test_dataset = InferDataset(torch.FloatTensor(test_data))
            test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = 1, shuffle = False)

            mat_G, num = get_G(cell, sel_gene, sc_folder)
            mat_G = mat_G.to(self.device)

            model_graph = ChebNet(num,num,num,2, device=self.device).to(self.device)
            model_graph.load_state_dict(torch.load(f'{model_folder}/graph_{cell}.pt'))
            model_linear = LinearModel(num).to(self.device)
            model_linear.load_state_dict(torch.load(f'{model_folder}/linear_{cell}.pt'))

            merged_ret = pd.DataFrame()
            for batch_idx, data in enumerate(test_loader):            
                data = data.to(torch.float32).to(self.device)

                zlist=torch.reshape(model_graph(mat_G,data), (1, -1))      
                output_frac = model_linear(zlist)

                partial_ret = (output_frac.cpu().detach().clone().numpy()).reshape((-1, 1))
                partial_ret = pd.DataFrame(partial_ret)
                merged_ret = pd.concat([merged_ret, partial_ret])

            final_ret = pd.concat([final_ret, merged_ret], axis=1)
        
        # for debuging
        print(type(tot_cell_list))
        final_ret.to_csv(f"{out_dir}/prediction_frac.csv", index=False, header=tot_cell_list)


    def test(
        self,
        out_dir,
        expression,
        fraction,
        marker=None,
        sc_folder=None,
        model_folder=None
    ):
        tot_cell_list = marker.keys() 

        loss_cross_type = 0
        final_ret = pd.DataFrame()
        for cell_idx, cell in enumerate(tot_cell_list):
            sel_gene = marker[cell]
            input_bulk = expression.loc[:, expression.columns.isin(sel_gene)]
            test_data = input_bulk.values
            test_label = fraction[cell].values

            test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(test_data), torch.FloatTensor(test_label))
            test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = 1, shuffle = False)

            mat_G, num = get_G(cell, sel_gene, sc_folder)
            mat_G = mat_G.to(self.device)

            model_graph = ChebNet(num,num,num,2, device=self.device).to(self.device)
            model_graph.load_state_dict(torch.load(f'{model_folder}/graph_{cell}.pt'))
            model_linear = LinearModel(num).to(self.device)
            model_linear.load_state_dict(torch.load(f'{model_folder}/linear_{cell}.pt'))

            loss_f_sum = 0
            merged_ret = pd.DataFrame()
            for batch_idx, (data, target) in tqdm(enumerate(test_loader), leave=False):
                target = torch.reshape(target,(1, 1))
            
                data = data.to(torch.float32).to(self.device)
                target = target.to(torch.float32).to(self.device)

                zlist=torch.reshape(model_graph(mat_G,data), (1, -1))      
                output_frac = model_linear(zlist)

                loss_f = ((output_frac-target)**2).sum(); loss_f_sum += loss_f

                partial_ret = (output_frac.cpu().detach().clone().numpy()).reshape((-1, 1))
                partial_ret = pd.DataFrame(partial_ret)
                merged_ret = pd.concat([merged_ret, partial_ret])

            print(f"{cell_idx+1} Loss of the model for {cell} cell type:", loss_f_sum / len(test_loader))
            loss_cross_type += loss_f_sum / len(test_loader)

            final_ret = pd.concat([final_ret, merged_ret], axis=1)
        
        print(f'Average loss for deconvolution:', loss_cross_type / len(tot_cell_list))
        # for debuging
        print("fererer", tot_cell_list)
        final_ret.to_csv(f"{out_dir}/testing_data/test_ret.csv", index=False, header=tot_cell_list)

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
        

        tot_cell_list = marker.keys() 

        if torch.backends.mps.is_available():
            for cell in tot_cell_list:
                train_cell_loop_once(cell, marker, expression, fraction, batch_size, sc_folder, out_dir, self.device)
        else:
            naive_parallel(train_cell_loop_once, [(cell, marker, expression, fraction, batch_size, sc_folder, out_dir, self.device) for cell in tot_cell_list])
