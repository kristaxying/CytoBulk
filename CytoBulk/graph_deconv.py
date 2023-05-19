import torch
import pandas as pd
import numpy as np
import torch.nn as nn

from tqdm import tqdm


class Const:
    """
    Some constants used in the class.
    """
    SAMPLE_COL = "Unnamed: 0"
    GENE_SYMBOL_COL = "GeneSymbol"
    BATCH_SIZE = 100
    LEARNING_RATE = 0.0005
    EPOCH_NUM = 40
    SEED = 3407


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
        
        L = L.to('cpu')
        lam, u = np.linalg.eig(L)
        lam = torch.FloatTensor(lam)
        lam = lam.to(self.device)
        lam= torch.diag(lam)
        u = torch.FloatTensor(u)
        u = u.to(self.device)
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

        torch.set_printoptions(profile="full")

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


# TODO
def configure_device(use_cpu):
    if use_cpu: return 'cpu'
    # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return "cpu"

class GraphDeconv:
    def __init__(
            self,
            cell_num=100,
            mode="prediction",
            expression_path=None,
            fraction_path=None,
            use_cpu=True
    ):
        """
            cell_num: int, the number of cell for each bulk sample.
            mode: string, prediction or training.
            expression_path: string, needed if `mode` is `training`, the path of the bulk expression file.
            fraction_path: string, needed if `mode` is `training`, the path of the bulk fraction file.
            use_cpu: bool, if `True`, the model will only use CPU otherwise, it will use CUDA or MPS.
        """
        self.cell_num = cell_num
        self.mode = mode

        if mode == "training":
            if (expression_path is None) or (fraction_path is None):
                raise ValueError("Please provide the expression file and the matching fraction file under training mode.")
        print("Reading expression data...")
        self.__expression = pd.read_csv(expression_path)
        print("Reading fraction data...")
        self.__fraction = pd.read_csv(fraction_path)
        if self.__expression.shape[0] != self.__fraction.shape[0]:
            raise ValueError(f"Please check the input, the shape of the expression file {self.expression.shape} does not match the one of fraction {self.fraction.shape}.")
        
        self.device = 'cpu' # TODO
    
    @staticmethod
    def __get_mat_YW(y_path, sel_gene):
        mat_Y_ori = pd.read_csv(y_path, sep="\t") # NOTE: hard coded as "\t"
        mat_Y_ori = mat_Y_ori[mat_Y_ori[Const.GENE_SYMBOL_COL].isin(sel_gene)] 
        mat_Y = mat_Y_ori.drop(Const.GENE_SYMBOL_COL,axis=1)
        mat_Y = torch.FloatTensor(mat_Y.values)
        mat_W = mat_Y @ mat_Y.t()
        return mat_Y , mat_W

    def __get_G(self, cell_name, sel_gene):
        sec_num = 1e-20
        mat_Y, mat_W = self.__get_mat_YW(
            f'../output/cell_feature/{cell_name}_scrna.txt', # NOTE: mind the path
            sel_gene
        )
        num = len(mat_W)
        mat_G = mat_W + sec_num*torch.eye(num) + sec_num*torch.ones(num, num)
        return mat_G, num

    def fit():
        """
            
        """

    def train(
        self,
        out_dir,
        batch_size = Const.BATCH_SIZE
    ):
        marker = pd.read_csv('../output/marker_gene.csv') # NOTE: check if needed
        tot_cell_list = list(marker.columns)[1:]

        for cell in tot_cell_list:
            print(f'Start training {cell}...')

            sel_gene = marker[cell].dropna()
            input_bulk = self.__expression.loc[:, self.__expression.columns.isin(sel_gene)]
            # print(input_bulk.shape)
            input_bulk = input_bulk.sample(n=batch_size*(input_bulk.shape[1] // batch_size), axis=1, random_state=Const.SEED)
            train_data = input_bulk.values

            train_label = self.__fraction[cell].values

            train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(train_data), torch.FloatTensor(train_label))
            train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = Const.BATCH_SIZE, shuffle = True)

            mat_G, num = self.__get_G(cell, sel_gene)
            mat_G = mat_G.to(self.device)

            model_graph = ChebNet(num,num,num,2, device=self.device).to(self.device)
            model_graph_optim = torch.optim.Adam(model_graph.parameters(), lr=Const.LEARNING_RATE, weight_decay = 1e-8)
            model_linear = LinearModel(num).to(self.device)
            model_linear_optim = torch.optim.Adam(model_linear.parameters(), lr=Const.LEARNING_RATE, weight_decay = 1e-8)

            should_break = False
            for epo in tqdm(range(Const.EPOCH_NUM), leave=False):
                for batch_idx, (data, target) in enumerate(train_loader):
                    target = torch.reshape(target,(Const.BATCH_SIZE, 1))
            
                    data = data.to(torch.float32).to(self.device)
                    target = target.to(torch.float32).to(self.device)
                    model_graph_optim.zero_grad()
                    model_linear_optim.zero_grad()
                    temp = model_graph(mat_G,data).view(Const.BATCH_SIZE, -1)

                    zlist=torch.reshape(temp,(Const.BATCH_SIZE, -1))      
                    fraction = model_linear(zlist)

                    loss_f = ((fraction-target)**2).sum() / 1 / Const.BATCH_SIZE

                    if batch_idx == 1:
                        # print(epo, "loss", loss_f)
                        if epo>20 and loss_f<0.01:
                            should_break = True; break
                    loss_f.backward()

                    model_graph_optim.step()
                    model_linear_optim.step()

                if should_break: break

            print(f'Saving {cell} model...', end=' ')
            torch.save(model_graph.state_dict(),f'{out_dir}/graph_{cell}.pt')
            torch.save(model_linear.state_dict(),f'{out_dir}/linear_{cell}.pt')
            print(f'Done.')


# for quick testing
if __name__ == "__main__":
    deconv = GraphDeconv(
        mode="training",
        expression_path="../output/training_data/expression.csv",
        fraction_path="../output/training_data/fraction.csv"
    )
    deconv.train(out_dir="../output/model")
