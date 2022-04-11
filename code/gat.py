import torch
import torch.nn as nn
import torch.nn.functional as F
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextCNNLayer(nn.Module):
    """
    TextCNN Layer
    """

    def __init__(self, embed_num, embed_dim, filter_dim, filter_size, output_size, st_length):
        super(TextCNNLayer, self).__init__()

        self.embed = nn.Embedding(embed_num, embed_dim)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, filter_dim, (h, embed_dim)),
                nn.ReLU(),
                nn.MaxPool2d((st_length - h + 1, 1))
            ) for h in filter_size
        ])
        self.linear = nn.Linear(len(filter_size) * filter_dim, output_size)

    def forward(self, x):
        embed_x = self.embed(x).unsqueeze(1)
        out = [conv(embed_x) for conv in self.convs]
        out = torch.cat(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


class GraphAttentionLayer(nn.Module):
    """
    GAT layer
    """

    def __init__(self, in_features, out_features, dropout, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)).to(DEVICE))
        # Glorot initialization
        nn.init.xavier_uniform_(self.W.data)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)).to(DEVICE))

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        """
        inp: input_features_matrix list([N, in_features] * batch_size)
        adj: adjacency matrix list([N, N] * batch_size)
        """
        h = [torch.mm(i, self.W) for i in inp]
        N = [i.size()[0] for i in h]

        result_list = []

        for item, n, adj_matrix in zip(h, N, adj):
            a_input = torch.cat([item.repeat(1, n).view(n * n, -1), item.repeat(n, 1)], dim=1).\
                view(n, -1, 2 * self.out_features)
            e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

            zero_mat = -1e12 * torch.ones_like(e)
            attention = torch.where(adj_matrix > 0, e, zero_mat)
            attention = F.softmax(attention, dim=1)
            attention = F.dropout(attention, self.dropout)
            h_prime = torch.matmul(attention, item)
            result_list.append(F.elu(h_prime))

        return result_list, N


class GraphAttentionEmbeddingLayer(GraphAttentionLayer):
    """
    GAT layer with embedding layer
    """

    def __init__(self, in_features, out_features, dropout, alpha, embed_num, embed_dim, filter_dim, filter_size,
                 output_size, st_length):
        super(GraphAttentionEmbeddingLayer, self).__init__(in_features, out_features, dropout, alpha)
        nn.init.xavier_uniform_(self.W.data)
        self.text_cnn = TextCNNLayer(embed_num, embed_dim, filter_dim, filter_size, output_size, st_length)

    def forward(self, inp, adj, text_inp):
        """
        inp: input_features_matrix list([N, in_features] * batch_size)
        adj: adjacency matrix list([N, N] * batch_size)
        text_inp: input_feature_matrix list([N, st_length] * batch_id)
        """

        text_out = [self.text_cnn(i) for i in text_inp]
        h = [torch.cat([i, j], 1) for i, j in zip(inp, text_out)]
        h = [torch.mm(i, self.W) for i in h]
        N = [i.size()[0] for i in h]

        result_list = []

        for item, n, adj_matrix in zip(h, N, adj):
            a_input = torch.cat([item.repeat(1, n).view(n * n, -1), item.repeat(n, 1)], dim=1).view(n, -1,
                                                                                                    2 * self.out_features)
            e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

            zero_mat = -1e12 * torch.ones_like(e)
            attention = torch.where(adj_matrix > 0, e, zero_mat)
            attention = F.softmax(attention, dim=1)
            attention = F.dropout(attention, self.dropout)
            h_prime = torch.matmul(attention, item)
            result_list.append(F.elu(h_prime))

        return result_list, N


class TextCNNGAT(nn.Module):
    def __init__(self, n_feat, n_hid_1, n_hid_2, n_class, dropout, alpha, embed_num, embed_dim, filter_dim, filter_size,
                 output_size, st_length):
        super(TextCNNGAT, self).__init__()
        self.dropout = dropout
        self.attentions_1 = GraphAttentionEmbeddingLayer(n_feat, n_hid_1, dropout, alpha, embed_num, embed_dim,
                                                         filter_dim, filter_size, output_size, st_length)
        self.attentions_2 = GraphAttentionLayer(n_hid_1, n_hid_2, dropout, alpha)
        self.fc = nn.Linear(n_hid_2, n_class)

    def forward(self, x, adj, text_info):
        x, N = self.attentions_1(x, adj, text_info)
        x, N = self.attentions_2(x, adj)
        x = torch.cat([torch.mean(i, 0).unsqueeze(0) for i in x], 0)
        out = F.dropout(x, self.dropout)
        out = self.fc(out)

        return F.log_softmax(out, 0), x
