from torch.utils import data
from auxiliary_function import *
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WeiboDataset(data.Dataset):
    """
    Construct weibo dataset
    """

    def __init__(self, base_path, eid_list, label_dict, vocab, st_length):
        self.base_path = base_path
        self.event_path_list = [str(eid) + ".json" for eid in eid_list]
        self.label_dict = label_dict
        self.st_length = st_length
        self.vocab = vocab

    def __getitem__(self, index):
        event_full_path = os.path.join(self.base_path, self.event_path_list[index])
        node_info, adj, text_info, label = data_integration(event_full_path, self.label_dict, self.vocab,
                                                            self.st_length)
        return node_info, adj, text_info, label

    def __len__(self):
        return len(self.event_path_list)


def collate_(samples):
    """
    Collate function

    :param samples: samples
    :return: batch
    """

    node_feature, adj_matrix, text_info, label = map(list, zip(*samples))
    node_feature = [torch.FloatTensor(i).to(DEVICE) for i in node_feature]
    adj_matrix = [torch.FloatTensor(i).to(DEVICE) for i in adj_matrix]
    text_info = [torch.LongTensor(i).to(DEVICE) for i in text_info]
    return node_feature, adj_matrix, text_info, torch.LongTensor(np.asarray(label)).to(DEVICE)
