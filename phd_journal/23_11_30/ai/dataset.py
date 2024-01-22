import torch
from numpy import ndarray
from torch import tensor
from torch_geometric.data import InMemoryDataset, Data

from graphs import to_edge_list, create_degree_node_features


class EEGRegressionDataset(InMemoryDataset):
    """
    A dataset of EEG connectivity graphs represented as Data objects, each associated with a float target.
    """

    def __init__(self, data: dict[str, tuple[ndarray, float]], root: str = '.'):
        super().__init__(root=root)
        data = sorted(data.items())
        self.subject_ids = [subject_id for subject_id, _ in data]  # e.g. ['C10', 'C11', ...]  # save them for later
        data = [x for _, x in data]  # e.g. [(matrix, 10.2), (matrix, 11.3), ...]
        data = [(self.to_data(matrix, target)) for matrix, target in data]  # e.g. [Data(...), Data(...), ...]
        self.data, self.slices = self.collate(data)

    @staticmethod
    def to_data(adjacency_matrix: ndarray, target: float) -> Data:
        """
        Converts an adjacency matrix to a Data object.
        :return: Data object containing edge_index, edge_weight, num_nodes, and y target.
        """

        assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1]  # square matrix

        # Conversion
        edge_list, edge_weights = to_edge_list(adjacency_matrix)

        # For node features use their degrees
        x = tensor(create_degree_node_features(adjacency_matrix), dtype=torch.float)

        return Data(edge_index=torch.tensor(edge_list, dtype=torch.long),
                    edge_weight=torch.tensor(edge_weights, dtype=torch.float),
                    num_nodes=adjacency_matrix.shape[0],
                    x=x, y=torch.tensor([target], dtype=torch.float))
