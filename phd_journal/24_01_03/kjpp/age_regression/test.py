import torch


def test(model, dataset):
    """
    Tests a regression graph neural network with MSE loss.
    :param model: A Torch Geometric model.
    :param dataset: A Torch Geometric dataset.
    """
    model.eval()

    loss_fn = torch.nn.MSELoss()

    loss_sum = 0
    for data in dataset:
        out = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_weight)
        loss = loss_fn(out, data.y)
        loss_sum += loss

        print(f'Test loss: {loss:.4f}')

    print("Average test loss: ", loss_sum / len(dataset) if len(dataset) > 0 else "N/A")
    print('Testing completed.')
