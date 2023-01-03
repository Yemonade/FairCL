import torch
import torch.nn.functional as F

from dataset import fetch_data
from models import FairBatch
from models.LogisticRegression import LogisticRegression, weights_init_normal


def run_epoch(model, train_features, labels, optimizer, criterion):
    """Trains the model with the given train data.

    Args:
        model: A torch model to train.
        train_features: A torch tensor indicating the train features.
        labels: A torch tensor indicating the true labels.
        optimizer: A torch optimizer.
        criterion: A torch criterion.

    Returns:
        loss value.
    """

    optimizer.zero_grad()

    label_predicted = model.forward(train_features)
    loss = criterion((F.tanh(label_predicted.squeeze()) + 1) / 2, (labels.squeeze() + 1) / 2)
    loss.backward()

    optimizer.step()

    return loss.item()

if __name__ == "__main__":
    name = "adult"
    data = fetch_data(name)

    model = LogisticRegression(3,1)
    model.apply(weights_init_normal)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999))
    criterion = torch.nn.BCELoss()

    losses = []

    # ---------------------
    #  Define FairBatch and DataLoader
    # ---------------------

    sampler = FairBatch(model, train_data.x, train_data.y, train_data.z, batch_size=100, alpha=0.005,
                        target_fairness='eqopp', replacement=False, seed=seed)
    train_loader = torch.utils.data.DataLoader(train_data, sampler=sampler, num_workers=0)

    # ---------------------
    #  Model training
    # ---------------------
    for epoch in range(300):

        tmp_loss = []

        for batch_idx, (data, target, z) in enumerate(train_loader):
            loss = run_epoch(model, data, target, optimizer, criterion)
            tmp_loss.append(loss)

        losses.append(sum(tmp_loss) / len(tmp_loss))

    tmp_test = test_model(model, xz_test, y_test, z_test)
    full_tests.append(tmp_test)

    print("  Test accuracy: {}, EO disparity: {}".format(tmp_test['Acc'], tmp_test['EO_Y1_diff']))
    print("----------------------------------------------------------------------")
    # %%
