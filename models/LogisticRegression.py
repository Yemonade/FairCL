import torch
import torch.nn as nn


def weights_init_normal(m):
    """Initializes the weight and bias of the model.

    Args:
        m: A torch model to initialize.

    Returns:
        None.
    """

    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.2)
        torch.nn.init.constant_(m.bias.data, 0)


class LogisticRegression(nn.Module):
    """Logistic Regression (classifier).

    Attributes:
        model: A models consisting of torch components.
    """

    def __init__(self, n_in, n_out):
        """Initializes classifier with torch components."""

        super(LogisticRegression, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            """Defines a block with torch components.

                Args:
                    in_feat: An integer value for the size of the input feature.
                    out_feat: An integer value for the size of the output feature.
                    normalize: A boolean indicating whether normalization is needed.

                Returns:
                    The stacked layer.
            """

            layers = [nn.Linear(in_feat, out_feat)]

            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            #             *block(3, 32, normalize=False),
            #             nn.Linear(32, 1),
            #             nn.Tanh()
            nn.Linear(n_in, n_out)
        )

    def forward(self, input_data):
        """Defines a forward operation of the models.

        Args:
            input_data: The input data.

        Returns:
            The predicted label (y_hat) for the given input data.
        """

        output = self.model(input_data)
        return output

