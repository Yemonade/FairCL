import argparse

class DefaultArguments:
    def __init__(self):
        """Contains the default arguments passed to the main training function

        Description of parameters:
            dataset: str, Dataset to use
            metric: str, the measure of fairness
            seed: float, random seed
            alpha: float, hyperparameter in outer loop objective
        """
        self.dataset = "adult"
        self.metric = "eop"
        self.seed = None
        self.alpha = None

    def update(self, new_args):
        """Change the class attributes given new arguments.

        Args:
            new_args: dict with {'attribute': value, [...]}.
        """
        for attr, value in new_args.items():
            setattr(self, attr, value)


def get_args():
    """Get the command line arguments.
    Returns:
         args: Namespace object from the argument parser.
    """
    parser = argparse.ArgumentParser()
    default = DefaultArguments()

    parser.add_argument(
        '--dataset',
        type=str,
        default=default.dataset,
        help="name of the dataset"
    )

    parser.add_argument(
        '--metric',
        type=str,
        default=default.metric,
        help="eop or dp"
    )

    parser.add_argument(
        '--seed',
        type=float,
        default=default.seed,
        help="random seed"
    )

    parser.add_argument(
        '--alpha',
        type=float,
        default=default.alpha,
        help="hyperparameter in outer loop objective"
    )

    args = parser.parse_args()

    return args