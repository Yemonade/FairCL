import time


from argparser import get_args
from dataset import fetch_data
from eval import Evaluator
from utils import set_seed


def main(args):
    """Main Function for the full training loop.

    Args:
        args: Namespace object from the argument parser.
    """

    tik = time.time()
    if args.seed is not None:
        set_seed(args.seed)
    else:
        set_seed(42)

    """ Initialization """
    data = fetch_data(args.dataset)
    # models = ...
    val_evaluator, test_evaluator = Evaluator(data.s_val, "val"), Evaluator(data.s_test, "test")


if __name__ == "__main__":
    # Get the default and command line arguments.
    args = get_args()

    # Run the models.
    main(args)