
import logging

logger = logging.getLogger(__name__)


def graph_oof(oof_l1, oof_l2, y_true):
    from matplotlib import pyplot as plt
    x = oof_l1
    y = oof_l2 - oof_l1

    plt.scatter(x, y, c=y_true, alpha=0.5)
    plt.show()
