import torch
from pydgn.training.callback.metric import Classification
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, balanced_accuracy_score


class F1Score(Classification):
    r"""
    Implements the F1 score metric.
    """
    @property
    def name(self) -> str:
        return 'F1 Score'

    def compute_metric(self,
                       targets: torch.Tensor,
                       predictions: torch.Tensor) -> torch.tensor:
        if len(predictions.shape) == 1:
            predictions = (torch.sigmoid(predictions)>0.5).float()
        elif len(predictions.shape) == 2:
            predictions = torch.argmax(predictions, 1)
        else:
            raise NotImplementedError(f"Only shape 1-d or 2-d tensors are implemented, got {predictions.shape}")
        metric = torch.tensor(f1_score(targets, predictions))        
        return metric


class RocAucScore(Classification):
    r"""
    Implements the ROC AUC score metric.
    """
    @property
    def name(self) -> str:
        return 'ROC AUC Score'

    def compute_metric(self,
                       targets: torch.Tensor,
                       predictions: torch.Tensor) -> torch.tensor:
        predictions = torch.sigmoid(predictions)
        # if len(predictions.shape) == 1:
        #     predictions = (torch.sigmoid(predictions)>0.5).float()
        # elif len(predictions.shape) == 2:
        #     predictions = torch.argmax(predictions, 1)
        # else:
        #     raise NotImplementedError(f"Only shape 1-d or 2-d tensors are implemented, got {predictions.shape}")
        metric = torch.tensor(roc_auc_score(targets, predictions))
        return metric


class AccuracyScore(Classification):
    r"""
    Implements the Accuracy score metric.
    """ 
    @property
    def name(self) -> str:
        return 'Acc Score'

    def compute_metric(self,
                       targets: torch.Tensor,
                       predictions: torch.Tensor) -> torch.tensor:
        if len(predictions.shape) == 1:
            predictions = (torch.sigmoid(predictions)>0.5).float()
        elif len(predictions.shape) == 2:
            predictions = torch.argmax(predictions, 1)
        else:
            raise NotImplementedError(f"Only shape 1-d or 2-d tensors are implemented, got {predictions.shape}")
        metric = torch.tensor(accuracy_score(targets, predictions))
        return metric

class BalancedAccuracyScore(Classification):
    r"""
    Implements the Balanced Accuracy score metric.
    """ 
    @property
    def name(self) -> str:
        return 'Balanced Acc Score'

    def compute_metric(self,
                       targets: torch.Tensor,
                       predictions: torch.Tensor) -> torch.tensor:
        if len(predictions.shape) == 1:
            predictions = (torch.sigmoid(predictions)>0.5).float()
        elif len(predictions.shape) == 2:
            predictions = torch.argmax(predictions, 1)
        else:
            raise NotImplementedError(f"Only shape 1-d or 2-d tensors are implemented, got {predictions.shape}")
        metric = torch.tensor(balanced_accuracy_score(targets, predictions))
        return metric

class BinaryClassificationLoss(Classification):
    r"""
    Wrapper around :class:`torch.nn.BCEWithLogitsLoss`.
    """
    def __init__(self, use_as_loss=False, reduction='mean',
                 accumulate_over_epoch: bool=True, force_cpu: bool=True):
        super().__init__(use_as_loss=use_as_loss, reduction=reduction,
                         accumulate_over_epoch=accumulate_over_epoch, force_cpu=force_cpu)
        self.metric = torch.nn.BCEWithLogitsLoss(reduction=reduction)

    @property
    def name(self) -> str:
        return 'Binary Classification'

    def compute_metric(self,
                       targets: torch.Tensor,
                       predictions: torch.Tensor) -> torch.tensor:

        metric = self.metric(predictions, targets.float())
        return metric
