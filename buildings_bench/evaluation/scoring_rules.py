import torch 
import math 
from buildings_bench.evaluation.metrics import MetricType
from buildings_bench.evaluation.metrics import BuildingsBenchMetric


class ScoringRule(BuildingsBenchMetric):
    """An abstract class for all scoring rules.
    """

    def __init__(self, name: str):
        super().__init__(name, MetricType.HOUR_OF_DAY)

    def reset(self):
        self.value = None

    def __call__(self, **kwargs):
        raise NotImplementedError()
    
    def mean(self):
        if self.value is None:
            return
        value = torch.stack(self.value,0)
        self.value = torch.mean(value, 0)    


class RankedProbabilityScore(ScoringRule):
    """A class that calculates the ranked probability score (RPS) metric
    for categorical distributions."""

    def __init__(self):
        super().__init__(name='rps')

    def rps(self, y_true, y_pred_logits, centroids) -> None:
        """A PyTorch method that calculates the ranked probability score metric
           for categorical distributions.

           Since the bin values are centroids of clusters along the real line,
           we have to compute the width of the bins by summing the distance to
           the left and right centroids of the bin (divided by 2), except for
           the first and last bins, where we only need to sum the distance to
           the right centroid of the first bin and the left centroid of the
           last bin, respectively.

        Args:
            y_true (torch.Tensor): of shape [batch_size, seq_len, 1] categorical labels
            y_pred_logits (torch.Tensor): of shape [batch_size, seq_len, vocab_size] logits
            centroids (torch.Tensor): of shape [vocab_size]
        """
        # Convert class labels y_true to one hot vectors [batch_size, seq_len, vocab_size]
        y_true = torch.nn.functional.one_hot(y_true.squeeze(2).long(),
                                             num_classes=centroids.shape[0]).to(y_pred_logits.device)
        # Sort the values, logits, and y_true
        centroids, indices = torch.sort(centroids, dim=-1)
        y_pred_logits = y_pred_logits[:, :, indices]
        y_true = y_true[:, :, indices]

        softmax_preds = torch.softmax(y_pred_logits, dim=-1)
        # Calculate the cumulative distribution function (CDF) of the predictions
        cdf = torch.cumsum(softmax_preds, dim=-1)

        y_true_cdf = torch.cumsum(y_true, dim=-1)
        # Calculate the difference between the CDF and the true values
        square = torch.square(cdf - y_true_cdf)
        
        # Calculate the widths of the bins:
        # we need to calculate
        # half the distance to the right centroid and left centroid.
        centroid_dist = centroids[1:] - centroids[:-1]
        half_dists = centroid_dist / 2
        widths = torch.unsqueeze(half_dists[1:] + half_dists[:-1], dim=0)
        widths = torch.cat([
            centroids[0].view(1, 1) + half_dists[0].view(1, 1),
            widths,
            half_dists[-1].view(1, 1)
        ],dim=1)
        
        # Calculate the RPS    
        rps = torch.mean(torch.sum(square * widths, dim=-1), dim=0)  # [seq_len]
        if self.value is None:
            self.value = [rps]
        else:
            self.value += [rps]

    def __call__(self, true_continuous, y_true, y_pred_logits, centroids):
        self.rps(y_true, y_pred_logits, centroids)


class ContinuousRankedProbabilityScore(ScoringRule):
    """
    A class that calculates the Gaussian continuous ranked probability score (CRPS) metric.
    """
    def __init__(self):
        super().__init__(name = 'crps')

    def crps(self, true_continuous, y_pred_distribution_params) -> None:
        """Computes the Gaussian CRPS.

        Args:
            true_continuous (torch.Tensor): of shape [batch_size, seq_len, 1]
            y_pred_distribution_params (torch.Tensor): of shape [batch_size, seq_len, 2]
        """
        pred_mu = y_pred_distribution_params[:, :, 0].unsqueeze(-1)
        pred_sigma = y_pred_distribution_params[:, :, 1].unsqueeze(-1)
        
        # standardize the true values to N(0,1)
        true_continuous = (true_continuous - pred_mu) / pred_sigma
        # Calculate the cumulative distribution function (CDF) of the predictions
        cdf = 0.5 * (1 + torch.erf(true_continuous / math.sqrt(2)))
        # Calculate the pdf of the predictions
        pdf = torch.exp(-torch.square(true_continuous) / 2) / math.sqrt(2 * math.pi)
        # Calculate pi inv
        pi_inv = 1 / math.sqrt(math.pi)
        # CRPS
        crps = pred_sigma * ( true_continuous * (2 * cdf - 1) + 2 * pdf - pi_inv)
        crps = torch.mean(crps, dim=0).squeeze(1)  # [seq_len]
        if self.value is None:
            self.value = [crps]
        else:
            self.value += [crps]

    def __call__(self, true_continuous, y_true, y_pred_distribution_params, centroids):
        self.crps(true_continuous, y_pred_distribution_params)
