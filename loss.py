from secml.array import CArray
from secml.ml.classifiers.loss import CLossClassification
from secml.ml.classifiers.loss import CLossLogistic

from commons import Group


class CLossDisparateImpact(CLossClassification):
    """Surrogate function of disparate impact.

    Attributes
    ----------
    class_type : 'log'
    suitable_for : 'classification'
    """
    __class_type = 'dimp_log'

    def __init__(self, privileged_condition):
        self._privileged = CArray(privileged_condition)
        self._unprivileged = self._get_unprivileged(self._privileged)

    @property
    def unprivileged(self):
        """Provide unprivileged vector.

        Returns:
            CArray: a binary vector with 1 on unprivileged position and 0 on privileged ones.
        """
        return self._unprivileged

    @property
    def privileged(self):
        """Provide privileged vector.

        Returns:
            CArray: a binary vector with 1 on privileged position and 0 on unprivileged ones.
        """
        return self._privileged

    def _get_unprivileged(self, conditions):
        """Get unprivileged from condition vector.

        The condition vector holds all the privileged and unprivileged
        statuses - as a binary vector. This function returns 1 for all the
        unprivileged positions in the condition vector.

        Returns:
            CArray: a binary vector with 1 from the unprivileged position in
            the input binary vector.
        """
        unprivileged = CArray.zeros(conditions.size)
        unprivileged[conditions == Group.UNPRIVILEGED] = 1
        return unprivileged

    def loss(self, y_true, score, pos_label=1):
        """Computes loss_priv-loss_unpriv, which is what we aim to max"""
        p_priv = self._privileged.sum() / self._privileged.size  # proportion of privileged
        p_unpriv = self._unprivileged.sum() / self._unprivileged.size  # proportion of unprivileged
        # loss = (score >= 0) != y  # zero-one loss
        loss = CLossLogistic().loss(y_true=self._unprivileged, score=score)  # smoothed version
        loss[self._unprivileged ==1 ] *= -p_priv / p_unpriv  # rebalance class weights
        return loss

    def dloss(self, y_true, score, pos_label=1):
        """Computes the derivative of the loss vs score."""
        p_priv = self._privileged.sum() / self._privileged.size  # proportion of privileged
        p_unpriv = self._unprivileged.sum() / self._unprivileged.size  # proportion of unprivileged
        grad = CLossLogistic().dloss(self._unprivileged, score, pos_label)
        grad[self._unprivileged == 1] *= -p_priv / p_unpriv  # rebalance class weights
        return grad
