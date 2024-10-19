from deep_learning_classifier import DeepLearningClassifier
from tsmodule import TimeSeriesClassificationModule
from lite import LITEModel

import lightning as L
import torch
"""
    LITEClassifier: A classifier built upon the LITEModel architecture for time-series classification.
    It leverages PyTorch Lightning for training, prediction, and testing processes.

    Parameters:
        n_classes (int, optional): Number of output classes for classification. Default is 7.
        input_shape (tuple): Shape of the input data as (n_samples, n_channels, n_timepoints).
        loss_fn (callable, optional): Loss function for training. If None, a default loss function will be used.
        batch_size (int, optional): Batch size for training. Default is 64.
        n_filters (int, optional): Number of filters in convolutional layers. Default is 32.
        kernel_size (int, optional): Base kernel size for convolutional layers. Default is 41.
        n_epochs (int, optional): Number of training epochs. Default is 1500.
        verbose (bool, optional): If True, prints training progress. Default is True.
        use_custom_filters (bool, optional): If True, uses custom filters in InceptionModule. Default is True.
        use_dilation (bool, optional): If True, applies dilation in FCN modules. Default is True.
        use_multiplexing (bool, optional): If True, uses multiple convolutions with different kernel sizes. Default is True.
"""
class LITEClassifier(DeepLearningClassifier):
    def __init__(
        self,
        n_classes=7,
        input_shape=None,
        loss_fn=None,
        batch_size=64,
        n_filters=32,
        kernel_size=41,
        n_epochs=1500,
        verbose=True,
        use_custom_filters=True,
        use_dilation=True,
        use_multiplexing=True,
    ):
        
        (n_samples, n_channels, n_timepoints) = input_shape
        self.length_TS = n_timepoints
        self.n_classes = n_classes

        self.verbose = verbose
        self.n_filters = n_filters
        self.use_custom_filters = use_custom_filters
        self.use_dilation = use_dilation
        self.use_multiplexing = use_multiplexing

        self.kernel_size = kernel_size - 1

        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.trainer = L.Trainer(max_epochs=self.n_epochs, accelerator="auto")

        self.model = LITEModel(
            length_TS=self.length_TS,
            n_classes=self.n_classes,
            n_filters=self.n_filters,
            kernel_size=self.kernel_size + 1,  # since kernel_size was decreased by 1 earlier
            use_custom_filters=self.use_custom_filters,
            use_dilation=self.use_dilation,
            use_multiplexing=self.use_multiplexing,
        )

        self.loss_fn = loss_fn
        self.module = TimeSeriesClassificationModule(self.model, self.loss_fn)
    
    def fit(self, datamodule):
        """
        Fit the model to the training data.

        Parameters
        ----------
        datamodule : LightningDataModule
            Data module containing the training data.

        Returns
        -------
        None
        """
        self.trainer.fit(model=self.module, datamodule=datamodule)
    
    def predict(self, datamodule):
        """
        Predict the classes for the input data.

        Parameters
        ----------
        datamodule : LightningDataModule
            Data module containing the data to predict.

        Returns
        -------
        pred : Tensor
            Predicted classes for the input data.
        """
        pred = self.trainer.predict(self.module, datamodule)
        pred = torch.cat(pred)
        _, pred = torch.topk(pred, k=1, dim=1)
        return pred
    
    def test(self, datamodule):
        """
        Test the model on the test data.

        Parameters
        ----------
        datamodule : LightningDataModule
            Data module containing the test data.

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries with test metrics.
        """
        return self.trainer.test(dataloaders=datamodule, model=self.module)
