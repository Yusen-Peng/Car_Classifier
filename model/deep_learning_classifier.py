from signalts.classification.models.classifier import TimeSeriesClassifier
from tsmodule import TimeSeriesClassificationModule


from lightning import Trainer


from torch import nn

class DeepLearningClassifier(TimeSeriesClassifier):
    """Deep Learning Classifier for Time Series Data.
    
    This classifier utilizes deep learning models for classifying time series data. 
    It supports various models, with MLP (Multi-Layer Perceptron) being the default.
    
    Parameters
    ----------
    model_name : str, default='mlp'
        The name of the model to be used.
    loss_fn : nn.Module, default=None
        The loss function to be used for training.
    input_shape : tuple
        The shape of the input data (n_samples, n_channels, n_timepoints).
    n_classes : int, default=7
        The number of output classes.
    model_kwargs : dict, default=None
        Additional keyword arguments for the model.
    epochs : int, default=1000
        The number of training epochs.
    
    Attributes
    ----------
    name : str
        The name of the classifier.
    model_name : str
        The name of the model being used.
    epochs : int
        The number of training epochs.
    trainer : Trainer
        PyTorch Lightning Trainer instance.
    model : nn.Module
        The deep learning model instance.
    loss_fn : nn.Module
        The loss function for training.
    module : TimeSeriesClassificationModule
        Wrapper module combining model and loss function for training.
    """

    def __init__(self,model_name='mlp',loss_fn = None,input_shape=None,n_classes=7,model_kwargs=None,epochs=1000):
        self.name = 'Deep Learning Classifier'
        self.model_name = model_name
        self.epochs = epochs
        self.trainer = Trainer(max_epochs=self.epochs)

        (n_samples,n_channels,n_timepoints) = input_shape

        self.loss_fn = loss_fn

        self.module = TimeSeriesClassificationModule(self.model,self.loss_fn)

        

    def fit(self,datamodule):
        """Train the model using the provided data module.

        Parameters
        ----------
        datamodule : LightningDataModule
            The data module containing training data.
        """
        self.trainer.fit(model = self.module,datamodule=datamodule)
    
    def predict(self,datamodule):
        """Predict the output using the trained model and provided data module.

        Parameters
        ----------
        datamodule : LightningDataModule
            The data module containing data for prediction.

        Returns
        -------
        array-like
            The predictions made by the model.
        """
        return self.trainer.predict(self.module,datamodule)
    
    def test(self,datamodule):
        """Test the model using the provided data module.

        Parameters
        ----------
        datamodule : LightningDataModule
            The data module containing test data.

        Returns
        -------
        dict
            The test results.
        """
        return self.trainer.test(dataloaders=datamodule,model=self.module)
