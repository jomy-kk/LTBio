###################################

# IT - PreEpiSeizures

# Package: ml
# File: SupervisedModel
# Description: Abstract Class representing a generic machine learning model.

# Contributors: João Saraiva
# Created: 4/05/2022

###################################

from src.ml.models.SupervisedModel import SupervisedModel
from src.ml.trainers.SupervisedTrainConditions import SupervisedTrainConditions


class SkLearnModel(SupervisedModel):

    def __init__(self, design, name: str = None, version: int = None):
        super().__init__(design, name, version)

    def setup(self, train_conditions:SupervisedTrainConditions, **kwargs):
        params = train_conditions.parameters
        if 'train_size' in params:
            del params['train_size']
        if 'test_size' in params:
            del params['test_size']
        if 'shuffle' in params:
            del params['shuffle']
        self.design.set_params(**params)

    def train(self, object, target):
        self.design.fit(object, target)

    def test(self, object, target=None):
        if target is None:
            return self.design.predict(object)
        else:
            print(self.design.score(object, target))
            return self.design.score(object, target)

    @property
    def trainable_parameters(self):
        return self.design.get_parms()
