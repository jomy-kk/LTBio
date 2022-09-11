import unittest

from ltbio.ml.supervised import SupervisedTrainConditions


class SupervisedTrainConditionsTestCase(unittest.TestCase):
    """
    Any SupervisedTrainConditions should be tested for:
    - Its mandatory properties: optimizer, loss (only content)
    - Its train-test split: train_size, train_ratio, test_size, test_ratio (content and types)
    - Its optional properties: validation_ratio, epochs, learning_rate, batch_size, shuffle, epoch_shuffle (content and types)
    - Existence of hyperparameters
    - Printing ability
    """

    def test_create_only_with_mandatory_properties(self):
        conditions = SupervisedTrainConditions(optimizer='a', loss='b', train_size=200)
        self.assertEqual(conditions.optimizer, 'a')
        self.assertEqual(conditions.loss, 'b')
        self.assertEqual(conditions.train_size, 200)
        self.assertEqual(conditions.validation_ratio, None)
        self.assertEqual(conditions.epochs, None)
        self.assertEqual(conditions.learning_rate, None)
        self.assertEqual(conditions.batch_size, None)
        self.assertEqual(conditions.shuffle, False)
        self.assertEqual(conditions.epoch_shuffle, False)

        conditions = SupervisedTrainConditions(optimizer='a', loss='b', test_size=50)
        self.assertEqual(conditions.test_size, 50)

        conditions = SupervisedTrainConditions(optimizer='a', loss='b', train_ratio=0.75)
        self.assertEqual(conditions.train_ratio, 0.75)

        conditions = SupervisedTrainConditions(optimizer='a', loss='b', test_ratio=0.25)
        self.assertEqual(conditions.test_ratio, 0.25)

    def test_create_with_optional_properties(self):
        conditions = SupervisedTrainConditions(optimizer='a', loss='b', train_size=200,
                                               validation_ratio=0.2,
                                               epochs=100, learning_rate=0.05, batch_size=32,
                                               shuffle=True, epoch_shuffle=True)
        self.assertEqual(conditions.optimizer, 'a')
        self.assertEqual(conditions.loss, 'b')
        self.assertEqual(conditions.train_size, 200)
        self.assertEqual(conditions.validation_ratio, 0.2)
        self.assertEqual(conditions.epochs, 100)
        self.assertEqual(conditions.learning_rate, 0.05)
        self.assertEqual(conditions.batch_size, 32)
        self.assertEqual(conditions.shuffle, True)
        self.assertEqual(conditions.epoch_shuffle, True)

    def test_create_with_incorrect_types_raises_error(self):
        with self.assertRaises(ValueError):
            conditions = SupervisedTrainConditions(optimizer='a', loss='b', train_size=[])
        with self.assertRaises(ValueError):
            conditions = SupervisedTrainConditions(optimizer='a', loss='b', test_size=[])
        with self.assertRaises(ValueError):
            conditions = SupervisedTrainConditions(optimizer='a', loss='b', train_ratio=[])
        with self.assertRaises(ValueError):
            conditions = SupervisedTrainConditions(optimizer='a', loss='b', test_ratio=[])
        with self.assertRaises(ValueError):
            conditions = SupervisedTrainConditions(optimizer='a', loss='b', test_ratio=0.2, validation_ratio=[])
        with self.assertRaises(ValueError):
            conditions = SupervisedTrainConditions(optimizer='a', loss='b', test_ratio=0.2, epochs=[])
        with self.assertRaises(ValueError):
            conditions = SupervisedTrainConditions(optimizer='a', loss='b', test_ratio=0.2, batch_size=[])
        with self.assertRaises(ValueError):
            conditions = SupervisedTrainConditions(optimizer='a', loss='b', test_ratio=0.2, learning_rate=[])
        with self.assertRaises(TypeError):
            conditions = SupervisedTrainConditions(optimizer='a', loss='b', test_ratio=0.2, shuffle=2)
        with self.assertRaises(TypeError):
            conditions = SupervisedTrainConditions(optimizer='a', loss='b', test_ratio=0.2, epoch_shuffle=2)

    def test_create_with_illegal_values_raises_error(self):
        with self.assertRaises(ValueError):
            conditions = SupervisedTrainConditions(optimizer='a', loss='b', train_size=-400)
        with self.assertRaises(ValueError):
            conditions = SupervisedTrainConditions(optimizer='a', loss='b', train_size=0)
        with self.assertRaises(ValueError):
            conditions = SupervisedTrainConditions(optimizer='a', loss='b', test_size=-400)
        with self.assertRaises(ValueError):
            conditions = SupervisedTrainConditions(optimizer='a', loss='b', test_size=0)
        with self.assertRaises(ValueError):
            conditions = SupervisedTrainConditions(optimizer='a', loss='b', train_ratio=-0.1)
        with self.assertRaises(ValueError):
            conditions = SupervisedTrainConditions(optimizer='a', loss='b', train_ratio=0.0)
        with self.assertRaises(ValueError):
            conditions = SupervisedTrainConditions(optimizer='a', loss='b', train_ratio=1.0)
        with self.assertRaises(ValueError):
            conditions = SupervisedTrainConditions(optimizer='a', loss='b', test_ratio=-0.1)
        with self.assertRaises(ValueError):
            conditions = SupervisedTrainConditions(optimizer='a', loss='b', test_ratio=0.0)
        with self.assertRaises(ValueError):
            conditions = SupervisedTrainConditions(optimizer='a', loss='b', test_ratio=1.0)
        with self.assertRaises(ValueError):
            conditions = SupervisedTrainConditions(optimizer='a', loss='b', test_ratio=0.2, validation_ratio=-0.1)
        with self.assertRaises(ValueError):
            conditions = SupervisedTrainConditions(optimizer='a', loss='b', test_ratio=0.2, validation_ratio=0.0)
        with self.assertRaises(ValueError):
            conditions = SupervisedTrainConditions(optimizer='a', loss='b', test_ratio=0.2, validation_ratio=1.0)
        with self.assertRaises(ValueError):
            conditions = SupervisedTrainConditions(optimizer='a', loss='b', test_ratio=0.2, epochs=-400)
        with self.assertRaises(ValueError):
            conditions = SupervisedTrainConditions(optimizer='a', loss='b', test_ratio=0.2, epochs=0)
        with self.assertRaises(ValueError):
            conditions = SupervisedTrainConditions(optimizer='a', loss='b', test_ratio=0.2, batch_size=-400)
        with self.assertRaises(ValueError):
            conditions = SupervisedTrainConditions(optimizer='a', loss='b', test_ratio=0.2, batch_size=0)
        with self.assertRaises(ValueError):
            conditions = SupervisedTrainConditions(optimizer='a', loss='b', test_ratio=0.2, learning_rate=-0.1)
        with self.assertRaises(ValueError):
            conditions = SupervisedTrainConditions(optimizer='a', loss='b', test_ratio=0.2, learning_rate=0.0)
        with self.assertRaises(ValueError):
            conditions = SupervisedTrainConditions(optimizer='a', loss='b', test_ratio=0.2, learning_rate=1.0)

    def test_create_with_hyperparameters(self):
        conditions = SupervisedTrainConditions(optimizer='a', loss='b', train_size=200,
                                               validation_ratio=0.2,
                                               epochs=100, learning_rate=0.05, batch_size=32,
                                               shuffle=True, epoch_shuffle=True,
                                               hyper1=5, hyper2=True, hyper3=[3.2, 4.3])

        self.assertEqual(conditions.hyperparameters['hyper1'], 5)
        self.assertEqual(conditions.hyperparameters['hyper2'], True)
        self.assertEqual(conditions.hyperparameters['hyper3'], [3.2, 4.3])

    def test_print(self):
        conditions = SupervisedTrainConditions(optimizer='a', loss='b', train_size=200,
                                               validation_ratio=0.2,
                                               epochs=100, learning_rate=0.05, batch_size=32,
                                               shuffle=True, epoch_shuffle=True)
        print(conditions)

if __name__ == '__main__':
    unittest.main()
