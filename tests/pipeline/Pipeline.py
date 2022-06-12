import unittest

from src.pipeline.Pipeline import Pipeline


class PipelineTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        pass

    def test_create_pipeline(self):
        name = 'My first pipeline'
        pipeline = Pipeline(name='My first pipeline')

        self.assertEqual(pipeline.name, name)
        self.assertEqual(len(pipeline), 0)
        with self.assertRaises(AttributeError):
            x = pipeline.current_step

    def test_add_unit_to_pipeline(self):
        pass

if __name__ == '__main__':
    unittest.main()
