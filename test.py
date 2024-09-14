import unittest
import app

class TestAPI(unittest.TestCase):

    def setUp(self):
        self.app = app.app.test_client()
        self.app.testing = True

    def test_home(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    def test_predict(self):
        response = self.app.post('/predict', json={'features': [8.3252, 41, 6.984127, 1, 322, 2.8, 37, 8.3252]})
        self.assertEqual(response.status_code, 200)
        self.assertIn('prediction', response.get_json())

if __name__ == '__main__':
    unittest.main()
