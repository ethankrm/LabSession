import unittest
from ml_code.data_load import load_data

class TestDataLoader(unittest.TestCase):
    
    def test_load_data(self):
        data = load_data("D:/UMD Grad School/ENPM611/LabSession/data.csv")
        self.assertIsNotNone(data)
        self.assertEqual(len(data), 1470)

if __name__ == '__main__':
    unittest.main()