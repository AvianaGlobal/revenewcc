import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual('hello', "hello")


if __name__ == '__main__':
    unittest.main()
