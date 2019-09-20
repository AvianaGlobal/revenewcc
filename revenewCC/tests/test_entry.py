import unittest


class TestImports(unittest.TestCase):
    def setUp(self) -> None:
        self.packages = ['fuzzywuzzy', 'Levenshtein', 'wxpython', 'pandas', 'xlsxwriter']

    def testPackage(self) -> None:
        try:
            import self.package
        except ImportError:
            print(f'{self.package} needs to be installed.')


if __name__ == '__main__':
    unittest.main()
