import unittest
from analysis import app, total_words, different_words, type_token_ratio, num_clauses

class TestAnalysis(unittest.TestCase):
    def test_total_words(self):
        # Test 1
        text = "This is a sample sentence."
        expected_result = 5
        result = total_words(text)
        self.assertEqual(result, expected_result)

        # Test 2
        text = " "
        expected_result = 0
        result = total_words(text)
        self.assertEqual(result, expected_result)

        # Test 3
        text = "The quick brown fox jumps over the lazy dog. 12345!@#$%"
        expected_result = 9
        result = total_words(text)
        self.assertEqual(result, expected_result)

        # Test 4
        text = ("In the heart of the bustling city, amidst the skyscrapers and bustling streets, there lies a quaint little cafe. "
             "Its exterior is unassuming, blending in with the surrounding buildings, but inside, it's a different world. "
             "The smell of freshly brewed coffee wafts through the air, mingling with the aroma of baked goods coming from the kitchen. "
             "The walls are adorned with local artwork, and soft jazz music plays in the background, creating a cozy and inviting atmosphere.\n"
             "\n"
             "The cafe is a favorite spot for locals and tourists alike, offering a refuge from the hustle and bustle of the city outside. "
             "People come here to relax, to catch up with friends, or simply to enjoy a quiet moment alone. "
             "The baristas are friendly and know many of the regulars by name, adding to the sense of community.\n"
             "\n"
             "As you sit at one of the small tables, sipping your coffee and watching the world go by outside, you can't help but feel at peace. "
             "In this little oasis, time seems to slow down, allowing you to savor the moment and forget about the stresses of daily life. "
             "It's a place where you can pause, reflect, and simply enjoy being in the present.")
        expected_result = 200
        result = total_words(text)
        self.assertEqual(result, expected_result)


    def test_different_words(self):
        text = "this This thIs THIS tHiS has three."
        expected_result = 3
        result = different_words(text)
        self.assertEqual(result, expected_result)

        text = text = "This is a sample sentence."
        expected_result = 5
        result = different_words(text)
        self.assertEqual(result, expected_result)

        text = text = " "
        expected_result = 0
        result = different_words(text)
        self.assertEqual(result, expected_result)

    def test_type_token_ratio(self):
        text =  text = "The quick brown fox jumps over the lazy dog. The fox is quick."
        expected_result = 9/13  # Unique words: 9, Total words: 13
        result = type_token_ratio(text)
        self.assertAlmostEqual(result, expected_result, places=2)

    def test_num_clauses(self):
        text = " "
        expected_result = 0
        result = num_clauses(text)
        self.assertEqual(result, expected_result)

        text = "The team worked late"
        expected_result = 1
        result = num_clauses(text)
        self.assertEqual(result, expected_result)

        text = "He went to the store and bought some groceries"
        expected_result = 1
        result = num_clauses(text)
        self.assertEqual(result, expected_result)
    

if __name__ == '__main__':
    unittest.main()
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAnalysis)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
