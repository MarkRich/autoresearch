import unittest

from autoresearch_utils import attention_window_covers_sequence


class ModelUtilityTests(unittest.TestCase):
    def test_full_training_window_also_covers_short_generation_context(self):
        self.assertTrue(attention_window_covers_sequence((2048, 0), 2048))
        self.assertTrue(attention_window_covers_sequence((2048, 0), 17))

    def test_short_window_does_not_cover_longer_sequence(self):
        self.assertFalse(attention_window_covers_sequence((1024, 0), 2048))


if __name__ == "__main__":
    unittest.main()
