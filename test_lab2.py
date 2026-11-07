import math
import tempfile
import unittest
from pathlib import Path

from correlation_tasks import analyze_brain_correlations
from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput
from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary
from pandas_tasks import analyze_titanic


class TitanicTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "titanic.csv"
        self.path.write_text(
            "Pclass,Age,Survived,Fare,Note\n"
            "1,40,1,100,\n"
            "1,,0,100,ok\n"
            "2,30,1,50,ok\n"
            "2,50,1,75,\n"
            "3,20,0,10,ok\n"
            "3,,1,25,ok\n",
            encoding="utf-8",
        )
        self.result = analyze_titanic(TitanicInput(str(self.path)))

    def test_row_count_and_missing_values(self):
        self.assertIsInstance(self.result, TitanicSummary)
        self.assertEqual(self.result.row_count, 6)
        self.assertEqual(
            self.result.missing_by_column,
            {"Pclass": 0, "Age": 2, "Survived": 0, "Fare": 0, "Note": 2},
        )

    def test_age_boundary_and_means_without_filling_missing_ages(self):
        self.assertEqual(self.result.adults_over_30_count, 2)
        self.assertEqual(self.result.mean_age_by_pclass, {1: 40.0, 2: 40.0, 3: 20.0})
        for key in self.result.mean_age_by_pclass:
            self.assertIs(type(key), int)

    def test_survival_proportions_by_class(self):
        self.assertEqual(self.result.survival_rate_by_pclass, {1: 0.5, 2: 1.0, 3: 0.5})
        for key in self.result.survival_rate_by_pclass:
            self.assertIs(type(key), int)

    def test_top_fares_keep_duplicates_and_skip_missing_values(self):
        self.assertEqual(self.result.highest_fares, [100.0, 100.0, 75.0, 50.0, 25.0])
        self.path.write_text(
            "Pclass,Age,Survived,Fare\n1,40,1,8\n1,20,0,\n2,30,1,12\n",
            encoding="utf-8",
        )
        result = analyze_titanic(TitanicInput(str(self.path)))
        self.assertEqual(result.highest_fares, [12.0, 8.0])


class BrainCorrelationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "brainsize.txt"
        path.write_text(
            "Gender\tFSIQ\tVIQ\tPIQ\tWeight\tHeight\tMRI_Count\n"
            "Female\t1\t4\t4\t10\t60\t1\n"
            "Female\t3\t2\t3\tNA\t60\t2\n"
            "Female\t2\t3\t2\t30\t60\t3\n"
            "Female\t4\t1\t1\t20\t60\t4\n"
            "Male\t4\t1\t2\t3\t60\t11\n"
            "Male\t2\t4\t1\t1\t62\t12\n"
            "Male\t3\t2\t4\t4\t61\t13\n"
            "Male\t1\t3\t3\t2\t63\t14\n",
            encoding="utf-8",
        )
        self.result = analyze_brain_correlations(BrainDataInput(str(path)))

    def test_groups_are_counted_and_analyzed_separately(self):
        self.assertIsInstance(self.result, BrainCorrelationSummary)
        self.assertEqual((self.result.men_count, self.result.women_count), (4, 4))
        features = {"FSIQ", "VIQ", "PIQ", "Weight", "Height"}
        self.assertEqual(set(self.result.women_mri_correlation), features)
        self.assertEqual(set(self.result.men_mri_correlation), features)
        self.assertAlmostEqual(self.result.women_mri_correlation["FSIQ"], 0.8)
        self.assertAlmostEqual(self.result.men_mri_correlation["FSIQ"], -0.8)
        self.assertAlmostEqual(self.result.men_mri_correlation["PIQ"], 0.6)

    def test_missing_weight_only_excludes_that_pair(self):
        women = self.result.women_mri_correlation
        self.assertAlmostEqual(women["Weight"], math.sqrt(3 / 7))
        self.assertAlmostEqual(women["VIQ"], -0.8)
        self.assertEqual(self.result.women_count, 4)

    def test_strongest_uses_absolute_value_and_ignores_constant_feature(self):
        self.assertTrue(math.isnan(self.result.women_mri_correlation["Height"]))
        self.assertAlmostEqual(self.result.women_mri_correlation["PIQ"], -1.0)
        self.assertEqual(self.result.strongest_mri_feature, "PIQ")


if __name__ == "__main__":
    unittest.main()
