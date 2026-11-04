"""Behavioral regressions for tutorial boundaries, model reuse and UI wiring.

Run: .venv/Scripts/python.exe -m unittest src.test_tutorial_integration -v
"""
import json
import unittest
from pathlib import Path

import numpy as np

import app
from src.tutorial_services import classify_text, extract_entities, search_notes, image_identity, text_models
from src.workspace_ui import TUTORIALS


class TutorialIntegration(unittest.TestCase):
    def test_t5_registration_geometry(self):
        from src.tutorial_services import registration_analysis
        image, *_ = app.load_example(2)
        result = registration_analysis(image)
        self.assertEqual(result['image'].shape, (224, 448, 3))
        self.assertGreaterEqual(result['inliers'], 4)
        self.assertLessEqual(result['inliers'], result['matches'])
        self.assertEqual(result['homography'].shape, (3, 3))
        self.assertLess(result['reprojection_rmse'], 5)
        blank = registration_analysis(np.zeros((224, 224, 3), dtype=np.uint8))
        self.assertIsNone(blank['homography'])
        self.assertEqual(blank['inliers'], 0)

    def test_fifteen_distinct_workspaces(self):
        self.assertEqual(len(TUTORIALS), 15)
        tabs = [c for c in app.demo.config['components'] if c['type'] == 'tabitem']
        self.assertEqual(len(tabs), 17)

    def test_text_classifier_is_independent(self):
        mild = classify_text('Mild nonproliferative diabetic retinopathy. Scattered microaneurysms. Follow-up in 6 months.')
        severe = classify_text('Severe nonproliferative diabetic retinopathy. Four quadrant hemorrhages, venous beading and IRMA. Urgent specialist referral.')
        self.assertIn('Grade 1', mild)
        self.assertIn('Grade 3', severe)
        self.assertIs(text_models(), text_models())

    def test_entity_spans_match_input(self):
        text = 'OS: Hard exudates. Refer to ophthalmology.'
        entities = extract_entities(text)
        self.assertEqual(entities[0]['Value'], 'OS')
        self.assertTrue(any(e['Category'] == 'ACTION' for e in entities))
        for entity in entities:
            self.assertEqual(text[entity['Start']:entity['End']], entity['Value'])
        self.assertEqual(extract_entities(''), [])

    def test_real_ranking(self):
        for method in ['BM25', 'TF-IDF']:
            results = search_notes('neovascularization emergency vitreoretinal', method)
            self.assertGreater(len(results), 0)
            self.assertTrue(results.score.is_monotonic_decreasing)
            self.assertTrue((results.diagnosis == 4).any())
        self.assertTrue(search_notes('').empty)

    def test_staffing_matches_t7(self):
        value = app.update_staffing_calculation(20000)
        self.assertIn('75', value)
        self.assertIn('**3**', value)
        self.assertNotIn('18 comprehensive', value)

    def test_empty_input(self):
        self.assertIsNone(app.run_screening_analysis(None, *([None] * 9)))
        self.assertEqual(len(app.dispatch_analysis_results(None)), 25)

    def test_screening_cases_and_exports(self):
        for grade in [0, 2, 4]:
            with self.subTest(sample_grade=grade):
                image, *values = app.load_example(grade)
                state = app.run_screening_analysis(image, *values, image_is_preprocessed=True)
                self.assertEqual(len(app.dispatch_analysis_results(state)), 25)
                self.assertEqual(state['age'], values[0])
                self.assertTrue(state['duplicate_match'][2])
                self.assertTrue(np.array_equal(image, state['normalized_rgb']))
                self.assertAlmostEqual(float(state['probs'].sum()), 1, places=5)
                self.assertGreater(state['pred_bmi'], 10)
                self.assertEqual(state['rule'], app.REFERRAL_RULES[state['pred_class']])
                with open(state['json_path'], encoding='utf-8') as file:
                    record = json.load(file)
                self.assertEqual(record['predicted_grade'], state['pred_class'])
                self.assertTrue(Path(state['pdf_path']).read_bytes().startswith(b'%PDF'))

    def test_raw_normalization_and_optional_behavior(self):
        image, *_ = app.load_example(0)
        state = app.run_screening_analysis(image, *([None] * 9))
        self.assertFalse(state['has_lifestyle'])
        self.assertIsNone(state['patient_record']['multimodal_score'])
        self.assertFalse(np.array_equal(image, state['normalized_rgb']))
        self.assertEqual(state['normalized_rgb'].shape, (224, 224, 3))
        import cv2
        from src.t2_preprocess_ehr import ben_graham_normalize
        expected = cv2.cvtColor(cv2.resize(ben_graham_normalize(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 512), (224,224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        self.assertTrue(np.array_equal(expected, state['normalized_rgb']))

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            app.run_screening_analysis(np.zeros((4, 4, 3), dtype=np.uint8), *([None] * 9))
        image, *values = app.load_example(0)
        values[3] = 0
        with self.assertRaises(ValueError):
            app.run_screening_analysis(image, *values)


if __name__ == '__main__':
    unittest.main()
