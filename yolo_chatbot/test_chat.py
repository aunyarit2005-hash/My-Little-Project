import unittest
from chat import answer, next_steps, select_intent

class ChatTests(unittest.TestCase):
    def data(self, *labels):
        return {'detections': [{'label': l, 'confidence': .9, 'bbox': [0,0,10,10]} for l in labels]}
    def test_empty_is_not_normal(self):
        self.assertIn('ไม่ได้หมายความว่าภาพปกติ', answer('summary', self.data()))
    def test_mixed_keeps_malignant(self):
        self.assertIn('malignant', next_steps(self.data('benign','malignant')))
    def test_benign_no_clearance(self):
        self.assertIn('ยังยืนยันว่าไม่มีโรคไม่ได้', next_steps(self.data('benign')))
    def test_confidence_not_disease_probability(self):
        self.assertIn('ไม่ใช่เปอร์เซ็นต์โอกาส', answer('confidence', self.data('malignant')))
    def test_treatment_boundary(self):
        self.assertEqual(select_intent('รักษายังไง')[0], 'limits')
    def test_summary_actual_count(self):
        self.assertIn('2 กรอบ', answer('summary', self.data('benign','benign')))
    def test_followup(self):
        self.assertEqual(select_intent('ควรทำอย่างไรต่อไป')[0], 'next')
    def test_unknown(self):
        self.assertEqual(select_intent('tell me a joke')[0], 'limits')
if __name__ == '__main__':
    unittest.main()
