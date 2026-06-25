import unittest

from scripts.import_multidoc2dial import (
    build_document_records,
    build_question_records,
)


class ImportMultiDoc2DialTests(unittest.TestCase):
    def setUp(self):
        self.documents = {
            "doc-1": {
                "doc_id": "doc-1",
                "title": "Change address",
                "domain": "dmv",
                "doc_text": "Report a new address within ten days.",
                "spans": {
                    "7": {
                        "text_sp": "within ten days",
                        "title": "Deadline",
                    }
                },
            }
        }

    def test_builds_document_record(self):
        records = list(build_document_records(self.documents, "dmv"))

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["document_id"], "doc-1")
        self.assertEqual(records[0]["domain"], "dmv")
        self.assertIn("7", records[0]["spans"])

    def test_builds_question_with_answer_and_gold_evidence(self):
        dialogues = [
            {
                "dial_id": "dialogue-1",
                "turns": [
                    {
                        "turn_id": 1,
                        "role": "user",
                        "utterance": "When must I report my address?",
                        "references": [],
                    },
                    {
                        "turn_id": 2,
                        "role": "agent",
                        "utterance": "Within ten days.",
                        "references": [
                            {
                                "doc_id": "doc-1",
                                "id_sp": "7",
                                "label": "solution",
                            }
                        ],
                    },
                ],
            }
        ]

        records = list(
            build_question_records(dialogues, self.documents, "dmv", "validation")
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["answer"], "Within ten days.")
        self.assertEqual(records[0]["gold_document_ids"], ["doc-1"])
        self.assertEqual(
            records[0]["gold_evidence"][0]["text"], "within ten days"
        )


if __name__ == "__main__":
    unittest.main()
