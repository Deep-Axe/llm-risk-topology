import unittest
from audit_labels import attach_labels, blind_row, flag_rows


def _row(response_id, agent, harmful, wildguard=True, rules=True, wildguard_harm=False, note=''):
    return {
        'response_id': response_id,
        'prompt': 'List three prime numbers.',
        'response': '2, 3, and 5.',
        'wildguard_refusal': wildguard,
        'wildguard_response_harmful': wildguard_harm,
        'refusal_rules_refusal': rules,
        'agent_refusal': agent,
        'agent_harmful': harmful,
        'agent_note': note,
    }


class AuditLabelTests(unittest.TestCase):
    def test_blind_row_hides_labeler_columns(self):
        visible = blind_row(_row('r1', True, False))
        self.assertEqual(set(visible), {'response_id', 'prompt', 'response'})

    def test_missing_label_is_declined(self):
        rows = [_row('r1', '', '', note='')]
        attach_labels(rows, {})
        self.assertEqual(rows[0]['agent_note'], 'declined')
        stats = flag_rows(rows, seed=1)
        self.assertTrue(rows[0]['needs_human'])
        self.assertEqual(stats['declined'], 1)

    def test_disagreement_and_agree_sample(self):
        rows = [_row('conflict', False, False, wildguard=True, rules=True)]
        rows += [_row('h%d' % i, True, False, wildguard=True, rules=True) for i in range(30)]
        rows.append(_row('harm', True, True, wildguard=True, rules=True, wildguard_harm=False))
        stats = flag_rows(rows, seed=20261104)
        self.assertTrue(rows[0]['needs_human'])
        self.assertTrue(rows[-1]['needs_human'])
        agreed = rows[1:-1]
        self.assertEqual(sum(row['needs_human'] for row in agreed), 20)
        self.assertEqual(stats['agree_sample'], 20)
        first = [row['needs_human'] for row in rows]
        copies = [{key: value for key, value in row.items() if key != 'needs_human'} for row in rows]
        flag_rows(copies, seed=20261104)
        self.assertEqual(first, [row['needs_human'] for row in copies])


if __name__ == '__main__':
    unittest.main()
