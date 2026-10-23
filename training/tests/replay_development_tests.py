"""Synthetic helper tests need no tokenizer, model, network or evaluation inputs."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from replay_development import replay_report,FIELDS


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.case={'id':'synthetic','locale':'en','category':'fixture','expected':{'first_tool':None},
                   'caseSha256':'case-hash','promptSha256':'prompt-hash'}
        self.contract={'suite':'development-validation','promptMode':'chat','caseIds':['synthetic'],
                       'caseCount':1,'caseInputsSha256':'inputs-hash','scoringImplementationSha256':'scorer-hash'}
        self.outcomes={'call':None,'schemaValid':True,'toolChoiceCorrect':True,'exactArgumentsCorrect':True}
        self.report={'completed':True,'hardwareExecution':False,'step':100,'contract':self.contract,
                     'results':[{**self.case,**self.outcomes,'text':'synthetic safe response'}]}

    def test_raw_text_actually_rescored(self):
        scorer=Mock(return_value=self.outcomes)
        out=replay_report(self.report,[self.case],self.contract,scorer)
        scorer.assert_called_once_with(self.case,'synthetic safe response')
        self.assertTrue(out['allOutcomesMatch'])

    def test_each_outcome_compared_separately(self):
        changed={key:not value if isinstance(value,bool) else {'name':'synthetic'} for key,value in self.outcomes.items()}
        out=replay_report(self.report,[self.case],self.contract,lambda *args:changed)
        self.assertEqual(out['mismatchCounts'],{key:1 for key in FIELDS})
        self.assertFalse(out['allOutcomesMatch'])

    def test_mismatched_scorer_or_prompt_fails_before_score(self):
        for target,key in [('contract','scoringImplementationSha256'),('case','promptSha256'),('case','caseSha256')]:
            contract=copy.deepcopy(self.contract);case=copy.deepcopy(self.case)
            (contract if target=='contract' else case)[key]='changed'
            scorer=Mock()
            with self.assertRaisesRegex(ValueError,'mismatch'):
                replay_report(self.report,[case],contract,scorer)
            scorer.assert_not_called()

    def test_all_case_identity_checked_before_any_scoring(self):
        report=copy.deepcopy(self.report);report['results'][0]['expected']={'first_tool':'get_status'}
        scorer=Mock()
        with self.assertRaisesRegex(ValueError,'expected'):replay_report(report,[self.case],self.contract,scorer)
        scorer.assert_not_called()


if __name__=='__main__':unittest.main()
