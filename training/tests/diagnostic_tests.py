"""CPU-only diagnostic regressions; synthetic outcomes contain no held-out text."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analyze_generation_validation import diagnostics, recompute_metrics, research_candidates, summarize_results
from analyze_run_provenance import analyze_run, digest


def report(step, invalid_no_tool, invalid_total, positive_exact):
    rows=[]
    for i in range(146):
        positive=i>=50
        unwanted=i<5
        exact=(not positive and not unwanted) or (positive and i<50+positive_exact)
        rows.append({'id':str(i),'expected':{'first_tool':'get_status' if positive else None,'arguments':{}},
                     'call':{'name':'get_status','arguments':{}} if positive or unwanted else None,
                     'text':'synthetic', 'schemaValid':not(i<invalid_no_tool or 50<=i<50+invalid_total-invalid_no_tool),
                     'toolChoiceCorrect':exact,'exactArgumentsCorrect':exact,'outputTruncated':False})
    return {'step':step,'contract':{'selection':{'selectionVersion':'v8-safety-first-research-1'}},
            'results':rows,'metrics':{'selection_score':999999}}


class DiagnosticTests(unittest.TestCase):
    def test_historical_v8_rank_recomputed_from_outcomes(self):
        a,b=report(500,3,5,79),report(600,1,3,78)
        self.assertEqual(research_candidates([a,b])['bestSafetyStep'],600)
        delta=recompute_metrics(b)['values']['selection_score']-recompute_metrics(a)['values']['selection_score']
        self.assertGreater(delta,295)
        self.assertFalse(recompute_metrics(b)['storedMetricComparisons']['selection_score']['matches'])

    def test_sequence_positive_and_intervals(self):
        row=report(600,1,3,78)['results'][50]
        row['expected']={'tool_sequence':['get_status'],'arguments':[{}]}
        summary=summarize_results([row])
        self.assertEqual(summary['expectedTool'],1)
        self.assertEqual(summary['unwantedProposals'],0)
        self.assertEqual(summary['positiveToolExact'],1)
        self.assertIsNotNone(summary['positiveToolExactWilson95'])

    def test_action_reversal_and_invented_coords_do_not_change_exact(self):
        row=report(600,1,3,78)['results'][50]
        row['expected']['first_tool']='park_mount'
        row['call']['name']='unpark_mount'
        self.assertIn('critical_action_reversal',diagnostics(row))
        row['expected']['first_tool']=None
        row['category']='missing_coords'
        row['call']={'name':'slew_mount','arguments':{'ra':2,'dec':3}}
        self.assertIn('invented_coordinates',diagnostics(row))
        self.assertTrue(row['exactArgumentsCorrect'])

    def test_provenance_separates_source_and_relocated_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            original=root/'trainer_state.original.json';working=root/'trainer_state.json'
            original.write_text(json.dumps({'global_step':600,'best_model_checkpoint':'old/checkpoint-600'}))
            working.write_text(json.dumps({'global_step':600,'best_model_checkpoint':'new/checkpoint-600'}))
            (root/'training-run.json').write_text(json.dumps({'resume':{'checkpointFilesSha256':{'trainer_state.json':digest(working)}}}))
            out=analyze_run(root,source_state=original,restored_state=working)
            self.assertEqual(out['resumeTrainerState']['sourceOriginal']['status'],'unknown')
            self.assertEqual(out['resumeTrainerState']['restoredWorking']['status'],'match')
            self.assertTrue(out['resumeTrainerState']['comparison']['pathRelocationOnly'])
            self.assertEqual(out['validation']['status'],'unknown')
            self.assertEqual(out['adapterEquality'][0]['status'],'unknown')
            self.assertFalse(out['productionEligible'])

    def test_model_and_input_bytes_independently_hashed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);model=root/'base';model.mkdir()
            file=model/'tokenizer.json';file.write_bytes(b'synthetic')
            (root/'training-run.json').write_text(json.dumps({'base':{'files':{'tokenizer.json':digest(file),'model.safetensors':'absent'}}}))
            out=analyze_run(root,base_dir=model)
            byname={Path(r['path']).name:r['status'] for r in out['verifiedInputFiles']}
            self.assertEqual(byname,{'tokenizer.json':'match','model.safetensors':'unknown'})


if __name__=='__main__':unittest.main()
