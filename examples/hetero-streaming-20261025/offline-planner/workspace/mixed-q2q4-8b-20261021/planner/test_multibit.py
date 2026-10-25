import unittest
from multibit_planner import (simulate, optimize, layer_cost, build_profile,
                              CalibrationRequired, WEIGHTS_LAYER, LAYERS)
from multibit_traces import extract
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def profile(route='direct', bits=(2,3)):
    def anchor(b):
        return dict(nodes_range=[8,32],points=[dict(nodes=n,read=10+(b-2)*5,copy=3 if route=='staged' else 0,
                    execute=5+n/32,evict=2,prep=.1) for n in (8,32)])
    return dict(schema=2,route=route,deployment='synthetic',bits=list(bits),
                layers={str(l):{str(b):anchor(b) for b in bits} for l in LAYERS})


class MultibitTests(unittest.TestCase):
    def test_payload_interpolation(self):
        p=profile()
        a=simulate(p,[0]*24,8);b=simulate(p,[32]*24,8)
        self.assertEqual(a['payload_bytes'],1635778560)
        self.assertEqual(b['payload_bytes'],2290089984)
        counts=[16]*24
        self.assertEqual(simulate(p,counts,8)['payload_bytes'],(a['payload_bytes']+b['payload_bytes'])//2)
        self.assertEqual(counts,[16]*24)

    def test_equal_budget378(self):
        counts=[16]*23+[10]
        self.assertEqual(sum(counts),378)
        self.assertEqual(simulate(profile(),counts,8)['payload_bytes']-1635778560,322043904)

    def test_missing_q3_copy_rejected(self):
        with self.assertRaises(CalibrationRequired):
            simulate(profile('staged',(4,)),[0]*24,8)

    def test_node_range(self):
        with self.assertRaises(CalibrationRequired):simulate(profile(),[0]*24,7)
        a=simulate(profile(),[0]*24,7,allow_extrapolation=True)
        b=simulate(profile(),[0]*24,8)
        self.assertEqual(a['tail_to_sync_ms'],b['tail_to_sync_ms'])

    def test_integer_bounds(self):
        with self.assertRaises(ValueError):simulate(profile(),[33]*24,8)
        with self.assertRaises(ValueError):simulate(profile(),[.5]*24,8)

    def test_copy_completion_and_slot_lifetime(self):
        p=simulate(profile('staged'),[16]*24,8,chunk_bytes=8<<20,staging_buffers=2)
        previous_copy=-1e10
        host_free={}
        for i,x in enumerate(p['timeline']):
            self.assertGreaterEqual(x['evict_begin_ms'],x['execute_end_ms'])
            self.assertGreaterEqual(x['execute_begin_ms'],x['chunks'][-1]['copy_end_ms'])
            for c in x['chunks']:
                self.assertGreaterEqual(c['copy_begin_ms'],c['read_end_ms'])
                self.assertGreaterEqual(c['copy_begin_ms'],previous_copy)
                self.assertGreaterEqual(c['read_begin_ms'],host_free.get(c['host_buffer'],-1e10))
                host_free[c['host_buffer']]=c['copy_end_ms'];previous_copy=c['copy_end_ms']
                if i>=2:self.assertGreaterEqual(c['copy_begin_ms'],p['timeline'][i-2]['execute_end_ms'])
                if i:self.assertGreaterEqual(c['read_begin_ms'],p['timeline'][i-1]['execute_begin_ms'])
        self.assertGreater(p['read_copy_overlap_ms'],0)

    def test_direct_lifetime(self):
        p=simulate(profile(),[0]*24,8)
        for i,x in enumerate(p['timeline']):
            if i>=2:self.assertGreaterEqual(x['chunks'][0]['read_begin_ms'],p['timeline'][i-2]['execute_end_ms'])
        self.assertEqual(p['copy_union_ms'],0)
        self.assertEqual(p['staging_reserved_bytes'],0)

    def test_union_not_sum(self):
        p=simulate(profile('staged'),[0]*24,8)
        self.assertLess(p['downstream_union_ms'],p['execute_union_ms']+p['copy_union_ms']+p['evict_union_ms'])
        self.assertGreater(p['evict_drain_ms'],0)

    def test_only_l8_prefetch(self):
        p=simulate(profile(),[0]*24,8,prefetch_ms=10)
        self.assertEqual(p['timeline'][0]['chunks'][0]['read_begin_ms'],-10)
        self.assertGreaterEqual(p['timeline'][1]['chunks'][0]['read_begin_ms'],p['timeline'][0]['execute_begin_ms'])

    def test_direct_copy_cost_rejected(self):
        p=profile()
        p['layers']['8']['2']['points'][0]['copy']=2
        with self.assertRaises(ValueError):simulate(p,[0]*24,8)

    def test_no_cross_route_fitting(self):
        with self.assertRaises(ValueError):build_profile([dict(route='staged',deployment='a')],[],'direct','a')

    def test_search_runs(self):
        counts,pred=optimize(profile(),8,2,3)
        self.assertEqual(len(counts),24)
        self.assertTrue(all(isinstance(k,int) and 0<=k<=32 for k in counts))

    def test_balance_band_preference(self):
        p=profile()
        for anchors in p['layers'].values():
            for anchor in anchors.values():
                for point in anchor['points']:
                    point['execute']=12
        precise,a=optimize(p,8,2,3,within_balance='precision')
        fast,b=optimize(p,8,2,3,within_balance='latency')
        self.assertGreaterEqual(sum(precise),sum(fast))
        self.assertLessEqual(a['balance_penalty'],1e-10)
        self.assertLessEqual(b['balance_penalty'],1e-10)

    def test_staging_reservation(self):
        x=simulate(profile('staged'),[0]*24,8)
        self.assertEqual(x['staging_reserved_bytes'],14*(8<<20))

    def test_actual_direct_traces(self):
        cases=[('q2-llama3-8b-20261021/phone-results/logs/cpu_n128.log',2,432),
               ('q3g32-8b-20261022/phone-results/llama-exp-q3g32-8b-20261022/logs/q3g32_gain_n128.log',3,528)]
        for path,bits,count in cases:
            rows,source=extract(ROOT/path,'direct',bits,'test-direct')
            self.assertEqual(len(rows),count)
            self.assertTrue(all(r['copy_ms']==0 and r['payload_bytes']==WEIGHTS_LAYER*(bits+.5)/8 for r in rows))

    def test_actual_staged_trace(self):
        path=ROOT/'q4g128-staged-t8-20261023/phone-results/llama-exp-q4g128-staged-t8-20261023/logs/q4g32_n128.log'
        rows,source=extract(path,'staged',4,'test-staged')
        self.assertEqual(len(rows),648)
        self.assertTrue(all(r['copy_ms']>0 and r['source_evict_policy']=='after-read' for r in rows))
        with self.assertRaises(ValueError):extract(path,'staged',3,'test-staged')

    def test_new_q23_staged_postcompute(self):
        for bits,n,count in ((2,16,96),(2,128,432),(3,16,120),(3,128,528)):
            path=ROOT/f'staged-q23-calibration-20261024/phone-results/logs/q{bits}_staged_n{n}.log'
            rows,source=extract(path,'staged',bits,'test-q23-staged')
            self.assertEqual(len(rows),count)
            self.assertTrue(all(r['source_evict_policy']=='postcompute' and r['copy_ms']>0 for r in rows))
            self.assertIn('deployment_manifest_sha256',source)


if __name__=='__main__':unittest.main()
