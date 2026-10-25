import unittest
from planner import union_ms,simulate,optimize,LAYERS,BASE_PAYLOAD,DELTA,collect,build_table

def uniform(load=10,execute=5,evict=2):
    def metric(v):return dict(intercept_ms=v,q4_32_ms=0.,nodes_32_ms=0.)
    return {str(l):dict(q4_range=[0,32],nodes_range=[1,64],load=metric(load),execute=metric(execute),evict=metric(evict),prep_ms=0.) for l in LAYERS}

class TestPlanner(unittest.TestCase):
    def test_union_not_sum(self):self.assertEqual(union_ms([(0,3),(2,5),(9,10)]),6)
    def test_empty_union(self):self.assertEqual(union_ms([]),0)
    def test_payload(self):
        self.assertEqual(simulate(uniform(),[1]*24,12)['compact_payload_bytes'],BASE_PAYLOAD+24*DELTA)
    def test_no_mutation(self):
        c=[0]*24;simulate(uniform(),c,12);self.assertEqual(c,[0]*24)
    def test_one_step_slot_lifetime(self):
        p=simulate(uniform(),[0]*24,12)['timeline']
        for i,x in enumerate(p):
            self.assertGreaterEqual(x['execute_begin_ms'],x['load_end_ms'])
            self.assertGreaterEqual(x['evict_begin_ms'],x['execute_end_ms'])
            if i:
                self.assertGreaterEqual(x['load_begin_ms'],p[i-1]['execute_begin_ms'])
                self.assertGreaterEqual(x['execute_begin_ms'],p[i-1]['execute_end_ms'])
            if i>1:self.assertGreaterEqual(x['load_begin_ms'],p[i-2]['execute_end_ms'])
    def test_drain_separate(self):
        x=simulate(uniform(),[0]*24,12);self.assertGreater(x['async_evict_drain_ms'],0)
    def test_prefetch(self):
        a=simulate(uniform(),[0]*24,12);b=simulate(uniform(),[0]*24,12,prefetch_ms=10)
        self.assertAlmostEqual(a['predicted_tail_to_sync_ms']-b['predicted_tail_to_sync_ms'],10)
    def test_io_scale(self):
        self.assertGreater(simulate(uniform(),[0]*24,12,2.)['load_union_ms'],simulate(uniform(),[0]*24,12)['load_union_ms'])
    def test_search_observed_ranges(self):
        t=uniform()
        for r in t.values():r['q4_range']=[3,3]
        c,_=optimize(t,12,1.,0.);self.assertEqual(c,[3]*24)
    def test_real_trace_count_and_costs(self):
        rows,sources=collect();self.assertEqual(len(sources),6);self.assertEqual(len(rows),3024)
        self.assertTrue(all(not r['quality_flags'] for r in rows))
        t=build_table(rows);self.assertEqual(len(t),24)
        self.assertTrue(all(r['load_ms']>0 and r['execute_ms']>0 and r['evict_ms']>=0 for r in rows))

if __name__=='__main__':unittest.main()
