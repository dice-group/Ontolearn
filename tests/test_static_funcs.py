from ontolearn.utils.static_funcs import compute_f1_score
class TestStaticFuncs:

    def test_f1(self):
        assert compute_f1_score(individuals={"A","B","C"},pos={"A","B","C"},neg={"D"})==1.0
        assert compute_f1_score(individuals={"A","B","C"},pos={"A","B","C"},neg={"D"})==1.0

    # TODO:CD: Add remaining tests