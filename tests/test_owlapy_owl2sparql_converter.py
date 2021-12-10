import unittest

from owlapy.model import OWLObjectProperty, IRI, OWLObjectSomeValuesFrom, OWLObjectMaxCardinality, OWLThing, \
    OWLObjectMinCardinality, OWLObjectUnionOf, OWLObjectIntersectionOf
from owlapy.owl2sparql.converter import Owl2SparqlConverter


class Test_Owl2SparqlConverter(unittest.TestCase):
    def test_as_query(self):
        prop_s = OWLObjectProperty(IRI.create("http://dl-learner.org/carcinogenesis#hasBond"))
        ce = OWLObjectSomeValuesFrom(
            prop_s,
            OWLObjectIntersectionOf((
                OWLObjectMaxCardinality(
                    4,
                    OWLObjectProperty(IRI.create("http://dl-learner.org/carcinogenesis#hasAtom")),
                    OWLThing
                ),
                OWLObjectMinCardinality(
                    1,
                    OWLObjectProperty(IRI.create("http://dl-learner.org/carcinogenesis#hasAtom")),
                    OWLThing
                )
            ))
        )
        cnv = Owl2SparqlConverter()
        root_var = "?x"
        query = cnv.as_query(root_var, ce, False)
        # print(query)
        query_t = """SELECT
 DISTINCT ?x WHERE { 
?x <http://dl-learner.org/carcinogenesis#hasBond> ?s_1 . 
?s_1 <http://dl-learner.org/carcinogenesis#hasAtom> ?s_2 . 
?s_1 <http://dl-learner.org/carcinogenesis#hasAtom> ?s_3 . 
 }
GROUP BY ?x
 HAVING ( 
COUNT ( ?s_2 ) <= 4 && COUNT ( ?s_3 ) >= 1
 )"""
        self.assertEqual(query, query_t)  # add assertion here


if __name__ == '__main__':
    unittest.main()
