from datetime import date
import unittest

from owlready2.prop import DataProperty
from ontolearn.value_splitter import BinningValueSplitter
from ontolearn.base.fast_instance_checker import OWLReasoner_FastInstanceChecker
from owlapy.owl_literal import OWLDataProperty, OWLLiteral
from owlapy.iri import IRI
from ontolearn.base import OWLOntologyManager_Owlready2, OWLReasoner_Owlready2


class BinningValueSplitter_Test(unittest.TestCase):

    def test_binning_splitter_numeric(self):
        namespace_ = "http://example.com/father#"
        mgr = OWLOntologyManager_Owlready2()
        onto = mgr.load_ontology(IRI.create("file://KGs/Family/father.owl"))

        with onto._onto:
            class test_int(DataProperty):
                range = [int]

            class test_float(DataProperty):
                range = [float]

        values_int = [3, -45, -36, -32, 20, -85, -26, 47, -66, 71, 25, 59, 69, -62, 73, -16, 40, -18, -67, -37]
        values_float = [1.2, 3.4, -5.6, 9.5, 20.1]
        onto._onto.markus.test_int = values_int
        onto._onto.markus.test_float = values_float

        base_reasoner = OWLReasoner_Owlready2(onto)
        reasoner = OWLReasoner_FastInstanceChecker(onto, base_reasoner=base_reasoner)

        test_int_dp = OWLDataProperty(IRI(namespace_, 'test_int'))
        test_float_dp = OWLDataProperty(IRI(namespace_, 'test_float'))

        splitter = BinningValueSplitter(max_nr_splits=10)
        splits = splitter.compute_splits_properties(reasoner, [test_int_dp, test_float_dp])
        results = {test_int_dp: [OWLLiteral(-85), OWLLiteral(-64), OWLLiteral(-41), OWLLiteral(-34),
                                 OWLLiteral(-22), OWLLiteral(-7), OWLLiteral(22), OWLLiteral(43),
                                 OWLLiteral(64), OWLLiteral(73)],
                   test_float_dp: [OWLLiteral(-5.6), OWLLiteral(-2.2), OWLLiteral(2.3), OWLLiteral(6.45),
                                   OWLLiteral(14.8), OWLLiteral(20.1)]}
        self.assertEqual(splits, results)

    def test_binning_splitter_time(self):
        namespace_ = "http://example.com/father#"
        mgr = OWLOntologyManager_Owlready2()
        onto = mgr.load_ontology(IRI.create("file://KGs/Family/father.owl"))

        with onto._onto:
            class test_time(DataProperty):
                range = [date]

        values_datetime = [date(2000, 10, 21), date(2003, 2, 10), date(1998, 7, 30), date(1990, 8, 3),
                           date(2006, 6, 6), date(2008, 5, 6), date(2012, 5, 3), date(2010, 1, 1),
                           date(1985, 4, 6), date(1999, 9, 9)]
        onto._onto.markus.test_time = values_datetime

        base_reasoner = OWLReasoner_Owlready2(onto)
        reasoner = OWLReasoner_FastInstanceChecker(onto, base_reasoner=base_reasoner)

        test_time_dp = OWLDataProperty(IRI(namespace_, 'test_time'))

        splitter = BinningValueSplitter(max_nr_splits=4)
        splits = splitter.compute_splits_properties(reasoner, [test_time_dp])
        results = {test_time_dp: [OWLLiteral(date(1985, 4, 6)), OWLLiteral(date(1998, 7, 30)),
                                  OWLLiteral(date(2003, 2, 10)), OWLLiteral(date(2012, 5, 3))]}
        self.assertEqual(splits, results)


if __name__ == '__main__':
    unittest.main()
