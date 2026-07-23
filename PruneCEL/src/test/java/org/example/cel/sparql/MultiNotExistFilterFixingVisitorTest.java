package org.example.cel.sparql;

import java.util.ArrayList;
import java.util.List;

import org.aksw.jenax.stmt.parser.query.SparqlQueryParserImpl;
import org.apache.jena.query.Query;
import org.apache.jena.vocabulary.OWL;
import org.example.cel.sparql.MultiNotExistFilterFixingVisitor;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(Parameterized.class)
public class MultiNotExistFilterFixingVisitorTest {

    protected String sparqlQuery;
    protected String variableName;
    protected String typeUri;
    protected String expectedQuery;

    public MultiNotExistFilterFixingVisitorTest(String sparqlQuery, String variableName, String typeUri,
            String expectedQuery) {
        super();
        this.sparqlQuery = sparqlQuery;
        this.variableName = variableName;
        this.typeUri = typeUri;
        this.expectedQuery = expectedQuery;
    }

    @Test
    public void test() throws Exception {
        MultiNotExistFilterFixingVisitor visitor = new MultiNotExistFilterFixingVisitor(variableName, typeUri);
        Query query = (new SparqlQueryParserImpl()).apply(sparqlQuery);
        visitor.fixQuery(query);
        String result = query.toString();
        result = result.replaceAll("[\r\n\t ]+", "");
        expectedQuery = expectedQuery.replaceAll("[\r\n\t ]+", "");
        Assert.assertEquals(expectedQuery, result);
    }

    @Parameters
    public static List<Object[]> parameters() {
        List<Object[]> testCases = new ArrayList<>();

        String variable = "?class";
        String typeUri = OWL.Class.getURI();
        String query;
        String expected;

        query = "SELECT ?class (COUNT(DISTINCT ?pos) AS ?tp) (0 AS ?fp) WHERE {"
                + "    VALUES ?pos {<https://github.com/KGQA/QALD-10/blob/main/data/qald_10/qald_10.json#Q274> <https://github.com/KGQA/QALD-10/blob/main/data/qald_10/qald_10.json#Q339>}"
                + " ?class a <http://www.w3.org/2002/07/owl#Class> } GROUP BY ?class";
        expected = "SELECT ?class (COUNT(DISTINCT ?pos) AS ?tp) (0 AS ?fp) WHERE {"
                + "    VALUES ?pos {<https://github.com/KGQA/QALD-10/blob/main/data/qald_10/qald_10.json#Q274> <https://github.com/KGQA/QALD-10/blob/main/data/qald_10/qald_10.json#Q339>}"
                + " ?class a <http://www.w3.org/2002/07/owl#Class> } GROUP BY ?class";
        testCases.add(new Object[] { query, variable, typeUri, expected });

        query = "SELECT ?class (COUNT(DISTINCT ?pos) AS ?tp) (0 AS ?fp) WHERE {"
                + "    VALUES ?pos {<https://github.com/KGQA/QALD-10/blob/main/data/qald_10/qald_10.json#Q274> <https://github.com/KGQA/QALD-10/blob/main/data/qald_10/qald_10.json#Q339>}"
                + " FILTER NOT EXISTS { " + "    ?class a <http://www.w3.org/2002/07/owl#Class> ."
                + "   FILTER NOT EXISTS { ?x2 a ?class . } " + "} ?class a <http://www.w3.org/2002/07/owl#Class> ."
                + "} GROUP BY ?class";
        expected = "SELECT ?class (COUNT(DISTINCT ?pos) AS ?tp) (0 AS ?fp) WHERE {"
                + "VALUES ?pos {<https://github.com/KGQA/QALD-10/blob/main/data/qald_10/qald_10.json#Q274> <https://github.com/KGQA/QALD-10/blob/main/data/qald_10/qald_10.json#Q339>}"
                + "?x2 a ?class . " + "?class a <http://www.w3.org/2002/07/owl#Class> " + "}GROUP BY ?class";
        testCases.add(new Object[] { query, variable, typeUri, expected });

        query = "SELECT ?class (COUNT(DISTINCT ?pos) AS ?tp) (0 AS ?fp) WHERE {"
                + "VALUES ?pos {<https://github.com/KGQA/QALD-10/blob/main/data/qald_10/qald_10.json#Q274> <https://github.com/KGQA/QALD-10/blob/main/data/qald_10/qald_10.json#Q339>}"
                + "FILTER NOT EXISTS {" + "        ?pos <http://w3id.org/dice-research/qa-bench#hasQuery> ?x0 ."
                + "    ?x0 <http://lsq.aksw.org/vocab#triplePath> ?x1 ." + "}" + "FILTER NOT EXISTS {"
                + "        ?pos <http://w3id.org/dice-research/qa-bench#hasQuery> ?x2 ." + "    FILTER NOT EXISTS { "
                + "        ?class a <http://www.w3.org/2002/07/owl#Class> ."
                + "    FILTER NOT EXISTS { ?x2 a ?class . } " + "}"
                + "    ?class a <http://www.w3.org/2002/07/owl#Class> ." + "}"
                + "?class a <http://www.w3.org/2002/07/owl#Class> ." + "}GROUP BY ?class";
        expected = "SELECT ?class (COUNT(DISTINCT ?pos) AS ?tp) (0 AS ?fp) WHERE {"
                + "VALUES ?pos {<https://github.com/KGQA/QALD-10/blob/main/data/qald_10/qald_10.json#Q274> <https://github.com/KGQA/QALD-10/blob/main/data/qald_10/qald_10.json#Q339>}"
                + "FILTER NOT EXISTS {" + "        ?pos <http://w3id.org/dice-research/qa-bench#hasQuery> ?x0 ."
                + "    ?x0 <http://lsq.aksw.org/vocab#triplePath> ?x1 " + "}" + "FILTER NOT EXISTS {"
                + "        ?pos <http://w3id.org/dice-research/qa-bench#hasQuery> ?x2 . ?x2 a ?class ."
                + "    ?class a <http://www.w3.org/2002/07/owl#Class> " + "}"
                + "?class a <http://www.w3.org/2002/07/owl#Class> " + "}GROUP BY ?class";
        testCases.add(new Object[] { query, variable, typeUri, expected });

        return testCases;
    }

}
