package org.example.cel.refine.suggest;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.aksw.jenax.arq.connection.core.QueryExecutionFactory;
import org.aksw.jenax.connection.query.QueryExecutionFactoryDataset;
import org.apache.commons.collections4.SetUtils;
import org.apache.jena.query.Dataset;
import org.apache.jena.query.DatasetFactory;
import org.apache.jena.rdf.model.Model;
import org.apache.jena.rdf.model.Property;
import org.apache.jena.rdf.model.Resource;
import org.apache.jena.rdf.model.ResourceFactory;
import org.apache.jena.vocabulary.OWL2;
import org.apache.jena.vocabulary.RDF;
import org.example.cel.DescriptionLogic;
import org.example.cel.TestHelper;
import org.example.cel.expression.ClassExpression;
import org.example.cel.expression.Junction;
import org.example.cel.expression.NamedClass;
import org.example.cel.expression.ScoredClassExpression;
import org.example.cel.expression.SimpleQuantifiedRole;
import org.example.cel.refine.SuggestorBasedRefinementOperator;
import org.example.cel.refine.suggest.sparql.SparqlBasedSuggestor;
import org.example.cel.score.AccuracyCalculator;
import org.example.cel.score.ScoreCalculator;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(Parameterized.class)
public class SuggestorBasedRefinementOperatorTest {

    protected String logic;
    protected Model model;
    protected Collection<String> positives;
    protected Collection<String> negatives;
    protected ClassExpression input;
    protected Set<ScoredClassExpression> expected;

    public SuggestorBasedRefinementOperatorTest(ClassExpression input, String logic, Model model, String[] positives,
            String[] negatives, ScoredClassExpression[] expected) {
        super();
        this.input = input;
        this.logic = logic;
        this.model = model;
        this.positives = Arrays.asList(positives);
        this.negatives = Arrays.asList(negatives);
        this.expected = new HashSet<ScoredClassExpression>(Arrays.asList(expected));
    }

    @Test
    public void test() throws Exception {
        DescriptionLogic dl = DescriptionLogic.parse(logic);
        Assert.assertNotNull(dl);
        Dataset dataset = DatasetFactory.create(model);
        try (QueryExecutionFactory qef = new QueryExecutionFactoryDataset(dataset);
                SparqlBasedSuggestor suggestor = new SparqlBasedSuggestor(qef, dl)) {
            suggestor.addToClassBlackList(OWL2.NamedIndividual.getURI());
            suggestor.addToPropertyBlackList(RDF.type.getURI());

            SuggestorBasedRefinementOperator operator = new SuggestorBasedRefinementOperator(suggestor, dl,
                    new AccuracyCalculator(positives.size(), negatives.size()), positives, negatives);
            Set<ScoredClassExpression> expressions = operator.refine(input, 0);

            Set<ScoredClassExpression> difference = SetUtils.difference(expected, expressions);
            Assert.assertTrue("Couldn't find expected expressions " + difference.toString() + " in " + expressions,
                    difference.isEmpty());
            difference = SetUtils.difference(expressions, expected);
            Assert.assertTrue("Got unexpected expressions " + difference.toString(), difference.isEmpty());
        }
    }

    @Parameters
    public static List<Object[]> parameters() {
        List<Object[]> testCases = new ArrayList<>();

        Model model;
        Resource pos1 = ResourceFactory.createResource("http://example.org/pos1");
        Resource pos2 = ResourceFactory.createResource("http://example.org/pos2");

        Resource neg1 = ResourceFactory.createResource("http://example.org/neg1");
        Resource neg2 = ResourceFactory.createResource("http://example.org/neg2");
        Resource neg3 = ResourceFactory.createResource("http://example.org/neg3");

        Resource x1 = ResourceFactory.createResource("http://example.org/x1");
        Resource x2 = ResourceFactory.createResource("http://example.org/x2");
        Resource x3 = ResourceFactory.createResource("http://example.org/x3");
        Resource x4 = ResourceFactory.createResource("http://example.org/x4");
        Resource[] individuals = new Resource[] { pos1, pos2, neg1, neg2, neg3, x1, x2, x3, x4 };

        Resource classA = ResourceFactory.createResource("http://example.org/classA");
        Resource classB = ResourceFactory.createResource("http://example.org/classB");
        Resource classC = ResourceFactory.createResource("http://example.org/classC");
        Resource[] classes = new Resource[] { classA, classB, classC };

        Property role1 = ResourceFactory.createProperty("http://example.org/role1");
        Property role2 = ResourceFactory.createProperty("http://example.org/role2");
        Resource[] roles = new Resource[] { role1, role2 };

        model = TestHelper.initModel(classes, roles, individuals);

        model.add(pos1, role1, x1);
        model.add(pos1, RDF.type, classA);

        model.add(pos2, RDF.type, classB);

        model.add(neg1, role1, x2);
        model.add(neg1, RDF.type, classA);

        model.add(neg2, role1, x3);
        model.add(neg2, RDF.type, classB);

        model.add(neg3, role1, x4);
        model.add(neg3, RDF.type, classA);
        model.add(neg3, RDF.type, classB);

        model.add(x1, RDF.type, classA);
        model.add(x2, RDF.type, classA);
        model.add(x3, RDF.type, classB);
        model.add(x4, role2, x1);

        String[] positives = new String[] { pos1.getURI(), pos2.getURI() };
        String[] negatives = new String[] { neg1.getURI(), neg2.getURI(), neg3.getURI() };

        ScoreCalculator calculator = new AccuracyCalculator(positives.length, negatives.length);

        // ⊤
        testCases.add(new Object[] { NamedClass.TOP, "ALC", model, positives, negatives, new ScoredClassExpression[] {
                // classes
                createExpression(new NamedClass(classA.getURI()), 1, 2, false, calculator),
                createExpression(new NamedClass(classB.getURI()), 1, 2, false, calculator),
                // negations of the above
                createExpression(new NamedClass(classA.getURI(), true), 1, 1, false, calculator),
                createExpression(new NamedClass(classB.getURI(), true), 1, 1, false, calculator),
                // roles
                createExpression(new SimpleQuantifiedRole(true, role1.getURI(), false, NamedClass.TOP), 1, 3, true,
                        calculator),
                // negations of the above
                createExpression(new SimpleQuantifiedRole(false, role1.getURI(), false, NamedClass.BOTTOM), 1, 0, true,
                        calculator) } });

        // A
        testCases.add(new Object[] { new NamedClass(classA.getURI()), "ALC", model, positives, negatives,
                new ScoredClassExpression[] {
                        // Junctions with suggested classes
                        createExpression(
                                new Junction(true, new NamedClass(classA.getURI()), new NamedClass(classB.getURI())), 0,
                                1, false, calculator),
                        createExpression(
                                new Junction(false, new NamedClass(classA.getURI()), new NamedClass(classB.getURI())),
                                2, 3, false, calculator),
                        // negations of the above
                        createExpression(new Junction(false, new NamedClass(classA.getURI(), true),
                                new NamedClass(classB.getURI(), true)), 2, 2, false, calculator),
                        createExpression(new Junction(true, new NamedClass(classA.getURI(), true),
                                new NamedClass(classB.getURI(), true)), 0, 0, false, calculator),
                        // Junctions with negated classes
                        createExpression(new Junction(true, new NamedClass(classA.getURI()),
                                new NamedClass(classB.getURI(), true)), 1, 1, false, calculator),
                        // A or not B does not occur because after removing the A cases, there are no
                        // not B elements anymore
                        // createExpression(new Junction(false, new NamedClass(classA.getURI()), new
                        // NamedClass(classB.getURI(), true)), 1, 1, false, calculator),
                        createExpression(new Junction(true, new NamedClass(classA.getURI()),
                                new NamedClass(classC.getURI(), true)), 1, 2, false, calculator),
                        createExpression(new Junction(false, new NamedClass(classA.getURI()),
                                new NamedClass(classC.getURI(), true)), 2, 3, false, calculator),
                        // negations of the above
                        createExpression(new Junction(false, new NamedClass(classA.getURI(), true),
                                new NamedClass(classB.getURI())), 1, 2, false, calculator),
                        createExpression(new Junction(false, new NamedClass(classA.getURI(), true),
                                new NamedClass(classC.getURI())), 1, 1, false, calculator),
                        createExpression(new Junction(true, new NamedClass(classA.getURI(), true),
                                new NamedClass(classC.getURI())), 0, 0, false, calculator),
                        // Junctions with roles
                        createExpression(
                                new Junction(true, new NamedClass(classA.getURI()),
                                        new SimpleQuantifiedRole(true, role1.getURI(), false, NamedClass.TOP)),
                                1, 2, true, calculator),
                        createExpression(
                                new Junction(false, new NamedClass(classA.getURI()),
                                        new SimpleQuantifiedRole(true, role1.getURI(), false, NamedClass.TOP)),
                                1, 3, true, calculator),
                        // negations of the above
                        createExpression(
                                new Junction(false, new NamedClass(classA.getURI(), true),
                                        new SimpleQuantifiedRole(false, role1.getURI(), false, NamedClass.BOTTOM)),
                                1, 1, true, calculator),
                        createExpression(
                                new Junction(true, new NamedClass(classA.getURI(), true),
                                        new SimpleQuantifiedRole(false, role1.getURI(), false, NamedClass.BOTTOM)),
                                1, 0, true, calculator) } });

        return testCases;
    }

    public static ScoredClassExpression createExpression(ClassExpression ce, int positives, int negatives,
            boolean addedEdge, ScoreCalculator calculator) {
        double score = calculator.calculateClassificationScore(positives, negatives);
        return new ScoredClassExpression(ce, score, score, positives, negatives, addedEdge);
    }

}
