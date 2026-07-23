package org.example.cel.refine.suggest.sparql;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;

import org.aksw.jenax.arq.connection.core.QueryExecutionFactory;
import org.aksw.jenax.connection.query.QueryExecutionFactoryDataset;
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
import org.example.cel.expression.SimpleQuantifiedRole;
import org.example.cel.refine.suggest.ScoredIRI;
import org.example.cel.refine.suggest.Suggestor;
import org.example.cel.refine.suggest.sparql.SparqlBasedSuggestor;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(Parameterized.class)
public class SparqlBasedSuggestorTest implements Comparator<ScoredIRI> {

    @Parameters
    public static List<Object[]> parameters() {
        List<Object[]> testCases = new ArrayList<>();

        Model model;
        ClassExpression input;
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

        // ⌖⊔∀role1.⊥
        input = new Junction(false, Suggestor.CONTEXT_POSITION_MARKER,
                new SimpleQuantifiedRole(false, role1.getURI(), false, NamedClass.BOTTOM));
        model = TestHelper.initModel(classes, roles, individuals);
        model.add(pos1, role1, x1);
        model.add(pos1, RDF.type, classA);
        model.add(pos2, RDF.type, classC);
        model.add(neg1, role1, x2);
        model.add(neg1, RDF.type, classA);
        model.add(neg2, role1, x3);
        model.add(neg2, RDF.type, classB);
        model.add(neg3, role1, x4);
        testCases.add(new Object[] { input, "ALC", model, new String[] { pos1.getURI(), pos2.getURI() },
                new String[] { neg1.getURI(), neg2.getURI(), neg3.getURI() },
                new ScoredIRI[] { new ScoredIRI(classA.getURI(), 2, 1), new ScoredIRI(classB.getURI(), 1, 1) },
                new ScoredIRI[] { new ScoredIRI(role1.getURI(), 2, 3) } });

        // ∃role1.(⌖⊓∀role2.⊥)
        input = new SimpleQuantifiedRole(true, role1.getURI(), false,
                new Junction(true, Suggestor.CONTEXT_POSITION_MARKER,
                        new SimpleQuantifiedRole(false, role2.getURI(), false, NamedClass.BOTTOM)));
        model = TestHelper.initModel(classes, roles, individuals);
        model.add(pos1, role1, x1);
        model.add(pos2, RDF.type, classC);
        model.add(neg1, role1, x2);
        model.add(neg2, role1, x3);
        model.add(neg3, role1, x4);
        model.add(x1, RDF.type, classA);
        model.add(x2, RDF.type, classA);
        model.add(x3, RDF.type, classB);
        model.add(x3, role1, x4);
        model.add(x4, role2, x1);
        testCases.add(new Object[] { input, "ALC", model, new String[] { pos1.getURI(), pos2.getURI() },
                new String[] { neg1.getURI(), neg2.getURI(), neg3.getURI() },
                new ScoredIRI[] { new ScoredIRI(classA.getURI(), 1, 1), new ScoredIRI(classB.getURI(), 0, 1) },
                new ScoredIRI[] { new ScoredIRI(role1.getURI(), 0, 1) } });

        // ∀role1.(⌖⊓∀role2.⊥)
        input = new SimpleQuantifiedRole(false, role1.getURI(), false,
                new Junction(true, Suggestor.CONTEXT_POSITION_MARKER,
                        new SimpleQuantifiedRole(false, role2.getURI(), false, NamedClass.BOTTOM)));
        model = TestHelper.initModel(classes, roles, individuals);
        model.add(pos1, role1, x1);
        model.add(pos2, RDF.type, classC);
        model.add(neg1, role1, x2);
        model.add(neg2, role1, x3);
        model.add(neg3, role1, x4);
        model.add(x1, RDF.type, classA);
        model.add(x2, RDF.type, classA);
        model.add(x3, RDF.type, classB);
        model.add(x3, role1, x4);
        model.add(x4, role2, x1);
        testCases.add(new Object[] { input, "ALC", model, new String[] { pos1.getURI(), pos2.getURI() },
                new String[] { neg1.getURI(), neg2.getURI(), neg3.getURI() },
                new ScoredIRI[] { new ScoredIRI(classA.getURI(), 2, 1), new ScoredIRI(classB.getURI(), 1, 1),
                        new ScoredIRI(classC.getURI(), 1, 0) },
                new ScoredIRI[] { new ScoredIRI(role1.getURI(), 1, 1), new ScoredIRI(role2.getURI(), 1, 0) } });

        // ∀role1.(⌖⊔∀role2.⊥)
        input = new SimpleQuantifiedRole(false, role1.getURI(), false,
                new Junction(false, Suggestor.CONTEXT_POSITION_MARKER,
                        new SimpleQuantifiedRole(false, role2.getURI(), false, NamedClass.BOTTOM)));
        model = TestHelper.initModel(classes, roles, individuals);
        model.add(pos1, role1, x1);
        model.add(pos2, RDF.type, classC);
        model.add(neg1, role1, x2);
        model.add(neg2, role1, x3);
        model.add(neg3, role1, x4);
        model.add(x1, RDF.type, classA);
        model.add(x2, RDF.type, classA);
        model.add(x3, RDF.type, classB);
        model.add(x3, role1, x4);
        model.add(x4, role2, x1);
        model.add(x4, role1, x3);
        model.add(x4, RDF.type, classC);
        testCases.add(new Object[] { input, "ALC", model, new String[] { pos1.getURI(), pos2.getURI() },
                new String[] { neg1.getURI(), neg2.getURI(), neg3.getURI() },
                new ScoredIRI[] { new ScoredIRI(classA.getURI(), 2, 2), new ScoredIRI(classB.getURI(), 2, 2),
                        new ScoredIRI(classC.getURI(), 2, 3) },
                new ScoredIRI[] { new ScoredIRI(role1.getURI(), 2, 3), new ScoredIRI(role2.getURI(), 2, 3) } });

        // ∀role1.⊥⊔∀role2.⌖
        input = new Junction(false, new SimpleQuantifiedRole(false, role1.getURI(), false, NamedClass.BOTTOM),
                new SimpleQuantifiedRole(false, role2.getURI(), false, Suggestor.CONTEXT_POSITION_MARKER));
        model = TestHelper.initModel(classes, roles, individuals);
        model.add(pos1, role1, x1);
        model.add(pos1, role2, x1);
        model.add(pos2, RDF.type, classC); // ∀role1.⊥
        model.add(pos2, role2, x3);
        model.add(neg1, role1, x2);
        model.add(neg1, role2, x3);
        model.add(neg2, role1, x3);
        model.add(neg3, role2, x4); // ∀role1.⊥
        model.add(x1, RDF.type, classA);
        model.add(x2, RDF.type, classA);
        model.add(x3, RDF.type, classB);
        model.add(x3, role1, x4);
        model.add(x4, role2, x1);
        model.add(x4, role1, x3);
        model.add(x4, RDF.type, classC);
        testCases.add(new Object[] { input, "ALC", model, new String[] { pos1.getURI(), pos2.getURI() },
                new String[] { neg1.getURI(), neg2.getURI(), neg3.getURI() },
                new ScoredIRI[] { new ScoredIRI(classA.getURI(), 2, 2), new ScoredIRI(classB.getURI(), 1, 3),
                        new ScoredIRI(classC.getURI(), 1, 2) },
                new ScoredIRI[] { new ScoredIRI(role1.getURI(), 1, 3), new ScoredIRI(role2.getURI(), 1, 2) } });

        return testCases;
    }

    protected String logic;
    protected Model model;
    protected Collection<String> positives;
    protected Collection<String> negatives;
    protected ClassExpression input;
    protected ScoredIRI[] expectedClasses;
    protected ScoredIRI[] expectedProperties;

    public SparqlBasedSuggestorTest(ClassExpression input, String logic, Model model, String[] positives,
            String[] negatives, ScoredIRI[] expectedClasses, ScoredIRI[] expectedProperties) {
        super();
        this.input = input;
        this.logic = logic;
        this.model = model;
        this.positives = Arrays.asList(positives);
        this.negatives = Arrays.asList(negatives);
        this.expectedClasses = expectedClasses;
        Arrays.sort(this.expectedClasses, this);
        this.expectedProperties = expectedProperties;
        Arrays.sort(this.expectedProperties, this);
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
            Collection<ScoredIRI> suggestions;
            ScoredIRI[] result;

            suggestions = suggestor.suggestClass(positives, negatives, input);
            result = suggestions.toArray(ScoredIRI[]::new);
            Arrays.sort(result, this);
            Assert.assertArrayEquals(expectedClasses, result);

            suggestions = suggestor.suggestProperty(positives, negatives, input);
            result = suggestions.toArray(ScoredIRI[]::new);
            Arrays.sort(result, this);
            Assert.assertArrayEquals(expectedProperties, result);
        }
    }

    @Override
    public int compare(ScoredIRI o1, ScoredIRI o2) {
        int diff = 0;
        if (o1 != null) {
            if (o1.iri != null) {
                if (o2.iri != null) {
                    diff = o1.iri.compareTo(o2.iri);
                } else {
                    return 1;
                }
            } else {
                diff = o2.iri != null ? -1 : 0;
            }
        }
        if (diff == 0) {
            diff = Boolean.compare(o1.inverted, o2.inverted);
        }
        if (diff == 0) {
            diff = Integer.compare(o1.posCount, o2.posCount);
        }
        if (diff == 0) {
            diff = Integer.compare(o1.negCount, o2.negCount);
        }
        return diff;
    }
}
