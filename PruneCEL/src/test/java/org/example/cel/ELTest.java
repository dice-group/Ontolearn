package org.example.cel;

import java.util.ArrayList;
import java.util.List;

import org.apache.jena.rdf.model.Model;
import org.apache.jena.rdf.model.Property;
import org.apache.jena.rdf.model.Resource;
import org.apache.jena.rdf.model.ResourceFactory;
import org.apache.jena.vocabulary.RDF;
import org.example.cel.expression.ClassExpression;
import org.example.cel.expression.Junction;
import org.example.cel.expression.NamedClass;
import org.example.cel.expression.SimpleQuantifiedRole;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(Parameterized.class)
public class ELTest extends AbstractCELTest {

    @Parameters
    public static List<Object[]> parameters() {
        List<Object[]> testCases = new ArrayList<>();

        Model model;
        ClassExpression expected;
        Resource pos1 = ResourceFactory.createResource("http://example.org/pos1");
        Resource pos2 = ResourceFactory.createResource("http://example.org/pos2");

        Resource neg1 = ResourceFactory.createResource("http://example.org/neg1");
        Resource neg2 = ResourceFactory.createResource("http://example.org/neg2");
        Resource[] individuals = new Resource[] { pos1, pos2, neg1, neg2 };

        Resource classA = ResourceFactory.createResource("http://example.org/classA");
        Resource classB = ResourceFactory.createResource("http://example.org/classB");
        Resource classC = ResourceFactory.createResource("http://example.org/classC");
        Resource[] classes = new Resource[] { classA, classB, classC };

        Property role1 = ResourceFactory.createProperty("http://example.org/role1");
        Resource[] roles = new Resource[] { role1 };
        // Basic example for a single named concept
        // A(pos1,pos2),
        model = TestHelper.initModel(classes, roles, individuals);
        model.add(pos1, RDF.type, classA);
        model.add(pos2, RDF.type, classA);
        expected = new NamedClass(classA.getURI());
        testCases.add(new Object[] { model, new String[] { pos1.getURI(), pos2.getURI() },
                new String[] { neg1.getURI(), neg2.getURI() }, expected });

        // Basic example for concept conjunction
        // A(pos1, pos2, neg1), B(pos1, pos2, neg2)
        model = TestHelper.initModel(classes, roles, individuals);
        model.add(pos1, RDF.type, classA);
        model.add(pos1, RDF.type, classB);
        model.add(pos2, RDF.type, classA);
        model.add(pos2, RDF.type, classB);
        model.add(neg1, RDF.type, classA);
        model.add(neg2, RDF.type, classB);
        expected = new Junction(true, new NamedClass(classA.getURI()), new NamedClass(classB.getURI()));
        testCases.add(new Object[] { model, new String[] { pos1.getURI(), pos2.getURI() },
                new String[] { neg1.getURI(), neg2.getURI() }, expected });

        // Basic example for a (limited) existential quantifier
        // r(pos1,c), r(pos2,c)
        model = TestHelper.initModel(classes, roles, individuals);
        model.add(pos1, role1, classC);
        model.add(pos2, role1, classC);
        expected = new SimpleQuantifiedRole(true, role1.getURI(), false, NamedClass.TOP);
        testCases.add(new Object[] { model, new String[] { pos1.getURI(), pos2.getURI() },
                new String[] { neg1.getURI(), neg2.getURI() }, expected });

        return testCases;
    }

    public ELTest(Model model, String[] positives, String[] negatives, ClassExpression expected) {
        super("EL", model, positives, negatives, expected);
    }
}
