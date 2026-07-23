package org.example.cel;

import org.apache.jena.rdf.model.Model;
import org.apache.jena.rdf.model.ModelFactory;
import org.apache.jena.rdf.model.Resource;
import org.apache.jena.vocabulary.OWL;
import org.apache.jena.vocabulary.OWL2;
import org.apache.jena.vocabulary.RDF;
import org.junit.Ignore;

@Ignore
public class TestHelper {

    public static Model initModel(Resource[] classes, Resource[] roles, Resource[] individuals) {
        Model model = ModelFactory.createDefaultModel();
        for (int i = 0; i < classes.length; ++i) {
            model.add(classes[i], RDF.type, OWL.Class);
        }
        for (int i = 0; i < roles.length; ++i) {
            model.add(roles[i], RDF.type, RDF.Property);
        }
        for (int i = 0; i < individuals.length; ++i) {
            model.add(individuals[i], RDF.type, OWL2.NamedIndividual);
        }
        return model;
    }
}
