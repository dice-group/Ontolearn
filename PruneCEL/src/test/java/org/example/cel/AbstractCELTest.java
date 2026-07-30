package org.example.cel;

import java.util.Arrays;
import java.util.List;

import org.aksw.jenax.arq.connection.core.QueryExecutionFactory;
import org.aksw.jenax.connection.query.QueryExecutionFactoryDataset;
import org.apache.jena.query.Dataset;
import org.apache.jena.query.DatasetFactory;
import org.apache.jena.rdf.model.Model;
import org.apache.jena.vocabulary.OWL2;
import org.apache.jena.vocabulary.RDF;
import org.example.cel.DescriptionLogic;
import org.example.cel.PruneCEL;
import org.example.cel.expression.ClassExpression;
import org.example.cel.expression.ScoredClassExpression;
import org.example.cel.refine.suggest.sparql.SparqlBasedSuggestor;
import org.example.cel.score.AccuracyCalculator;
import org.example.cel.score.ScoreCalculatorFactory;
import org.junit.Assert;
import org.junit.Test;

public abstract class AbstractCELTest {

    protected String logicName;
    protected Model model;
    protected String[] positives;
    protected String[] negatives;
    protected ClassExpression expected;

    public AbstractCELTest(String logicName, Model model, String[] positives, String[] negatives,
            ClassExpression expected) {
        super();
        this.logicName = logicName;
        this.model = model;
        this.positives = positives;
        this.negatives = negatives;
        this.expected = expected;
    }

    protected ScoreCalculatorFactory createScoreCalculatorFactory() {
        return new AccuracyCalculator.Factory();
    }

    @Test
    public void test() throws Exception {
        DescriptionLogic logic = DescriptionLogic.parse(logicName);
        Assert.assertNotNull(logic);

        Dataset dataset = DatasetFactory.create(model);
        try (QueryExecutionFactory qef = new QueryExecutionFactoryDataset(dataset);
                SparqlBasedSuggestor suggestor = new SparqlBasedSuggestor(qef, logic)) {
            suggestor.addToClassBlackList(OWL2.NamedIndividual.getURI());
            suggestor.addToPropertyBlackList(RDF.type.getURI());

            PruneCEL cel = new PruneCEL(suggestor, logic, createScoreCalculatorFactory());
            // 100 iterations and 10 seconds maximum
            cel.setMaxIterations(100);
//            cel.setMaxTime(10000);

            List<ScoredClassExpression> expressions = cel.findClassExpression(Arrays.asList(positives),
                    Arrays.asList(negatives), null, null);
            Assert.assertFalse("Didn't find any expressions.", expressions.isEmpty());
            // Compare the best expression
            StringBuilder builder = new StringBuilder();
            builder.append("The found expression ");
            expressions.get(0).getClassExpression().toString(builder);
            builder.append(" does not match the expected expression ");
            expected.toString(builder);
            builder.append("!");
            Assert.assertEquals(builder.toString(), expected, expressions.get(0).getClassExpression());
        }
    }
}
