package org.example.cel.refine.suggest.sparql;

import java.util.ArrayList;
import java.util.List;

import org.example.cel.expression.ClassExpression;
import org.example.cel.expression.Junction;
import org.example.cel.expression.NamedClass;
import org.example.cel.expression.SimpleQuantifiedRole;
import org.example.cel.refine.suggest.Suggestor;
import org.example.cel.refine.suggest.sparql.ExpressionPreProcessor;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(Parameterized.class)
public class ExpressionPreProcessorTest {

    @Parameters
    public static List<Object[]> parameters() {
        List<Object[]> testCases = new ArrayList<>();

        // Some simple cases without any real activity
        testCases.add(new Object[] { new NamedClass("A"), new NamedClass("A") });
        testCases.add(new Object[] { new SimpleQuantifiedRole(true, "r", false, Suggestor.CONTEXT_POSITION_MARKER),
                new SimpleQuantifiedRole(true, "r", false, Suggestor.CONTEXT_POSITION_MARKER) });
        testCases.add(new Object[] {
                new SimpleQuantifiedRole(true, "r1", false,
                        new SimpleQuantifiedRole(true, "r2", false, Suggestor.CONTEXT_POSITION_MARKER)),
                new SimpleQuantifiedRole(true, "r1", false,
                        new SimpleQuantifiedRole(true, "r2", false, Suggestor.CONTEXT_POSITION_MARKER)) });
        // (A⊔B)⊓C -> (A⊓C)⊔(B⊓C)
        testCases.add(new Object[] {
                new Junction(true, new Junction(false, new NamedClass("A"), new NamedClass("B")), new NamedClass("C")),
                new Junction(false, new Junction(true, new NamedClass("A"), new NamedClass("C")),
                        new Junction(true, new NamedClass("B"), new NamedClass("C"))) });
        // Covering this case doesn't seem to be necessary
//        // ((A⊔B)⊓C)⊓D -> (A⊓C⊓D)⊔(B⊓C⊓D)
//        testCases.add(new Object[] {
//                new Junction(true,
//                        new Junction(true, new Junction(false, new NamedClass("A"), new NamedClass("B")),
//                                new NamedClass("C")),
//                        new NamedClass("D")),
//                new Junction(false, new Junction(true, new NamedClass("A"), new NamedClass("C"), new NamedClass("D")),
//                        new Junction(true, new NamedClass("B"), new NamedClass("C"), new NamedClass("D"))) });
        // (∀r.(A⊔B)⊔C)⊓D -> (C⊓D)⊔(∀r.(A⊔B)⊓D)
        testCases.add(new Object[] {
                new Junction(true, new Junction(false,
                        new SimpleQuantifiedRole(false, "r", false,
                                new Junction(false, new NamedClass("A"), new NamedClass("B"))),
                        new NamedClass("C")), new NamedClass("D")),
                new Junction(false, new Junction(true, new NamedClass("C"), new NamedClass("D")),
                        new Junction(true,
                                new SimpleQuantifiedRole(false, "r", false,
                                        new Junction(false, new NamedClass("A"), new NamedClass("B"))),
                                new NamedClass("D"))) });

        return testCases;
    }

    private ClassExpression givenExpression;
    private ClassExpression expectedExpression;

    public ExpressionPreProcessorTest(ClassExpression givenExpression, ClassExpression expectedExpression) {
        super();
        this.givenExpression = givenExpression;
        this.expectedExpression = expectedExpression;
    }

    @Test
    public void test() {
        ExpressionPreProcessor preprocessor = new ExpressionPreProcessor();
        Assert.assertEquals(expectedExpression, preprocessor.preprocess(givenExpression));
    }

}
