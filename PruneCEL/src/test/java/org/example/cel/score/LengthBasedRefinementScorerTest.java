package org.example.cel.score;

import java.util.ArrayList;
import java.util.List;

import org.example.cel.expression.ClassExpression;
import org.example.cel.expression.Junction;
import org.example.cel.expression.NamedClass;
import org.example.cel.expression.SimpleQuantifiedRole;
import org.example.cel.score.AccuracyCalculator;
import org.example.cel.score.LengthBasedRefinementScorer;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(Parameterized.class)
public class LengthBasedRefinementScorerTest {

    public static final double LENGTH_PENALTY = 0.01;

    @Parameters
    public static List<Object[]> parameters() {
        List<Object[]> testCases = new ArrayList<>();

        final NamedClass A = new NamedClass("A");
        final NamedClass B = new NamedClass("B");
        // A
        testCases.add(new Object[] { 10, 20, 5, 10, A, 0.5 - LENGTH_PENALTY });
        // A ⊓ B
        testCases.add(new Object[] { 10, 20, 5, 10, new Junction(true, A, B), 0.5 - 3 * LENGTH_PENALTY });
        testCases.add(new Object[] { 10, 20, 5, 10, new Junction(false, A, B), 0.5 - 3 * LENGTH_PENALTY });
        // ∃r.A
        testCases.add(new Object[] { 10, 20, 5, 10, new SimpleQuantifiedRole(true, "r", false, A),
                0.5 - 3 * LENGTH_PENALTY });
        // ∃r-.A
        testCases.add(
                new Object[] { 10, 20, 5, 10, new SimpleQuantifiedRole(true, "r", true, A), 0.5 - 3 * LENGTH_PENALTY });
        // ∃r.(A ⊓ B)
        testCases.add(new Object[] { 10, 20, 5, 10, new SimpleQuantifiedRole(true, "r", true, new Junction(true, A, B)),
                0.5 - 5 * LENGTH_PENALTY });
        // ∃r.(A ⊓ B ⊓ ∃r.B)
        testCases.add(new Object[] { 10, 20, 5, 10,
                new SimpleQuantifiedRole(true, "r", true,
                        new Junction(true, A, B, new SimpleQuantifiedRole(true, "r", true, B))),
                0.5 - 8 * LENGTH_PENALTY });
        // (A ⊓ ∃r.B)
        testCases.add(new Object[] { 10, 20, 5, 10, new Junction(true, A, new SimpleQuantifiedRole(true, "r", true, B)),
                0.5 - 5 * LENGTH_PENALTY });
        // ∃r1.∃r2.∃r3.A
        testCases.add(new Object[] { 10, 20, 5, 10,
                new SimpleQuantifiedRole(true, "r1", false,
                        new SimpleQuantifiedRole(true, "r2", false, new SimpleQuantifiedRole(true, "r3", false, A))),
                0.5 - 7 * LENGTH_PENALTY });

        return testCases;
    }

    private int numPosExamples;
    private int numNegExamples;
    private int numPosSelected;
    private int numNegSelected;
    private ClassExpression classExpression;
    private double expected;

    public LengthBasedRefinementScorerTest(int numPosExamples, int numNegExamples, int numPosSelected,
            int numNegSelected, ClassExpression classExpression, double expected) {
        super();
        this.numPosExamples = numPosExamples;
        this.numNegExamples = numNegExamples;
        this.numPosSelected = numPosSelected;
        this.numNegSelected = numNegSelected;
        this.classExpression = classExpression;
        this.expected = expected;
    }

    @Test
    public void test() {
        Assert.assertTrue("Faulty test case. There are more positive selected than positive examples exist.",
                numPosSelected <= numPosExamples);
        Assert.assertTrue("Faulty test case. There are more negative selected than negative examples exist.",
                numNegSelected <= numNegExamples);
        LengthBasedRefinementScorer calculator = new LengthBasedRefinementScorer(
                new AccuracyCalculator(numPosExamples, numNegExamples), LENGTH_PENALTY);
        Assert.assertEquals(expected,
                calculator.calculateRefinementScore(numPosSelected, numNegSelected,
                        calculator.calculateClassificationScore(numPosSelected, numNegSelected), classExpression),
                0.00001);
    }
}
