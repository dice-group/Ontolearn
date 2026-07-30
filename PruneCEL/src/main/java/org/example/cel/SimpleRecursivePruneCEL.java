package org.example.cel;

import java.io.OutputStream;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;
import java.util.stream.Stream;

import org.apache.commons.collections.SetUtils;
import org.dice_research.topicmodeling.commons.collections.TopDoubleObjectCollection;
import org.dice_research.topicmodeling.commons.collections.TopIntObjectCollection;
import org.example.cel.expression.ClassExpression;
import org.example.cel.expression.Junction;
import org.example.cel.expression.NamedClass;
import org.example.cel.expression.ScoredCEComparatorForRefinement;
import org.example.cel.expression.ScoredClassExpression;
import org.example.cel.io.IntermediateResultPrinter;
import org.example.cel.io.NullIntermediateResultPrinter;
import org.example.cel.refine.RefinementOperator;
import org.example.cel.refine.SuggestorBasedRefinementOperator;
import org.example.cel.refine.suggest.ExtendedSuggestor;
import org.example.cel.refine.suggest.SelectionScores;
import org.example.cel.score.ScoreCalculator;
import org.example.cel.score.ScoreCalculatorFactory;
import org.example.cel.sparql.InstanceRetriever;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SimpleRecursivePruneCEL extends PruneCEL {

    private static final Logger LOGGER = LoggerFactory.getLogger(SimpleRecursivePruneCEL.class);

    protected static final int MAX_ITERATIONS_FOR_RECURSION = 10;

    public SimpleRecursivePruneCEL(ExtendedSuggestor suggestor, DescriptionLogic logic,
            ScoreCalculatorFactory calculatorFactory, InstanceRetriever retriever) {
        super(suggestor, logic, calculatorFactory);
        this.retriever = retriever;
    }

    public List<ScoredClassExpression> findClassExpression(Collection<String> positive, Collection<String> negative,
            OutputStream logStream, IntermediateResultPrinter iResultPrinter, long startTime, long timeToStop) {
        return findClassExpression(positive, negative, logStream, iResultPrinter, timeToStop, maxIterations);
    }

    public List<ScoredClassExpression> findClassExpression(Collection<String> positive, Collection<String> negative,
            OutputStream logStream, IntermediateResultPrinter iResultPrinter, long timeToStop, int localMaxIterations) {
        LOGGER.info("Starting search with {} positives and {} negatives.", positive.size(), negative.size());
        if (iResultPrinter == null) {
            iResultPrinter = new NullIntermediateResultPrinter();
        }
        int minRecursionPosCount = (int) Math.ceil(Math.max(positive.size() * 0.1, 2.0));
        TopDoubleObjectCollection<ScoredClassExpression> topExpressions = new TopDoubleObjectCollection<>(10, false);
        Set<ScoredClassExpression> seenExpressions = new HashSet<>();
        Queue<ScoredClassExpression> queue = new PriorityQueue<ScoredClassExpression>(
                new ScoredCEComparatorForRefinement());
        ScoreCalculator scoreCalculator = calculatorFactory.create(positive.size(), negative.size());
        double perfectScore = scoreCalculator.getPerfectScore();

        TopIntObjectCollection<ScoredClassExpression> mostPreciseExpressions = new TopIntObjectCollection<>(10, false);

        RefinementOperator rho = new SuggestorBasedRefinementOperator(suggestor, logic, scoreCalculator, positive,
                negative);
        ((SuggestorBasedRefinementOperator) rho).setLogStream(logStream);
        ((SuggestorBasedRefinementOperator) rho).setDebugMode(debugMode);

        Set<String> subProblemKeys = new HashSet<>();
        Collection<ScoredClassExpression> newExpressions;
        ScoredClassExpression nextBestExpression;
        // Create Top expression
        nextBestExpression = scoreCalculator.score(NamedClass.TOP, positive.size(), negative.size(), false);
        queue.add(nextBestExpression);
        topExpressions.add(nextBestExpression.getClassificationScore(), nextBestExpression);
        iResultPrinter.printIntermediateResults(topExpressions);
        int iterationCount = 0;
        // Iterate over the queue while
        while (// 1. the queue is not empty
        !queue.isEmpty() &&
        // 2. the best expression has not the perfect score
                (topExpressions.values.length == 0 || topExpressions.values[0] < perfectScore) &&
                // 3. We haven't reached the maximum number of iterations (if defined)
                (localMaxIterations == 0 || iterationCount < localMaxIterations) &&
                // 4. We haven't reached the maximum amount of time that we are allowed to use
                (maxTime == 0 || (System.currentTimeMillis() < timeToStop))) {
            nextBestExpression = queue.poll();
            LOGGER.info("Refining rScore={}, cScore={}, ce={}", nextBestExpression.getRefinementScore(),
                    nextBestExpression.getClassificationScore(), nextBestExpression.getClassExpression());
            // Refine this expression
            newExpressions = rho.refine(nextBestExpression.getClassExpression(), timeToStop);
            // Check the expressions
            for (ScoredClassExpression newExpression : newExpressions) {
                // If 1) we haven't seen this before AND 2a) we are configured to not further
                // look into this OR 2b) the classification score of this
                // new expression is better than the expression that we refined OR 2c) we
                // extended the previous expression by adding an edge
                if (seenExpressions.add(newExpression) && (!skipNonImprovingStmts
                        || (nextBestExpression.getClassificationScore() < newExpression.getClassificationScore())
                        || (newExpression.isAddedEdge()))) {
                    queue.add(newExpression);
                    topExpressions.add(newExpression.getClassificationScore(), newExpression);
                    if ((getPrecision(newExpression) >= precisionThreshold) && (newExpression.getPosCount() >= minRecursionPosCount)
                            && (newExpression.getPosCount() < (positive.size() - 1))) {
                        mostPreciseExpressions.add(newExpression.getPosCount(), newExpression);
                    }
                }
            }
            if (
            // The best expression has not the perfect score
            (topExpressions.values.length == 0 || topExpressions.values[0] < perfectScore) &&
            // There are very precise expressions
                    mostPreciseExpressions.size() > 0) {
                int pos = 0;
                int length = mostPreciseExpressions.size();
                while ((pos < length) &&
                // The best expression has not the perfect score
                        (topExpressions.values.length == 0 || topExpressions.values[0] < perfectScore) &&
                        // Maximum iterations haven't been reached yet
                        (localMaxIterations == 0 || iterationCount < localMaxIterations)
                        // Maximum runtime hasn't been reached yet
                        && (maxTime == 0 || (System.currentTimeMillis() < timeToStop))) {
                    // Try recursion
                    // 1. Figure out the remaining positives
                    ScoredClassExpression mostPrecise = (ScoredClassExpression) mostPreciseExpressions
                            .getObjects()[pos];
                    Set<String> remainingPositives = new HashSet<String>(positive);
                    @SuppressWarnings("unchecked")
                    Set<String> selectedInstances = retriever.retrieveInstances(mostPrecise.getClassExpression(),
                            positive, SetUtils.EMPTY_SET);
                    remainingPositives.removeAll(selectedInstances);
                    // 2. Generate a key from it
                    String remainingPosKey = generateKey(remainingPositives);
                    // If we do not already know this key, we have identified a new sub problem
                    if ((subProblemKeys.add(remainingPosKey)) && (remainingPositives.size() > 1)) {
                        LOGGER.info("Removing positives selected by {}", mostPrecise);
                        LOGGER.info(
                                "Trying to solve the remaining sub problem recursively for the remaining {} positives ...",
                                remainingPositives.size());
                        iResultPrinter.recursionStarts();
                        List<ScoredClassExpression> subProblemResults = findClassExpression(remainingPositives,
                                negative, logStream, iResultPrinter, timeToStop, MAX_ITERATIONS_FOR_RECURSION);
                        iResultPrinter.recursionEnds();
                        if (subProblemResults.size() > 1) {
                            ClassExpression newExpression = new Junction(false, mostPrecise.getClassExpression(),
                                    subProblemResults.get(0).getClassExpression());
                            SelectionScores scores = suggestor.scoreExpression(newExpression, positive, negative);
                            ScoredClassExpression scoredExp = scoreCalculator.score(newExpression, scores.getPosCount(),
                                    scores.getNegCount(), false);
                            topExpressions.add(scoredExp.getClassificationScore(), scoredExp);
                            queue.add(scoredExp);
                            LOGGER.info("Recursion resulted in expression with cscore={} ce={}",
                                    scoredExp.getClassificationScore(), scoredExp.getClassExpression());
                        }
                    }
                    ++pos;
                }
                mostPreciseExpressions.clear();
            }
            iterationCount++;
            iResultPrinter.printIntermediateResults(topExpressions);
        }
        LOGGER.info("Stopping search. Saw {} expressions.", seenExpressions.size());
        return Stream.of(topExpressions.getObjects()).map(o -> (ScoredClassExpression) o).toList();
    }

    public static String generateKey(Set<String> remainingPositives) {
        return remainingPositives.stream().sorted()
                .collect(StringBuilder::new, StringBuilder::append, StringBuilder::append).toString();
    }

}
