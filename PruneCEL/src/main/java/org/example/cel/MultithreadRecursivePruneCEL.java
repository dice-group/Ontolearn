package org.example.cel;

import java.io.OutputStream;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
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
import org.example.cel.refine.RefinementOperator;
import org.example.cel.refine.SuggestorBasedRefinementOperator;
import org.example.cel.refine.suggest.ExtendedSuggestor;
import org.example.cel.refine.suggest.SelectionScores;
import org.example.cel.score.ScoreCalculator;
import org.example.cel.score.ScoreCalculatorFactory;
import org.example.cel.sparql.InstanceRetriever;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MultithreadRecursivePruneCEL extends PruneCEL {

    private static final Logger LOGGER = LoggerFactory.getLogger(MultithreadRecursivePruneCEL.class);

    protected ExecutorService executor = Executors.newCachedThreadPool();

    public MultithreadRecursivePruneCEL(ExtendedSuggestor suggestor, DescriptionLogic logic,
            ScoreCalculatorFactory calculatorFactory, InstanceRetriever retriever) {
        super(suggestor, logic, calculatorFactory);
        this.retriever = retriever;
    }

    public List<ScoredClassExpression> findClassExpression(Collection<String> positive, Collection<String> negative,
            OutputStream logStream, IntermediateResultPrinter iResultPrinter, long startTime, long timeToStop) {
        // ((SuggestorBasedRefinementOperator) rho).setLogStream(logStream);
        Future<List<ScoredClassExpression>> result = executor.submit(new SearchTask(positive, negative,timeToStop));
        try {
            return result.get();
        } catch (Exception e) {
            LOGGER.error("Got an exception while waiting for the search task to terminate. Returning null.", e);
            return null;
        }
    }

    public class SearchTask implements Callable<List<ScoredClassExpression>> {

        public Collection<String> positives;
        public Collection<String> negatives;
        public RefinementOperator rho;
        public ScoreCalculator scoreCalculator;
        public double perfectScore;

        protected int maxIterations;
        protected long timeToStop;

        protected boolean cancelled = false;

        public TopDoubleObjectCollection<ScoredClassExpression> topExpressions = new TopDoubleObjectCollection<>(10,
                false);
        public Set<ScoredClassExpression> seenExpressions = new HashSet<>();
        public Queue<ScoredClassExpression> queue = new PriorityQueue<ScoredClassExpression>(
                new ScoredCEComparatorForRefinement());
        public TopIntObjectCollection<ScoredClassExpression> mostPreciseExpressions = new TopIntObjectCollection<>(10,
                false);

        public Map<String, SearchSubTask> subTasks = new HashMap<>();

        public SearchTask(Collection<String> positives, Collection<String> negatives, long timeToStop) {
            super();
            this.positives = positives;
            this.negatives = negatives;
            this.timeToStop = timeToStop;
            this.scoreCalculator = calculatorFactory.create(positives.size(), negatives.size());
            this.perfectScore = scoreCalculator.getPerfectScore();
            this.rho = new SuggestorBasedRefinementOperator(suggestor, logic, scoreCalculator, positives, negatives);
        }

        @SuppressWarnings("unchecked")
        @Override
        public List<ScoredClassExpression> call() throws Exception {
            LOGGER.info("Starting sub task!");
            // TODO implement the search (above) as a task
            // TODO the job is allowed to start a sub-task, if it finds an expression that
            // has precision = 1.0 (recall 1.0 would be possible as well, if we swap pos and
            // neg; but we have to watch out that we do not start that too early (e.g., with
            // top, that already has a recall of 1.0))
            // TODO ensure that the job keeps track for which pos / neg combinations it has
            // started a job. Otherwise, it may start a lot of them if statements tend to
            // select the same subset
            // TODO A job with children should check from time to time whether their
            // children are done
            // TODO if a task finds a 1.0, it has to be able to kill its children since it
            // is not interested in their result, anymore
            try {
                Collection<ScoredClassExpression> newExpressions;
                ScoredClassExpression nextBestExpression;
                // Create Top expression
                nextBestExpression = scoreCalculator.score(NamedClass.TOP, positives.size(), negatives.size(), false);
                queue.add(nextBestExpression);
                topExpressions.add(nextBestExpression.getClassificationScore(), nextBestExpression);
                int iterationCount = 0;
                // Iterate over the queue while
                while (// 1. the queue is not empty
                !queue.isEmpty() &&
                // 2. the best expression has not the perfect score
                        (topExpressions.values.length == 0 || topExpressions.values[0] < perfectScore) &&
                        // 3. We haven't reached the maximum number of iterations (if defined)
                        (maxIterations == 0 || iterationCount < maxIterations) &&
                        // 4. We haven't reached the maximum amount of time that we are allowed to use
                        (timeToStop == 0 || (System.currentTimeMillis() < timeToStop)) && !cancelled) {
                    nextBestExpression = queue.poll();
                    LOGGER.info("Refining rScore={}, cScore={}, ce={}", nextBestExpression.getRefinementScore(),
                            nextBestExpression.getClassificationScore(), nextBestExpression.getClassExpression());
                    // Refine this expression
                    newExpressions = rho.refine(nextBestExpression.getClassExpression(), timeToStop);
                    if (cancelled) {
                        return null;
                    }
                    // Check the expressions
                    for (ScoredClassExpression newExpression : newExpressions) {
                        // If 1) we haven't seen this before AND 2a) we are configured to not further
                        // look into this OR 2b) the classification score of this
                        // new expression is better than the expression that we refined OR 2c) we
                        // extended the previous expression by adding an edge
                        if (seenExpressions.add(newExpression) && (!skipNonImprovingStmts || (nextBestExpression
                                .getClassificationScore() < newExpression.getClassificationScore())
                                || (newExpression.isAddedEdge()))) {
                            queue.add(newExpression);
                            topExpressions.add(newExpression.getClassificationScore(), newExpression);
                            if ((getPrecision(newExpression) >= precisionThreshold) && (newExpression.getPosCount() > 1)
                                    && (newExpression.getPosCount() < (positives.size() - 1))) {
                                startSubTaskIfPossible(newExpression);
                            }
                        }
                    }
                    iterationCount++;
                    if (cancelled) {
                        return null;
                    }
                    // Check sub tasks
                    for (SearchSubTask subTask : subTasks.values()) {
                        if (subTask.result.isDone()) {
                            // Merge result with the results of the sub task
                            merge(subTask);
                        }
                    }
                }
                LOGGER.info("Stopping search. Saw {} expressions.", seenExpressions.size());
                // If there was no perfect solution, the sub searches should terminate and we
                // should wait for that
                if (topExpressions.values.length == 0 || topExpressions.values[0] > perfectScore) {
                    for (SearchSubTask subTask : subTasks.values()) {
                        merge(subTask);
                    }
                }
                return Objects.requireNonNullElse(
                        Stream.of(topExpressions.getObjects()).map(o -> (ScoredClassExpression) o).toList(),
                        Collections.EMPTY_LIST);
            } finally {
                // stop all sub tasks
                for (SearchSubTask subTask : subTasks.values()) {
                    subTask.result.cancel(true);
                    subTask.task.cancel();
                }
            }
        }

        public void startSubTaskIfPossible(ScoredClassExpression preciseExp) {
            Set<String> remainingPositives = new HashSet<String>(positives);
            @SuppressWarnings("unchecked")
            Set<String> selectedInstances = retriever.retrieveInstances(preciseExp.getClassExpression(), positives,
                    SetUtils.EMPTY_SET);
            remainingPositives.removeAll(selectedInstances);
            String remainingPosKey = generateKey(remainingPositives);
            if (!subTasks.containsKey(remainingPosKey)) {
                LOGGER.info("Removing positives selected by {}", preciseExp);
                LOGGER.info("Trying to solve the remaining sub problem recursively for the remaining {} positives ...",
                        remainingPositives.size());
                SearchTask subTask = new SearchTask(remainingPositives, negatives, timeToStop);
                subTasks.put(remainingPosKey, new SearchSubTask(subTask, executor.submit(subTask), preciseExp));
            }
        }

        public void merge(SearchSubTask finishedSubTask) {
            List<ScoredClassExpression> subSolutions;
            if (!finishedSubTask.result.isCancelled()) {
                try {
                    subSolutions = finishedSubTask.result.get();
                    for (ScoredClassExpression subSolution : subSolutions) {
                        if (subSolution != null) {
                            ClassExpression newExpression = new Junction(false,
                                    finishedSubTask.parentExpression.getClassExpression(),
                                    subSolution.getClassExpression());
                            SelectionScores scores = suggestor.scoreExpression(newExpression, positives, negatives);
                            ScoredClassExpression scoredExp = scoreCalculator.score(newExpression, scores.getPosCount(),
                                    scores.getNegCount(), false);
                            topExpressions.add(scoredExp.getClassificationScore(), scoredExp);
                        }
                    }
                } catch (Exception e) {
                    LOGGER.error("Error while accessing the solution of a sub task. Ignoring it.", e);
                    e.printStackTrace();
                }
            }
        }

        public void cancel() {
            this.cancelled = true;
        }

        public static String generateKey(Set<String> remainingPositives) {
            return remainingPositives.stream().sorted()
                    .collect(StringBuilder::new, StringBuilder::append, StringBuilder::append).toString();
        }

    }

    public static class SearchSubTask {
        public SearchTask task;
        public Future<List<ScoredClassExpression>> result;
        public ScoredClassExpression parentExpression;

        public SearchSubTask(SearchTask task, Future<List<ScoredClassExpression>> result,
                ScoredClassExpression parentExpression) {
            super();
            this.task = task;
            this.result = result;
            this.parentExpression = parentExpression;
        }
    }

}
