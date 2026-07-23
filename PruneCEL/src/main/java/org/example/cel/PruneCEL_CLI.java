package org.example.cel;

import java.io.*;
import java.nio.file.FileVisitResult;
import java.nio.file.Paths;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.jena.vocabulary.OWL2;
import org.apache.jena.vocabulary.RDF;
import org.example.cel.expression.ClassExpression;
import org.example.cel.expression.Junction;
import org.example.cel.expression.ScoredClassExpression;
import org.example.cel.io.IntermediateResultPrinter;
import org.example.cel.io.LearningProblem;
import org.example.cel.io.csv.CSVIntermediateResultPrinter;
import org.example.cel.io.json.JSONLearningProblemReader;
import org.example.cel.refine.suggest.SelectionScores;
import org.example.cel.refine.suggest.sparql.SparqlBasedSuggestor;
import org.example.cel.score.AccuracyCalculator;
import org.example.cel.score.AvoidingPickySolutionsDecorator;
import org.example.cel.score.BalancedAccuracyCalculator;
import org.example.cel.score.F1MeasureCalculator;
import org.example.cel.score.LengthBasedRefinementScorer;
import org.example.cel.score.ScoreCalculatorFactory;

import java.util.Collections;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.io.IOException;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.List;



public class PruneCEL_CLI {

    private static boolean isHeaderPrinted = false;
    private static ClassExpression currentClassExpression = null;
    private static ClassExpression BestClassExpression = null;


    public static void main(String[] args) {
        // Check if there are enough arguments
        if (args.length < 14) {
            System.out.println("Insufficient arguments provided.");
            System.out.println("Usage: --sparqlUrl <url> --ontology <description_logic> ...");
            return;
        }

        String sparqlUrl = "";
        String descriptionLogic = "";
        int accuracyFunction = 0;
        boolean punishLongExpression = false;
        boolean avoidPickySolutionsDecorator = false;
        int iteration = 0;
        int time = 0;
        boolean recursive = false;
        boolean skipNonImprovingStmts = false;
        String tfjson = "";
        String savePlace = "";
        boolean cluster = false;
        int folds = 0;
        String foldTrainTestSavePath = "";

        // Parse command-line arguments
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--sparqlUrl":
                    sparqlUrl = args[++i];
                    break;
                case "--ontology":
                    descriptionLogic = args[++i];
                    break;
                case "--accuracyfunction":
                    accuracyFunction = Integer.parseInt(args[++i]);
                    break;
                case "--punishLongExpression":
                    punishLongExpression = Boolean.parseBoolean(args[++i]);
                    break;
                case "--avoidPickySolutionsDecorator":
                    avoidPickySolutionsDecorator = Boolean.parseBoolean(args[++i]);
                    break;
                case "--iteration":
                    iteration = Integer.parseInt(args[++i]);
                    break;
                case "--time":
                    time = Integer.parseInt(args[++i]);
                    break;
                case "--recursive":
                    recursive = Boolean.parseBoolean(args[++i]);
                    break;
                case "--skipNone":
                    skipNonImprovingStmts = Boolean.parseBoolean(args[++i]);
                    break;
                case "--inputFile":
                    tfjson = args[++i];
                    break;
                case "--outputFile":
                    savePlace = args[++i];
                    break;
                case "--cluster":
                    cluster = Boolean.parseBoolean(args[++i]);
                    break;
                case "--folds":
                    folds = Integer.parseInt(args[++i]);
                    break;
                case "--foldTrainTestSavePath":
                    foldTrainTestSavePath = args[++i];
                    break;
                default:
                    System.out.println("Unknown argument: " + args[i]);
                    return;
            }
        }

        // Run the method with parsed arguments
        try {
            runPruneCEL(sparqlUrl, descriptionLogic, accuracyFunction, punishLongExpression, avoidPickySolutionsDecorator,
                iteration, time, recursive, skipNonImprovingStmts, tfjson, savePlace, cluster, folds, foldTrainTestSavePath);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void runPruneCEL(String spqrql_endpoint, String description_logic, int accuracyfunction,
                                   boolean punishlongexpression, boolean AvoidPickySolutionsDecorator, int iteration, int time,
                                   boolean recursive, boolean setSkipNonImprovingStmts, String tfjson, String saveplace, boolean cluster,
                                   int folds, String Fold_train_test_save_path)
        throws Exception {

        boolean isbascore = false;
        boolean isf1score = false;
        boolean isacscore = false;


        // XXX Create folds
        if (folds != 1) {
            String filePath = "./././././" + Fold_train_test_save_path;
            File file = new File(filePath);
            if (file.exists()) {

            } else {
                creat_dir_for_fold(folds, tfjson, Fold_train_test_save_path);
            }
        }

        // XXX Set SPARQL endpoint
        String endpoint = spqrql_endpoint;

        // XXX Set description logic
        DescriptionLogic logic = DescriptionLogic.parse(description_logic);

        ScoreCalculatorFactory factory = null;

        // XXX Choose F1 or balanced accuracy or accuracy
        if (accuracyfunction == 0) {
            factory = new F1MeasureCalculator.Factory();
            isf1score = true;
        }
        if (accuracyfunction == 1) {
            factory = new BalancedAccuracyCalculator.Factory();
            isbascore = true;
        }
        if (accuracyfunction == 2) {
            factory = new AccuracyCalculator.Factory();
            isacscore = true;
        }

        // Punish long expressions
        if (punishlongexpression) {
            factory = new LengthBasedRefinementScorer.Factory(factory);
        }

        // XXX (Optional) avoid choosing solutions that work only for a single example
        if (AvoidPickySolutionsDecorator) {
            factory = new AvoidingPickySolutionsDecorator.Factory(factory);
        }

        boolean useCache = true;

        try (SparqlBasedSuggestor suggestor = SparqlBasedSuggestor.create(endpoint, logic, useCache)) {
            suggestor.addToClassBlackList(OWL2.NamedIndividual.getURI());
            suggestor.addToPropertyBlackList(RDF.type.getURI());

            boolean printLogs = false;

            // recursive: find cluster by prototype itself
            PruneCEL cel = null;
            if (recursive) {
                //cel = new RecursivePruneCEL(suggestor, logic, factory, suggestor);
                cel = new SimpleRecursivePruneCEL(suggestor, logic, factory, suggestor);
            } else {
                cel = new PruneCEL(suggestor, logic, factory);

            }

            // XXX Max iterations of the refinement
            cel.setMaxIterations(iteration);

            // XXX Maximum time (in ms)
            cel.setMaxTime(time);

            // XXX (Optional) try to avoid refining expressions that have not been created
            // in a promising way (i.e., just added a class to an existing expression
            // without changing the accuracy of the expression)
            if (setSkipNonImprovingStmts) {
                cel.setSkipNonImprovingStmts(true);
            }

            // XXX Keep this commented for now
            // {cel.activateRecursiveIteration(suggestor, 1.0, 0.5);

            // XXX Choose the learning problem (as JSON file)
            JSONLearningProblemReader reader = new JSONLearningProblemReader();
            Collection<LearningProblem> problems;
            if (folds == 1) {
                problems = reader.readProblems(tfjson);


                // Check if saveplace exist
                File file = new File(saveplace);
                File parentDir = file.getParentFile();

                if (parentDir != null && !parentDir.exists()) {
                    if (parentDir.mkdirs()) {
                        System.out.println("Directory created: " + parentDir.getAbsolutePath());
                    } else {
                        System.err.println("Failed to create directory: " + parentDir.getAbsolutePath());
                        return;
                    }
                }


                try (PrintStream pout = new PrintStream(saveplace)) {
//                for (int i = 0; i < names.size(); ++i) {
                    for (LearningProblem problem : problems) {
                        if (printLogs) {
                            try (OutputStream logStream = new BufferedOutputStream(
                                new FileOutputStream(problem.getName() + ".log"));
                                 CSVIntermediateResultPrinter irp = new CSVIntermediateResultPrinter(
                                     new PrintStream(problem.getName() + ".csv"))) {
                                runSearch(problem.getName(), problem.getPositiveExamples(), problem.getNegativeExamples(),
                                    cel, pout, logStream, irp, isbascore, isf1score, isacscore, folds);
                            }
                        } else {
                            runSearch(problem.getName(), problem.getPositiveExamples(), problem.getNegativeExamples(), cel,
                                pout, null, null, isbascore, isf1score, isacscore, folds);
                        }
                    }
                    StringBuilder statistics = new StringBuilder();
                    statistics.append("description_logic:").append(description_logic).append(", IsF1Score:")
                        .append(isf1score).append(", IsBascore:").append(isbascore).append(", IsAcscore:")
                        .append(isacscore).append(", IsPunishLongExpression:").append(punishlongexpression)
                        .append(", IsAvoidPickySolutionsDecorator:").append(AvoidPickySolutionsDecorator)
                        .append(", Isrecursive:").append(recursive).append(", IssetSkipNonImprovingStmts:")
                        .append(setSkipNonImprovingStmts).append(", usecluster:").append(cluster);
                    pout.println(statistics.toString());

                    if (cluster) {

                        Collection<LearningProblem> allproblems = reader
                            .readProblems("/home/quannian/Tentris_Graph/QALD10/TandF/ALL_TandF/MST5/TandF_MST5.json");
                        for (LearningProblem problem : allproblems) {
                            SelectionScores scores = suggestor.scoreExpression(currentClassExpression,
                                problem.getPositiveExamples(), problem.getNegativeExamples());
                            ScoredClassExpression a = factory.create(275, 119).score(currentClassExpression,
                                scores.getPosCount(), scores.getNegCount(), false);
                            StringBuilder row = new StringBuilder();
                            row.append("Result").append(",").append("test").append(",").append("test").append(",")
                                .append(a.getClassificationScore()).append(",").append(a.getRefinementScore())
                                .append(",").append(a.getPosCount()).append(",").append(a.getNegCount()).append(",")
                                .append("275").append(",").append("119").append(",").append(a.getClassExpression());
                            pout.println(row.toString());
                        }
                    }
                }
            } else {
                List<String> fileNames = listFiles("./././././" + Fold_train_test_save_path + "/Training");
                Collections.sort(fileNames);

                // Check if saveplace exist
                File file = new File(saveplace);
                File parentDir = file.getParentFile();

                if (parentDir != null && !parentDir.exists()) {
                    if (parentDir.mkdirs()) {
                        System.out.println("Directory created: " + parentDir.getAbsolutePath());
                    } else {
                        System.err.println("Failed to create directory: " + parentDir.getAbsolutePath());
                        return;
                    }
                }

                try (PrintStream pout = new PrintStream(saveplace)) {
                    for (String trainfileName : fileNames) {
                        String testfileName = trainfileName.replace("Train", "Test");
                        problems = reader.readProblems(trainfileName);
                        for (LearningProblem problem : problems) {
                            if (printLogs) {
                                try (OutputStream logStream = new BufferedOutputStream(
                                    new FileOutputStream(problem.getName() + ".log"));
                                     CSVIntermediateResultPrinter irp = new CSVIntermediateResultPrinter(
                                         new PrintStream(problem.getName() + ".csv"))) {
                                    runSearch(problem.getName(), problem.getPositiveExamples(), problem.getNegativeExamples(),
                                        cel, pout, logStream, irp, isbascore, isf1score, isacscore, folds);
                                }
                            } else {
                                runSearch(problem.getName(), problem.getPositiveExamples(), problem.getNegativeExamples(), cel,
                                    pout, null, null, isbascore, isf1score, isacscore, folds);
                            }
                        }
                        StringBuilder statistics = new StringBuilder();
                        statistics.append("description_logic:").append(description_logic).append(", IsF1Score:")
                            .append(isf1score).append(", IsBascore:").append(isbascore).append(", IsAcscore:")
                            .append(isacscore).append(", IsPunishLongExpression:").append(punishlongexpression)
                            .append(", IsAvoidPickySolutionsDecorator:").append(AvoidPickySolutionsDecorator)
                            .append(", Isrecursive:").append(recursive).append(", IssetSkipNonImprovingStmts:")
                            .append(setSkipNonImprovingStmts).append(", usecluster:").append(cluster);
                        pout.println(statistics.toString());

                        if (cluster) {
                            Collection<LearningProblem> allproblems = reader
                                .readProblems("/home/quannian/Tentris_Graph/QALD10/TandF/ALL_TandF/MST5/TandF_MST5.json");
                            for (LearningProblem problem : allproblems) {
                                SelectionScores scores = suggestor.scoreExpression(currentClassExpression,
                                    problem.getPositiveExamples(), problem.getNegativeExamples());
                                ScoredClassExpression a = factory.create(275, 119).score(currentClassExpression,
                                    scores.getPosCount(), scores.getNegCount(), false);
                                StringBuilder row = new StringBuilder();
                                row.append("Result").append(",").append("test").append(",").append("test").append(",")
                                    .append(a.getClassificationScore()).append(",").append(a.getRefinementScore())
                                    .append(",").append(a.getPosCount()).append(",").append(a.getNegCount()).append(",")
                                    .append("275").append(",").append("119").append(",").append(a.getClassExpression());
                                pout.println(row.toString());
                            }
                        }

                        if (folds != 1) {
                            pout.println("TESTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT");
                            Collection<LearningProblem> allproblems = reader
                                .readProblems(testfileName);
                            for (LearningProblem problem : allproblems) {
                                int numberofpositive = problem.getPositiveExamples().size();
                                int numberofnegative = problem.getNegativeExamples().size();
                                String name = problem.getName();
                                SelectionScores scores = suggestor.scoreExpression(BestClassExpression,
                                    problem.getPositiveExamples(), problem.getNegativeExamples());
                                ScoredClassExpression a = factory.create(numberofpositive, numberofnegative).score(BestClassExpression,
                                    scores.getPosCount(), scores.getNegCount(), false);
                                StringBuilder row = new StringBuilder();
                                row.append("Result").append(",").append(name).append(",").append("test").append(",")
                                    .append(a.getClassificationScore()).append(",").append(a.getRefinementScore())
                                    .append(",").append(a.getPosCount()).append(",").append(a.getNegCount()).append(",")
                                    .append(String.valueOf(numberofpositive)).append(",").append(String.valueOf(numberofnegative))
                                    .append(",").append(a.getClassExpression()).append("\n").append("\n");
                                pout.println(row.toString());
                            }
                        }
                    }
                }
            }
        }
        isHeaderPrinted = false;
        currentClassExpression = null;
        BestClassExpression = null;
    }

    public static void runSearch(String name, List<String> positive, List<String> negative, PruneCEL cel,
            PrintStream pout, OutputStream logStream, IntermediateResultPrinter iResultPrinter, boolean isbascore,
            boolean isf1score, boolean isacscore, int folds) throws IOException {
        System.out.println("Starting " + name);
        long time = System.currentTimeMillis();
        List<ScoredClassExpression> expressions = cel.findClassExpression(positive, negative, logStream,
                iResultPrinter);
        time = System.currentTimeMillis() - time;
        String number_of_positive = Integer.toString(positive.size());
        String number_of_negative = Integer.toString(negative.size());

        // For Folds
        if (folds!=1){

            BestClassExpression = expressions.get(0).getClassExpression();

        }

        // for calculating clustering
        if (currentClassExpression == null)
            currentClassExpression = expressions.get(0).getClassExpression();
        else
            currentClassExpression = new Junction(false, currentClassExpression,
                    expressions.get(0).getClassExpression());

        saveClassExpressionsToCsv(expressions, name, time, pout, number_of_positive, number_of_negative, isbascore,
                isf1score, isacscore);
        // printClassExpressions(expressions, name, time, pout);
    }


    public static void printClassExpressions(List<ScoredClassExpression> expressions, String name, long runtime,
            PrintStream pout) {
        pout.print("Result");
        if (name != null) {
            pout.print(" for ");
            pout.print(name);
        }
        pout.print(" after ");
        pout.print(runtime);
        pout.println("ms:");
        for (ScoredClassExpression exp : expressions) {
            pout.println(exp.toString());
        }
    }

    public static void saveClassExpressionsToCsv(List<ScoredClassExpression> expressions, String name, long runtime,
            PrintStream pout, String number_of_positive, String number_of_negative, boolean isbascore,
            boolean isf1score, boolean isacscore) {
        // Print the header only if it hasn't been printed yet
        if (!isHeaderPrinted && isf1score) {
            pout.println(
                    "Result,Name,Runtime,F1-score,R-score,PosCount,NegCount,Number-of-Pos,Number-of-Neg,Expressions");
            isHeaderPrinted = true; // Set the flag to true after printing
        }
        if (!isHeaderPrinted && isbascore) {
            pout.println(
                    "Result,Name,Runtime,Balanced-accuracy-score,R-score,PosCount,NegCount,Number-of-Pos,Number-of-Neg,Expressions");
            isHeaderPrinted = true; // Set the flag to true after printing
        }
        if (!isHeaderPrinted && isacscore) {
            pout.println(
                    "Result,Name,Runtime,Accuracy-score,R-score,PosCount,NegCount,Number-of-Pos,Number-of-Neg,Expressions");
            isHeaderPrinted = true; // Set the flag to true after printing
        }

        // Create a CSV row for the results
        StringBuilder row = new StringBuilder();
        row.append("Result");

        // Append the name, or leave it blank if null
        if (name != null) {
            row.append(",").append(name);
        } else {
            row.append(","); // Placeholder for null name
        }

        // Append the runtime
        row.append(",").append(runtime).append("ms,");
        // Append Cscore
        String expressionsList = expressions.isEmpty() ? "" : expressions.get(0).toString();
        int cScoreStart = expressionsList.indexOf("cScore=") + "cScore=".length();
        int cScoreEnd = expressionsList.indexOf(",", cScoreStart);
        String cScore = expressionsList.substring(cScoreStart, cScoreEnd);

        int rScoreStart = expressionsList.indexOf("rScore=") + "rScore=".length();
        int rScoreEnd = expressionsList.indexOf(",", rScoreStart);
        String rScore = expressionsList.substring(rScoreStart, rScoreEnd);

        row.append(cScore).append(",").append(rScore).append(",");

        // Append positive and negative
        int posCountStart = expressionsList.indexOf("posCount=") + "posCount=".length();
        int posCountEnd = expressionsList.indexOf(",", posCountStart);
        String posCount = expressionsList.substring(posCountStart, posCountEnd);

        int negCountStart = expressionsList.indexOf("negCount=") + "negCount=".length();
        int negCountEnd = expressionsList.indexOf("]", negCountStart);
        String negCount = expressionsList.substring(negCountStart, negCountEnd);

        row.append(posCount).append(",").append(negCount).append(",");

        // Append positive and negative number from the cluster
        row.append(number_of_positive).append(",").append(number_of_negative).append(",");

        // Get the first expression only, if the list is not empty
        String expressionsString = expressions.isEmpty() ? "" : "\"" + expressions.toString() + "\"";

        // Append the first expression to the row
        row.append(expressionsString);

        // Print the CSV row
        pout.println(row.toString());
    }

    public static List<List<String>> separateIntoGroups(List<String> list, int numGroups) {
        // Shuffle the list to randomize the order
        Collections.shuffle(list);

        // Create a list of groups
        List<List<String>> groups = new ArrayList<>();

        // Initialize the groups
        for (int i = 0; i < numGroups; i++) {
            groups.add(new ArrayList<>());
        }

        // Distribute the elements into the groups
        for (int i = 0; i < list.size(); i++) {
            groups.get(i % numGroups).add(list.get(i));
        }

        return groups;
    }

    public static void saveJsonFileWithProblemsWrapper(String name, List<String> positiveExamples, List<String> negativeExamples, String filePath) {
        // Step 1: Create the main JSON structure with "problems" as a top-level key
        Map<String, Object> jsonData = new HashMap<>();
        Map<String, Object> problemsData = new HashMap<>(); // This will hold the individual problem entries

        // Step 2: Loop over each entry and add it to the "problems" structure

        Map<String, List<String>> examplesData = new HashMap<>();
        examplesData.put("positive_examples", positiveExamples);
        examplesData.put("negative_examples", negativeExamples);

        problemsData.put(name, examplesData);

        // Add the "problems" data to the main JSON data
        jsonData.put("problems", problemsData);

        // Step 3: Convert the object to JSON using Gson
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        String jsonString = gson.toJson(jsonData);

        File file = new File(filePath);

        // Ensure that the parent directories exist
        File parentDir = file.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            boolean dirsCreated = parentDir.mkdirs(); // Create directories if they don't exist
            if (!dirsCreated) {
                System.err.println("Failed to create directories: " + parentDir.getAbsolutePath());
                return;
            }
        }

        // Step 5: Write the JSON data to the file
        try (FileWriter writer = new FileWriter(file)) {
            writer.write(jsonString);
            System.out.println("JSON file saved at: " + filePath);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    public static List<String> listFiles(String directoryPath) throws IOException {
        List<String> fileNames = new ArrayList<>();

        // Use Files.walkFileTree to walk the directory tree and collect file names
        Files.walkFileTree(Paths.get(directoryPath), new SimpleFileVisitor<Path>() {
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                // Add each file name to the list
                fileNames.add(file.toString());
                return FileVisitResult.CONTINUE;
            }
        });

        return fileNames;
    }
    public static void creat_dir_for_fold(int folds, String tfjson, String fold_train_test_save_path) throws IOException {

        List<List<String>> random_positive_groups;
        List<List<String>> random_negative_groups;

        JSONLearningProblemReader reader = new JSONLearningProblemReader();
        Collection<LearningProblem> problems = reader.readProblems(tfjson);
        for (LearningProblem problem : problems){
            List<String> list_positive = problem.getPositiveExamples();
            List<String> list_negative = problem.getNegativeExamples();
            String name = problem.getName();

            random_positive_groups = separateIntoGroups(list_positive, folds);
            random_negative_groups = separateIntoGroups(list_negative, folds);

            for (int i = 0; i < folds; i++) {
                List<String> testing_positiveGroup = random_positive_groups.get(i);
                List<String> testing_negativeGroup = random_negative_groups.get(i);

                List<String> training_positiveGroup = new ArrayList<>();
                List<String> training_negativeGroup = new ArrayList<>();

                // Combine all other groups except the current 'i' group
                for (int j = 0; j < folds; j++) {
                    if (j != i) {
                        training_positiveGroup.addAll(random_positive_groups.get(j));
                        training_negativeGroup.addAll(random_negative_groups.get(j));
                    }
                }
                saveJsonFileWithProblemsWrapper(name, testing_positiveGroup, testing_negativeGroup, "./././././"+fold_train_test_save_path+"/Testing/"+name+"Test_Fold_" + i + ".json");
                saveJsonFileWithProblemsWrapper(name, training_positiveGroup, training_negativeGroup, "././././"+fold_train_test_save_path+"/Training/"+name+"Train_Fold_" + i + ".json");

            }
        }
    }
}
