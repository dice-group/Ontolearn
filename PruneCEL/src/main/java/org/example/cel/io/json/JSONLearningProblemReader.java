package org.example.cel.io.json;

import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.example.cel.io.LearningProblem;

import com.google.gson.stream.JsonReader;

/**
 * A simple class used to read learning problems from a JSON file. The file
 * should have the following structure:
 * 
 * <pre>
 * {
 *  "problems": {
 *  "problem-name-1": {
 *    "positive_examples": [
 *      "http://example.org/positive-1",
 *      "http://example.org/positive-2", 
 *    ... ],
 *    "negative_examples": [
 *      "http://example.org/negative-1",
 *      "http://example.org/negative-2", 
 *    ... ]
 *  },
 *  "problem-name-2": {
 *    ...
 *  }}}
 * </pre>
 * 
 * @author Michael R&ouml;der (michael.roeder@uni-paderborn.de)
 *
 */
public class JSONLearningProblemReader {

    public List<LearningProblem> readProblems(String file) throws IOException {
        try (FileReader fReader = new FileReader(file); JsonReader reader = new JsonReader(fReader);) {
            return readProblems(reader);
        }
    }

    public List<LearningProblem> readProblems(JsonReader reader) throws IOException {
        reader.beginObject();
        String key;
        while (reader.hasNext()) {
            key = reader.nextName();
            if ("problems".equals(key)) {
                return readProblemsArray(reader);
            }
        }
        reader.endObject();
        return null;
    }

    public List<LearningProblem> readProblemsArray(JsonReader reader) throws IOException {
        List<LearningProblem> problems = new ArrayList<>();
        reader.beginObject();
        while (reader.hasNext()) {
            problems.add(readProblem(reader));
        }
        reader.endObject();
        return problems;
    }

    public LearningProblem readProblem(JsonReader reader) throws IOException {
        String name = reader.nextName();
        List<String> positives = null;
        List<String> negatives = null;
        reader.beginObject();
        String key;
        while (reader.hasNext()) {
            key = reader.nextName();
            if ("positive_examples".equals(key)) {
                positives = readStringArray(reader);
            } else if ("negative_examples".equals(key)) {
                negatives = readStringArray(reader);
            }

        }
        reader.endObject();
        return new LearningProblem(positives, negatives, name);
    }

    public List<String> readStringArray(JsonReader reader) throws IOException {
        List<String> strings = new ArrayList<>();
        reader.beginArray();
        while (reader.hasNext()) {
            strings.add(reader.nextString());
        }
        reader.endArray();
        return strings;
    }

}
