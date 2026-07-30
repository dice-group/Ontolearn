package org.example.cel.io;

import java.util.List;

public class LearningProblem {

    private List<String> positiveExamples;
    private List<String> negativeExamples;
    private String name;

    public LearningProblem(List<String> positiveExamples, List<String> negativeExamples, String name) {
        super();
        this.positiveExamples = positiveExamples;
        this.negativeExamples = negativeExamples;
        this.name = name;
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + ((name == null) ? 0 : name.hashCode());
        result = prime * result + ((negativeExamples == null) ? 0 : negativeExamples.hashCode());
        result = prime * result + ((positiveExamples == null) ? 0 : positiveExamples.hashCode());
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        LearningProblem other = (LearningProblem) obj;
        if (name == null) {
            if (other.name != null)
                return false;
        } else if (!name.equals(other.name))
            return false;
        if (negativeExamples == null) {
            if (other.negativeExamples != null)
                return false;
        } else if (!negativeExamples.equals(other.negativeExamples))
            return false;
        if (positiveExamples == null) {
            if (other.positiveExamples != null)
                return false;
        } else if (!positiveExamples.equals(other.positiveExamples))
            return false;
        return true;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("LearningProblem [positiveExamples=");
        builder.append(positiveExamples);
        builder.append(", negativeExamples=");
        builder.append(negativeExamples);
        builder.append(", name=");
        builder.append(name);
        builder.append("]");
        return builder.toString();
    }

    /**
     * @return the positiveExamples
     */
    public List<String> getPositiveExamples() {
        return positiveExamples;
    }

    /**
     * @param positiveExamples the positiveExamples to set
     */
    public void setPositiveExamples(List<String> positiveExamples) {
        this.positiveExamples = positiveExamples;
    }

    /**
     * @return the negativeExamples
     */
    public List<String> getNegativeExamples() {
        return negativeExamples;
    }

    /**
     * @param negativeExamples the negativeExamples to set
     */
    public void setNegativeExamples(List<String> negativeExamples) {
        this.negativeExamples = negativeExamples;
    }

    /**
     * @return the name
     */
    public String getName() {
        return name;
    }

    /**
     * @param name the name to set
     */
    public void setName(String name) {
        this.name = name;
    }

}
