package ml.shifu.core.container;

import com.fasterxml.jackson.annotation.JsonIgnore;

import java.util.HashMap;
import java.util.Map;

public class ClassificationResult {

    private String trueClass;
    private Double normalizedTrueClass;
    private Map<String, Double> scoreMap = null;
    private Map<String, String> supplementaryMap = null;
    private Double weight = 1.0;


    public Double getNormalizedTrueClass() {
        return normalizedTrueClass;
    }

    public void setNormalizedTrueClass(Double normalizedTrueClass) {
        this.normalizedTrueClass = normalizedTrueClass;
    }

    public Double getWeight() {
        return weight;
    }

    public void setWeight(Double weight) {
        this.weight = weight;
    }


    @JsonIgnore
    public void putScore(String modelName, Double score) {
        if (scoreMap == null) {
            scoreMap = new HashMap<String, Double>();

        }
        scoreMap.put(modelName, score);
    }

    @JsonIgnore
    public void putSupplementary(String fieldName, Object supplementary) {
        if (supplementaryMap == null) {
            supplementaryMap = new HashMap<String, String>();
        }
        supplementaryMap.put(fieldName, supplementary.toString());
    }

    @JsonIgnore
    public Double getMeanScore() {
        Double sum = 0.0;
        for (Double score : scoreMap.values()) {
            sum += score;
        }
        return sum / scoreMap.size();
    }

    public Map<String, String> getSupplementaryMap() {
        return supplementaryMap;
    }

    public void setSupplementaryMap(Map<String, String> supplementaryMap) {
        this.supplementaryMap = supplementaryMap;
    }


    public String getTrueClass() {
        return trueClass;
    }

    public void setTrueClass(String trueClass) {
        this.trueClass = trueClass;
    }

    public Map<String, Double> getScoreMap() {
        return scoreMap;
    }

    public void setScoreMap(Map<String, Double> scoreMap) {
        this.scoreMap = scoreMap;
    }


}
