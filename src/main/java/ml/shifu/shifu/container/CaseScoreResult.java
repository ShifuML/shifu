/*
 * Copyright [2012-2014] PayPal Software Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.container;

import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;

/**
 * CaseScoreResult class
 */
public class CaseScoreResult {
    private String inputData;

    private List<Double> scores;
    private double maxScore;
    private double minScore;
    private double avgScore;
    private double medianScore;

    private Map<String, CaseScoreResult> subModelScores;

    private SortedMap<String, Double> hiddenLayerScores;

    public CaseScoreResult() {
        super();
    }

    public List<Double> getScores() {
        return scores;
    }

    public void setScores(List<Double> scores) {
        this.scores = scores;
    }

    public double getMaxScore() {
        return maxScore;
    }

    public void setMaxScore(double maxScore) {
        this.maxScore = maxScore;
    }

    public double getMinScore() {
        return minScore;
    }

    public void setMinScore(double minScore) {
        this.minScore = minScore;
    }

    public double getAvgScore() {
        return avgScore;
    }

    public void setAvgScore(double avgScore) {
        this.avgScore = avgScore;
    }

    /**
     * @return the medianScore
     */
    public double getMedianScore() {
        return medianScore;
    }

    /**
     * @param medianScore
     *            the medianScore to set
     */
    public void setMedianScore(double medianScore) {
        this.medianScore = medianScore;
    }

    public String getInputData() {
        return inputData;
    }

    public void setInputData(String inputData) {
        this.inputData = inputData;
    }

    public void addSubModelScore(String modelName, ScoreObject so) {
        if(this.subModelScores == null) {
            this.subModelScores = new TreeMap<String, CaseScoreResult>();
        }

        CaseScoreResult scoreResult = new CaseScoreResult();
        scoreResult.setScores(so.getScores());
        scoreResult.setMaxScore(so.getMaxScore());
        scoreResult.setMinScore(so.getMinScore());
        scoreResult.setAvgScore(so.getMeanScore());
        scoreResult.setMedianScore(so.getMedianScore());

        this.subModelScores.put(modelName, scoreResult);
    }

    public Map<String, CaseScoreResult> getSubModelScores() {
        return subModelScores;
    }

    /**
     * @return the hiddenLayerScores
     */
    public SortedMap<String, Double> getHiddenLayerScores() {
        return hiddenLayerScores;
    }

    /**
     * @param hiddenLayerScores
     *            the hiddenLayerScores to set
     */
    public void setHiddenLayerScores(SortedMap<String, Double> hiddenLayerScores) {
        this.hiddenLayerScores = hiddenLayerScores;
    }
}
