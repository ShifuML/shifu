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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.SortedMap;

import org.apache.commons.collections.CollectionUtils;

/**
 * Score object
 */
public class ScoreObject {

    // Inputs
    private List<Double> scores;
    private Object tag;

    // Derived
    private double meanScore = 0;
    private double maxScore = -Integer.MAX_VALUE;
    private double minScore = Integer.MAX_VALUE;
    private double medianScore = 0;

    private SortedMap<String, Double> hiddenLayerScores;

    public ScoreObject(List<Double> scores, Object tag) {
        this(scores, tag, new ArrayList<Integer>());
    }

    public ScoreObject(List<Double> scores, Object tag, List<Integer> modelSizeList) {
        this(scores, tag, modelSizeList, null);
    }

    public ScoreObject(List<Double> scores, Object tag, List<Integer> modelSizeList,
            SortedMap<String, Double> hiddenLayerScores) {
        this.scores = scores;
        this.tag = tag;

        int modelSizeSum = 0;
        for(int i = 0; i < scores.size(); i++) {
            Double score = scores.get(i);
            // by default model size is 1 per each model
            int modelSize = 1;
            if(modelSizeList.size() > 0) {
                modelSize = modelSizeList.get(i);
            }
            modelSizeSum += modelSize;
            meanScore += score * modelSize;
            maxScore = Math.max(maxScore, score);
            minScore = Math.min(minScore, score);
        }

        List<Double> tmpScoreList = new ArrayList<Double>(scores);
        Collections.sort(tmpScoreList);

        if(CollectionUtils.isNotEmpty(tmpScoreList)) {
            meanScore /= modelSizeSum;
            medianScore = tmpScoreList.get(scores.size() / 2);
        }

        this.hiddenLayerScores = hiddenLayerScores;
    }

    public List<Double> getScores() {
        return scores;
    }

    public void setScores(List<Double> scores) {
        this.scores = scores;
    }

    public Object getTag() {
        return tag;
    }

    public void setTag(Object tag) {
        this.tag = tag;
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

    public double getMedianScore() {
        return medianScore;
    }

    public void setMedianScore(double medianScore) {
        this.medianScore = medianScore;
    }

    public double getMeanScore() {
        return meanScore;
    }

    public void setMeanScore(double meanScore) {
        this.meanScore = meanScore;
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
