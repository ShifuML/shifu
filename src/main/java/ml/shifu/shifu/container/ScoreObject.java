/**
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

import org.apache.commons.collections.CollectionUtils;

/**
 * Score object
 */
public class ScoreObject {

    // Inputs
    private List<Integer> scores;
    private Object tag;

    // Derived
    private Integer meanScore = 0;
    private Integer maxScore = -Integer.MAX_VALUE;
    private Integer minScore = Integer.MAX_VALUE;
    private Integer medianScore = 0;

    public ScoreObject(List<Integer> scores, Object tag) {
        this(scores, tag, new ArrayList<Integer>());
    }

    public ScoreObject(List<Integer> scores, Object tag, List<Integer> modelSizeList) {
        this.scores = scores;
        this.tag = tag;

        int modelSizeSum = 0;
        for(int i = 0; i < scores.size(); i++) {
            Integer score = scores.get(i);
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

        List<Integer> tmpScoreList = new ArrayList<Integer>(scores);
        Collections.sort(tmpScoreList);

        if(CollectionUtils.isNotEmpty(tmpScoreList)) {
            meanScore /= modelSizeSum;
            medianScore = tmpScoreList.get(scores.size() / 2);
        }
    }

    public List<Integer> getScores() {
        return scores;
    }

    public void setScores(List<Integer> scores) {
        this.scores = scores;
    }

    public Object getTag() {
        return tag;
    }

    public void setTag(Object tag) {
        this.tag = tag;
    }

    public Integer getMaxScore() {
        return maxScore;
    }

    public void setMaxScore(Integer maxScore) {
        this.maxScore = maxScore;
    }

    public Integer getMinScore() {
        return minScore;
    }

    public void setMinScore(Integer minScore) {
        this.minScore = minScore;
    }

    public Integer getMedianScore() {
        return medianScore;
    }

    public void setMedianScore(Integer medianScore) {
        this.medianScore = medianScore;
    }

    public Integer getMeanScore() {
        return meanScore;
    }

    public void setMeanScore(Integer meanScore) {
        this.meanScore = meanScore;
    }

}
