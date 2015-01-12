/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.container;

import org.apache.commons.collections.CollectionUtils;

import java.util.*;
import java.util.Map.Entry;

/**
 * DtScore class
 */
public class DtScore {

    private List<Integer> scoreList;
    private int maxScore = Integer.MIN_VALUE;
    private int minScore = Integer.MAX_VALUE;
    private int meanScore = 0;
    private int medianScore = 0;

    private Map<Object, Integer> modelTargetVoteMap;

    public DtScore() {
        scoreList = new ArrayList<Integer>();
        modelTargetVoteMap = new HashMap<Object, Integer>();
    }

    public void addDtScoreEntry(int score, Object target) {
        scoreList.add(score);

        maxScore = Math.max(score, maxScore);
        minScore = Math.min(score, minScore);

        int targetVoteCnt = 1;
        if (modelTargetVoteMap.containsKey(target)) {
            targetVoteCnt += modelTargetVoteMap.get(target);
        }
        modelTargetVoteMap.put(target, targetVoteCnt);
    }

    public List<Integer> getScoreList() {
        return scoreList;
    }

    public void setScoreList(List<Integer> scoreList) {
        this.scoreList = scoreList;
    }

    public int getMaxScore() {
        return maxScore;
    }

    public void setMaxScore(int maxScore) {
        this.maxScore = maxScore;
    }

    public int getMinScore() {
        return minScore;
    }

    public void setMinScore(int minScore) {
        this.minScore = minScore;
    }

    public int getMeanScore() {
        if (CollectionUtils.isNotEmpty(scoreList)) {
            int totalScore = 0;
            for (int i = 0; i < scoreList.size(); i++) {
                totalScore += scoreList.get(i);
            }

            meanScore = totalScore / scoreList.size();
        }

        return meanScore;
    }

    public void setMeanScore(int meanScore) {
        this.meanScore = meanScore;
    }

    public int getMedianScore() {
        if (CollectionUtils.isNotEmpty(scoreList)) {
            List<Integer> tmpScoreList = new ArrayList<Integer>(scoreList);
            Collections.sort(tmpScoreList);
            medianScore = tmpScoreList.get(tmpScoreList.size() / 2);
        }

        return medianScore;
    }

    public void setMedianScore(int medianScore) {
        this.medianScore = medianScore;
    }

    public Object getMostFavoriateModelTarget() {
        if (modelTargetVoteMap == null || modelTargetVoteMap.size() == 0) {
            return null;
        }

        Object mostFavoriateTarget = null;
        int mostVoteCnt = Integer.MIN_VALUE;

        Iterator<Entry<Object, Integer>> iterator = modelTargetVoteMap.entrySet().iterator();
        while (iterator.hasNext()) {
            Entry<Object, Integer> entry = iterator.next();
            if (entry.getValue() > mostVoteCnt) {
                mostVoteCnt = entry.getValue();
                mostFavoriateTarget = entry.getKey();
            }
        }

        return mostFavoriateTarget;
    }
}
