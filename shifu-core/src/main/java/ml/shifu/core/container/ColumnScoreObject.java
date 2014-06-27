/**
 * Copyright [2012-2014] eBay Software Foundation
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
package ml.shifu.core.container;

import java.util.List;

/**
 * ColumnScoreObject class
 */
public class ColumnScoreObject {

    private int columnNum;
    private String columnVal;
    private List<Integer> scores;
    private int maxScore;
    private int minScore;
    private int avgScore;

    public ColumnScoreObject(int columnNum, String columnVal) {
        super();
        this.columnNum = columnNum;
        this.columnVal = columnVal;
    }

    public int getColumnNum() {
        return columnNum;
    }

    public void setColumnNum(int columnNum) {
        this.columnNum = columnNum;
    }

    public String getColumnVal() {
        return columnVal;
    }

    public void setColumnVal(String columnVal) {
        this.columnVal = columnVal;
    }

    public List<Integer> getScores() {
        return scores;
    }

    public void setScores(List<Integer> scores) {
        this.scores = scores;
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

    public int getAvgScore() {
        return avgScore;
    }

    public void setAvgScore(int avgScore) {
        this.avgScore = avgScore;
    }

}
