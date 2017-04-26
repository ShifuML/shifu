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
package ml.shifu.shifu.core;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ml.shifu.shifu.container.obj.ColumnConfig;

import ml.shifu.shifu.util.CommonUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Reasoner, it helps to find the the majority contributor to the model
 */
public class Reasoner {
    private static Logger log = LoggerFactory.getLogger(Reasoner.class);

    private Integer numTopVariables = 5;
    private List<ScoreDiffObject> sdList;
    private List<String> reasons;
    private Map<String, String> reasonCodeMap;

    public Reasoner(Map<String, String> reasonCodeMap) {
        this.reasonCodeMap = reasonCodeMap;
    }

    public void calculateReasonCodes(List<ColumnConfig> columnConfigList, Map<String, String> rawDataMap) {
        if (columnConfigList == null || columnConfigList.size() == 0) {
            throw new RuntimeException("ColumnConfig is empty.");
        }

        sdList = new ArrayList<ScoreDiffObject>();
        reasons = new ArrayList<String>();

        for (ColumnConfig config : columnConfigList) {
            if (!config.isFinalSelect()) {
                continue;
            }

            String key = config.getColumnName();
            if (!rawDataMap.containsKey(key)) {
                log.error("Variable Missing in Test Data: " + key);
                continue;
            }

            // add this, for user may not run post-process and he/she don't want to have reason code
            // make it common. Just skip this column
            if (config.getBinAvgScore() == null) {
                // log.info("No bin average score for " + config.getColumnName());
                continue;
            }

            ScoreDiffObject sd = new ScoreDiffObject();

            int binLength = config.getBinLength();
            Integer binNum = binLength;
            if (config.isNumerical()) {
                if (rawDataMap.get(key).equals("")) {
                    sd.varValue = config.getMean();
                } else {
                    sd.varValue = Double.parseDouble(rawDataMap.get(key));
                }
                List<Double> binBoundary = config.getBinBoundary();

                while ((--binNum) >= 0) {
                    if (sd.varValue >= binBoundary.get(binNum)) {
                        break;
                    }
                }
                if (binNum == -1) {
                    log.info(sd.varValue.toString());
                    log.info(binBoundary.toString());
                    break;
                }

                sd.binBoundary = config.getBinBoundary();
                sd.binNum = binNum;
            } else if (config.isCategorical()) {
                List<String> binCategory = config.getBinCategory();
                sd.varCategory = rawDataMap.get(key);

                while ((--binNum) >= 0) {
                    if (CommonUtils.isCategoricalBinValue(binCategory.get(binNum), sd.varCategory)) {
                    // if (sd.varCategory.equals(binCategory.get(binNum))) {
                        break;
                    }
                }
                if (binNum == -1) {
                    log.info("Unknown value.");
                    break;
                }
                sd.binCategory = config.getBinCategory();
                sd.binNum = binNum;
            }

            sd.columnName = config.getColumnName();
            sd.columnNum = config.getColumnNum();
            sd.scoreDiff = config.getBinAvgScore().get(binNum);
            sd.binAvgScore = config.getBinAvgScore();
            sd.binCountNeg = config.getBinCountNeg();
            sd.binCountPos = config.getBinCountPos();

            sdList.add(sd);
        }

        Collections.sort(sdList, new ScoreDiffComparator());
        String reason = null;
        int n = numTopVariables;
        if (n > sdList.size()) {
            n = sdList.size();
        }

        for (int i = 0; i < n; i++) {
            log.debug(sdList.get(i).columnName + "==>" + sdList.get(i).scoreDiff);
            reason = reasonCodeMap.get(sdList.get(i).columnName);
            if (!reasons.contains(reason)) {
                reasons.add(reason);
            }
        }
    }

    public List<String> getReasonCodes() {
        return reasons;
    }

    public Map<String, Object> getReasonDetails() {
        Map<String, Object> map = new HashMap<String, Object>();
        map.put("details", sdList.subList(0, numTopVariables));
        map.put("reasons", reasons);

        return map;
    }

    static class ScoreDiffObject {
        public String columnName;
        public Integer columnNum;
        public Integer binNum;
        public Integer scoreDiff;
        public Double varValue;
        public String varCategory;
        public List<Double> binBoundary;
        public List<String> binCategory;
        public List<Integer> binAvgScore;
        public List<Integer> binCountPos;
        public List<Integer> binCountNeg;
    }

    public void setNumTopVariables(Integer numTopVariables) {
        this.numTopVariables = numTopVariables;
    }

    static class ScoreDiffComparator implements Comparator<ScoreDiffObject>, Serializable {
        
        private static final long serialVersionUID = 652346402551215269L;

        public int compare(ScoreDiffObject a, ScoreDiffObject b) {
            if (!a.scoreDiff.equals(b.scoreDiff)) {
                return b.scoreDiff.compareTo(a.scoreDiff);
            } else {
                return a.columnNum.compareTo(b.columnNum);
            }
        }
    }
}
