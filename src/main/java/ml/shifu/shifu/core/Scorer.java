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
package ml.shifu.shifu.core;

import ml.shifu.shifu.container.ScoreObject;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.util.CommonUtils;
import org.encog.ml.BasicML;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.svm.SVM;
import org.encog.neural.networks.BasicNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Scorer, calculate the score for a specify input
 */
public class Scorer {

    private static Logger log = LoggerFactory.getLogger(Scorer.class);

    @SuppressWarnings("unused")
    private String alg;

    private List<BasicML> models;
    private List<ColumnConfig> columnConfigList;
    private double cutoff = 4.0d;

    // private boolean verbose = false;

    public Scorer(List<BasicML> models, List<ColumnConfig> columnConfigList, String algorithm) {
        this(models, columnConfigList, algorithm, 4.0d);
    }

    public Scorer(List<BasicML> models, List<ColumnConfig> columnConfigList, String algorithm, Double cutoff) {
        this.models = models;
        this.columnConfigList = columnConfigList;
        this.cutoff = cutoff;
        this.alg = algorithm;
    }

    public ScoreObject score(Map<String, String> rawDataMap) {
        MLDataPair pair = CommonUtils.assembleDataPair(columnConfigList, rawDataMap, cutoff);
        return score(pair, rawDataMap);
    }

    public ScoreObject score(MLDataPair pair, Map<String, String> rawDataMap) {
        if (pair == null) {
            return null;
        }

        List<Integer> scores = new ArrayList<Integer>();

        for (BasicML model : models) {
            if (model instanceof BasicNetwork) {
                BasicNetwork network = (BasicNetwork) model;
                if (network.getInputCount() != pair.getInput().size()) {
                    log.error("Network and input size mismatch: Network Size = " + network.getInputCount() + "; Input Size = " + pair.getInput().size());
                    continue;
                }
                MLData score = network.compute(pair.getInput());
                scores.add(toScore(score.getData(0)));
            } else if (model instanceof SVM) {
                SVM svm = (SVM) model;
                if (svm.getInputCount() != pair.getInput().size()) {
                    log.error("SVM and input size mismatch: SVM Size = " + svm.getInputCount() + "; Input Size = " + pair.getInput().size());
                    continue;
                }
                MLData score = svm.compute(pair.getInput());
                scores.add(toScore(score.getData(0)));
            } else {
                throw new RuntimeException("unspport models");
            }
        }

        Integer tag = (int) pair.getIdeal().getData(0);

        if (scores.size() == 0) {
            log.error("No Scores Calculated...");
            return null;
        }

        return new ScoreObject(scores, tag);
    }

    private Integer toScore(Double d) {
        return (int) Math.round(d * 1000);
    }
}
