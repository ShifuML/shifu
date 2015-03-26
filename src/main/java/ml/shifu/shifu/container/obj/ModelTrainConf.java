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
package ml.shifu.shifu.container.obj;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import ml.shifu.shifu.core.alg.LogisticRegressionTrainer;
import ml.shifu.shifu.core.alg.NNTrainer;
import ml.shifu.shifu.core.alg.SVMTrainer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * ModelTrainConf class
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ModelTrainConf {

    public static enum ALGORITHM {
        NN, LR, SVM, DT
    }

    private Integer baggingNum = Integer.valueOf(5);
    // change it to false by default, as we often don't use this way.
    private Boolean baggingWithReplacement = Boolean.FALSE;
    private Double baggingSampleRate = Double.valueOf(0.8);
    private Double validSetRate = Double.valueOf(0.2);
    private Integer numTrainEpochs = Integer.valueOf(100);
    private Integer epochsPerIteration = Integer.valueOf(1);

    private Boolean trainOnDisk = Boolean.FALSE;
    private Boolean fixInitInput = Boolean.FALSE;
    
    private Boolean isContinuous = Boolean.FALSE; 
    
    private Boolean isCrossOver = Boolean.FALSE;

    private String algorithm = "NN";

    private Map<String, Object> params;

    private Map<String, String> customPaths;

    public ModelTrainConf() {
        customPaths = new HashMap<String, String>(1);

        /**
         * Since most user won't use this function,
         * hidden the custom paths for creating new model.
         */
        /*
         * customPaths.put(Constants.KEY_PRE_TRAIN_STATS_PATH, null);
         * customPaths.put(Constants.KEY_SELECTED_RAW_DATA_PATH, null);
         * customPaths.put(Constants.KEY_NORMALIZED_DATA_PATH, null);
         * customPaths.put(Constants.KEY_TRAIN_SCORES_PATH, null);
         * customPaths.put(Constants.KEY_BIN_AVG_SCORE_PATH, null);
         */
    }

    public Integer getBaggingNum() {
        return baggingNum;
    }

    public void setBaggingNum(Integer baggingNum) {
        this.baggingNum = baggingNum;
    }

    public Boolean getBaggingWithReplacement() {
        return baggingWithReplacement;
    }

    public void setBaggingWithReplacement(Boolean baggingWithReplacement) {
        this.baggingWithReplacement = baggingWithReplacement;
    }

    public Double getBaggingSampleRate() {
        return baggingSampleRate;
    }

    public void setBaggingSampleRate(Double baggingSampleRate) {
        this.baggingSampleRate = baggingSampleRate;
    }

    public Double getValidSetRate() {
        return validSetRate;
    }

    public void setValidSetRate(Double validSetRate) {
        this.validSetRate = validSetRate;
    }

    public Boolean getTrainOnDisk() {
        return trainOnDisk;
    }

    public void setTrainOnDisk(Boolean trainOnDisk) {
        this.trainOnDisk = trainOnDisk;
    }

    @JsonIgnore
    public Boolean getFixInitInput() {
        return fixInitInput;
    }

    public void setFixInitInput(Boolean fixInitInput) {
        this.fixInitInput = fixInitInput;
    }

    public Integer getNumTrainEpochs() {
        return numTrainEpochs;
    }

    public void setNumTrainEpochs(Integer numTrainEpochs) {
        this.numTrainEpochs = numTrainEpochs;
    }

    public String getAlgorithm() {
        return algorithm;
    }

    public void setAlgorithm(String algorithm) {
        this.algorithm = algorithm;
    }

    public Map<String, Object> getParams() {
        return params;
    }

    public void setParams(Map<String, Object> params) {
        this.params = params;
    }

    public Map<String, String> getCustomPaths() {
        return customPaths;
    }

    public void setCustomPaths(Map<String, String> customPaths) {
        this.customPaths = customPaths;
    }

    /**
     * @param alg
     * @return
     */
    public static Map<String, Object> createParamsByAlg(ALGORITHM alg) {
        Map<String, Object> params = new HashMap<String, Object>();

        if(ALGORITHM.NN.equals(alg)) {
            params.put(NNTrainer.PROPAGATION, "Q");
            params.put(NNTrainer.LEARNING_RATE, 0.1);
            params.put("LearningDecay", 0.0);
            params.put(NNTrainer.NUM_HIDDEN_LAYERS, 2);

            List<Integer> nodes = new ArrayList<Integer>();
            nodes.add(30);
            nodes.add(20);
            params.put(NNTrainer.NUM_HIDDEN_NODES, nodes);

            List<String> func = new ArrayList<String>();
            func.add("sigmoid");
            func.add("sigmoid");
            params.put(NNTrainer.ACTIVATION_FUNC, func);
        } else if(ALGORITHM.SVM.equals(alg)) {
            params.put(SVMTrainer.SVM_KERNEL, "linear");
            params.put(SVMTrainer.SVM_GAMMA, 1.0);
            params.put(SVMTrainer.SVM_CONST, 1.0);
        } else if(ALGORITHM.DT.equals(alg)) {
            // To be decide
            // DecisionTreeTrainer
        } else if(ALGORITHM.LR.equals(alg)) {
            params.put(LogisticRegressionTrainer.LEARNING_RATE, 0.1);
        }

        return params;
    }

    /**
     * @return the epochsPerIteration
     */
    public Integer getEpochsPerIteration() {
        return epochsPerIteration;
    }

    /**
     * @param epochsPerIteration
     *            the epochsPerIteration to set
     */
    public void setEpochsPerIteration(Integer epochsPerIteration) {
        this.epochsPerIteration = epochsPerIteration;
    }

    /**
     * @return the isCrossOver
     */
    public Boolean getIsCrossOver() {
        return isCrossOver;
    }

    /**
     * @param isCrossOver the isCrossOver to set
     */
    public void setIsCrossOver(Boolean isCrossOver) {
        this.isCrossOver = isCrossOver;
    }

    /**
     * @return the isContinuous
     */
    public Boolean getIsContinuous() {
        return isContinuous;
    }

    /**
     * @param isContinuous the isContinuous to set
     */
    public void setIsContinuous(Boolean isContinuous) {
        this.isContinuous = isContinuous;
    }

}
