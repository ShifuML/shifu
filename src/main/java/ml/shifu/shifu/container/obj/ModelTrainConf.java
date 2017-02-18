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
package ml.shifu.shifu.container.obj;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ml.shifu.shifu.core.alg.LogisticRegressionTrainer;
import ml.shifu.shifu.core.alg.NNTrainer;
import ml.shifu.shifu.core.alg.SVMTrainer;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

/**
 * ModelTrainConf class
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ModelTrainConf {

    public static enum ALGORITHM {
        NN, LR, SVM, DT, RF, GBT
    }

    @JsonDeserialize(using = MultipleClassificationDeserializer.class)
    public static enum MultipleClassification {
        NATIVE, // means using NN regression or RF classification, not one vs all or one vs one
        ONEVSALL, ONVVSREST, // the same as ONEVSALL
        ONVVSONE; // ONEVSONE is not impl yet.

        /**
         * Get {@link MultipleClassification} by string, case can be ignored.
         * @param strategy - name for multi-classification
         * @return {@link MultipleClassification}
         */
        public static MultipleClassification of(String strategy) {
            for(MultipleClassification element: values()) {
                if(element.toString().equalsIgnoreCase(strategy)) {
                    return element;
                }
            }
            throw new IllegalArgumentException("cannot find such enum in MULTIPLE_CLASSIFICATION");
        }
    }

    private Integer baggingNum = Integer.valueOf(5);
    // this is set default as true as bagging often with replacement sampleing.
    private Boolean baggingWithReplacement = Boolean.TRUE;
    private Double baggingSampleRate = Double.valueOf(1.0);
    private Double validSetRate = Double.valueOf(0.2);
    private Double convergenceThreshold = Double.valueOf(0.0);
    private Integer numTrainEpochs = Integer.valueOf(100);
    private Integer epochsPerIteration = Integer.valueOf(1);
    
    /**
     * Only sample negative records out
     */
    private Boolean sampleNegOnly = Boolean.FALSE;

    private Boolean trainOnDisk = Boolean.FALSE;
    private Boolean fixInitInput = Boolean.FALSE;
    private Boolean stratifiedSample = Boolean.FALSE;


    private Boolean isContinuous = Boolean.FALSE;

    private Boolean isCrossOver = Boolean.FALSE;

    private Integer workerThreadCount = 4;
    
    private Integer numKFold = -1;

    private Double upSampleWeight = Double.valueOf(1d);

    private String algorithm = "NN";

    private Map<String, Object> params;

    private Map<String, String> customPaths;

    private MultipleClassification multiClassifyMethod = MultipleClassification.NATIVE;

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
     * @param alg - algorithm to build model
     * @param trainConf - configuration for training model
     * @return parameters for model training
     */
    public static Map<String, Object> createParamsByAlg(ALGORITHM alg, ModelTrainConf trainConf) {
        Map<String, Object> params = new HashMap<String, Object>();

        if(ALGORITHM.NN.equals(alg)) {
            params.put(NNTrainer.PROPAGATION, "R");
            params.put(NNTrainer.LEARNING_RATE, 0.1);
            params.put(NNTrainer.NUM_HIDDEN_LAYERS, 1);

            List<Integer> nodes = new ArrayList<Integer>();
            nodes.add(50);
            params.put(NNTrainer.NUM_HIDDEN_NODES, nodes);

            List<String> func = new ArrayList<String>();
            func.add("tanh");
            params.put(NNTrainer.ACTIVATION_FUNC, func);
            params.put("RegularizedConstant", 0.0);
        } else if(ALGORITHM.SVM.equals(alg)) {
            params.put(SVMTrainer.SVM_KERNEL, "linear");
            params.put(SVMTrainer.SVM_GAMMA, 1.0);
            params.put(SVMTrainer.SVM_CONST, 1.0);
        } else if(ALGORITHM.RF.equals(alg)) {
            params.put("TreeNum", "10");
            params.put("FeatureSubsetStrategy", "TWOTHIRDS");
            params.put("MaxDepth", 10);
            params.put("MinInstancesPerNode", 1);
            params.put("MinInfoGain", 0.0);
            params.put("Impurity", "variance");
            params.put("Loss", "squared");
            trainConf.setNumTrainEpochs(1000);
        } else if(ALGORITHM.GBT.equals(alg)) {
            params.put("TreeNum", "100");
            params.put("FeatureSubsetStrategy", "TWOTHIRDS");
            params.put("MaxDepth", 7);
            params.put("MinInstancesPerNode", 5);
            params.put("MinInfoGain", 0.0);
            params.put("Impurity", "variance");
            params.put(NNTrainer.LEARNING_RATE, 0.05);
            params.put("Loss", "squared");
            trainConf.setNumTrainEpochs(1000);
        } else if(ALGORITHM.LR.equals(alg)) {
            params.put(LogisticRegressionTrainer.LEARNING_RATE, 0.1);
            params.put("RegularizedConstant", 0.0);
            params.put("L1orL2", "NONE");
        }
        return params;
    }

    /**
     * @return the epochsPerIteration
     */
    @JsonIgnore
    public Integer getEpochsPerIteration() {
        return epochsPerIteration;
    }

    /**
     * @param epochsPerIteration
     *            the epochsPerIteration to set
     */
    @JsonProperty
    public void setEpochsPerIteration(Integer epochsPerIteration) {
        this.epochsPerIteration = epochsPerIteration;
    }

    /**
     * As threshold is an optional setting, Use @{@link JsonIgnore} to ignore threshold when initially write
     * out to ModelConfig.json.
     * 
     * @return Convergence threshold.
     */
    @JsonIgnore
    public Double getConvergenceThreshold() {
        return convergenceThreshold;
    }

    @JsonProperty
    public void setConvergenceThreshold(Double convergenceThreshold) {
        this.convergenceThreshold = convergenceThreshold;
    }

    @JsonIgnore
    public Boolean getIsCrossOver() {
        return isCrossOver;
    }

    /**
     * @param isCrossOver
     *            the isCrossOver to set
     */
    @JsonProperty
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
     * @param isContinuous
     *            the isContinuous to set
     */
    public void setIsContinuous(Boolean isContinuous) {
        this.isContinuous = isContinuous;
    }

    /**
     * @return the workerThreadCount
     */
    public Integer getWorkerThreadCount() {
        return workerThreadCount;
    }

    /**
     * @param workerThreadCount
     *            the workerThreadCount to set
     */
    public void setWorkerThreadCount(Integer workerThreadCount) {
        this.workerThreadCount = workerThreadCount;
    }

    /**
     * @return the upSampleWeight
     */
    @JsonIgnore
    public Double getUpSampleWeight() {
        return upSampleWeight;
    }

    /**
     * @param upSampleWeight
     *            the upSampleWeight to set
     */
    public void setUpSampleWeight(Double upSampleWeight) {
        this.upSampleWeight = upSampleWeight;
    }

    /**
     * @return the multiClassifyMethod
     */
    @JsonIgnore
    public MultipleClassification getMultiClassifyMethod() {
        return multiClassifyMethod;
    }

    /**
     * @param multiClassifyMethod
     *            the multiClassifyMethod to set
     */
    @JsonProperty
    public void setMultiClassifyMethod(MultipleClassification multiClassifyMethod) {
        this.multiClassifyMethod = multiClassifyMethod;
    }

    @JsonIgnore
    public boolean isOneVsAll() {
        return this.multiClassifyMethod == MultipleClassification.ONEVSALL
                || this.multiClassifyMethod == MultipleClassification.ONVVSREST;
    }

    @Override
    public boolean equals(Object obj) {
        if ( obj == null || !(obj instanceof ModelTrainConf) ) {
            return false;
        }

        ModelTrainConf other = (ModelTrainConf) obj;
        if ( this == other ) {
            return true;
        }

        return this.algorithm.equals(other.getAlgorithm())
                && this.baggingNum.equals(other.getBaggingNum())
                && this.getNumTrainEpochs().equals(other.getNumTrainEpochs())
                && this.validSetRate.equals(other.getValidSetRate());
    }

    /**
     * @return the sampleNegOnly
     */
    @JsonIgnore
    public Boolean getSampleNegOnly() {
        return sampleNegOnly;
    }

    /**
     * @param sampleNegOnly the sampleNegOnly to set
     */
    @JsonProperty
    public void setSampleNegOnly(Boolean sampleNegOnly) {
        this.sampleNegOnly = sampleNegOnly;
    }

    @Override
    public ModelTrainConf clone() {
        ModelTrainConf other = new ModelTrainConf();
        other.setAlgorithm(algorithm);
        other.setBaggingNum(baggingNum);
        other.setBaggingSampleRate(baggingSampleRate);
        other.setConvergenceThreshold(convergenceThreshold);
        if ( customPaths != null ) {
            other.setCustomPaths(new HashMap<String, String>(customPaths));
        }
        other.setEpochsPerIteration(epochsPerIteration);
        other.setFixInitInput(fixInitInput);
        other.setIsContinuous(isContinuous);
        other.setMultiClassifyMethod(multiClassifyMethod);
        other.setNumTrainEpochs(numTrainEpochs);
        other.setParams(new HashMap<String, Object>(params));
        other.setTrainOnDisk(trainOnDisk);
        other.setUpSampleWeight(upSampleWeight);
        other.setValidSetRate(validSetRate);
        other.setWorkerThreadCount(workerThreadCount);
        return other;
    }

    /**
     * @return the stratifiedSampling
     */
    @JsonIgnore
    public Boolean getStratifiedSample() {
        return stratifiedSample;
    }

    /**
     * @param stratifiedSample the stratifiedSampling to set
     */
    @JsonProperty
    public void setStratifiedSample(Boolean stratifiedSample) {
        this.stratifiedSample = stratifiedSample;
    }

    /**
     * @return the kCrossValidation
     */
    @JsonIgnore
    public Integer getNumKFold() {
        return numKFold;
    }

    /**
     * @param numKFold the kCrossValidation to set
     */
    @JsonProperty
    public void setNumKFold(Integer numKFold) {
        this.numKFold = numKFold;
    }
}
