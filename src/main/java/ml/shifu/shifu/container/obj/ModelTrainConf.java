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
import ml.shifu.shifu.core.alg.SVMTrainer;
import ml.shifu.shifu.core.dtrain.CommonConstants;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

/**
 * {@link ModelTrainConf} is train part in ModelConfig.json.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ModelTrainConf {

    /**
     * Different training algorithms supported in Shifu. SVM actuall is not implemented well. DT is replaced by RF and
     * GBT. TF_DNN is used for tensorflow dnn training, TENSORFLOW is used for generic tensorflow model evaluation.
     * 
     * @author Zhang David (pengzhang@paypal.com)
     */
    public static enum ALGORITHM {
        NN, LR, SVM, DT, RF, GBT,TF_DNN, TENSORFLOW
    }

    /**
     * Multiple classification algorithm. NATIVE is supported in NN/RF. ONEVSALL/ONEVSREST is by enabling multiple
     * regerssion running.
     * 
     * @author Zhang David (pengzhang@paypal.com)
     */
    @JsonDeserialize(using = MultipleClassificationDeserializer.class)
    public static enum MultipleClassification {
        NATIVE, // means using NN regression or RF classification, not one vs all or one vs one
        ONEVSALL, ONEVSREST, // the same as ONEVSALL
        ONEVSONE; // ONEVSONE is not impl yet.
        /*
         * Get {@link MultipleClassification} by string, case can be ignored.
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

    /**
     * How many bagging jobs in training.
     */
    private Integer baggingNum = Integer.valueOf(1);

    /**
     * Bagging sampling with replacement, this is only works well in NN. In RF, bagging sampling with replacement is
     * enabled no matter true or false. In GBT, bagging sampling with replacement is disabled no matter true or false
     */
    private Boolean baggingWithReplacement = Boolean.FALSE;

    /**
     * In each bagging job to do sampling according to this sample rate.
     */
    private Double baggingSampleRate = Double.valueOf(1.0);

    /**
     * After bagging sampling, current rate of records is used to do validation.
     */
    private Double validSetRate = Double.valueOf(0.2);

    /**
     * Only sample negative records out, this works with {@link #baggingSampleRate}.
     */
    private Boolean sampleNegOnly = Boolean.FALSE;

    /**
     * If training is converged. 0 means not enabled early stop feature.
     */
    private Double convergenceThreshold = Double.valueOf(0.0);

    /**
     * Iterations used in training.
     */
    private Integer numTrainEpochs = Integer.valueOf(100);

    /**
     * For NN only, how many epochs training in one iteration.
     */
    private Integer epochsPerIteration = Integer.valueOf(1);

    /**
     * Train data located on disk or not, this parameter is deprecated because of in NN/LR MemoryDiskList is used if not
     * enough memory, disk will be automatically used. In GBDT/RF, because of data with prediction is changed in each
     * tree, only memory list is supported.
     */
    @Deprecated
    private Boolean trainOnDisk = Boolean.FALSE;

    /**
     * If enabled by true, training data and validation data will be fixed in training even another job is started.
     */
    private Boolean fixInitInput = Boolean.FALSE;

    /**
     * Only works in regression, if enabled by true, both positive and negative records will be sampled independent.
     */
    private Boolean stratifiedSample = Boolean.FALSE;

    /**
     * If continue model training based on existing model in model path, this is like warm-start in scikit-learn.
     */
    private Boolean isContinuous = Boolean.FALSE;

    /**
     * Only works in NN and do swapping training, validation data in differnent epochs.
     */
    private Boolean isCrossOver = Boolean.FALSE;

    /**
     * How many threads in each worker, this will enable multiple threading running in workers.
     */
    private Integer workerThreadCount = 4;

    /**
     * If enabled by a value in (1 - 20], cross validation will be enabled. Jobs will be started to train according to
     * k-fold training data. Final average validation error will be printed in console.
     */
    private Integer numKFold = -1;

    /**
     * Random sample seed is used to generate Random instance when sampling.
     * It's a hidden feature support in shifu. If user not configure this value, shifu will will fallback to generate
     * random for bagging each time.
     */
    @JsonIgnore
    private Long baggingSampleSeed = CommonConstants.NOT_CONFIGURED_BAGGING_SEED;

    /**
     * Up sampling for positive tags, this is to solve class imbalance.
     */
    private Double upSampleWeight = Double.valueOf(1d);

    /**
     * Algorithm: LR, NN, RF, GBT, TF-DNN
     */
    private String algorithm = "NN";

    /**
     * Model params for training like learning rate, tree depth ...
     */
    private Map<String, Object> params;
    
    /**
     * Grid search params config file path.
     */
    private String gridConfigFile = null;
    
    /**
     * Grid search params in config file.
     * Read from {@link #gridConfigFile} after loading {@link ModelConfig} from JSON file.
     */
    @JsonIgnore
    private List<String> gridConfigFileContent = null;

    /**
     * Multiple classification method: NATIVE or ONEVSALL(ONEVSREST)
     */
    private MultipleClassification multiClassifyMethod = MultipleClassification.NATIVE;

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

    @JsonIgnore
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

    @JsonIgnore
    public String getGridConfigFile() {
        return gridConfigFile;
    }

    @JsonProperty
    public void setGridConfigFile(String gridConfigFile) {
        this.gridConfigFile = gridConfigFile;
    }

    public List<String> getGridConfigFileContent() {
        return gridConfigFileContent;
    }

    public void setGridConfigFileContent(List<String> gridConfigFileContent) {
        this.gridConfigFileContent = gridConfigFileContent;
    }

    public Map<String, String> getCustomPaths() {
        return customPaths;
    }

    public void setCustomPaths(Map<String, String> customPaths) {
        this.customPaths = customPaths;
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
     * @return the baggingSampleSeed
     */
    public Long getBaggingSampleSeed() {
        return baggingSampleSeed;
    }

    /**
     * @param baggingSampleSeed
     *              the baggingSampleSeed to set
     */
    public void setBaggingSampleSeed(Long baggingSampleSeed) {
        this.baggingSampleSeed = baggingSampleSeed;
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
                || this.multiClassifyMethod == MultipleClassification.ONEVSREST;
    }

    /**
     * @return the sampleNegOnly
     */
    @JsonIgnore
    public Boolean getSampleNegOnly() {
        return sampleNegOnly;
    }

    /**
     * @param sampleNegOnly
     *            the sampleNegOnly to set
     */
    @JsonProperty
    public void setSampleNegOnly(Boolean sampleNegOnly) {
        this.sampleNegOnly = sampleNegOnly;
    }

    /**
     * @return the stratifiedSample
     */
    @JsonIgnore
    public Boolean getStratifiedSample() {
        return stratifiedSample;
    }

    /**
     * @param stratifiedSample
     *            the stratifiedSampling to set
     */
    @JsonProperty
    public void setStratifiedSample(Boolean stratifiedSample) {
        this.stratifiedSample = stratifiedSample;
    }

    /**
     * @return the numKFold
     */
    @JsonIgnore
    public Integer getNumKFold() {
        return numKFold;
    }

    /**
     * @param numKFold
     *            the numKFold to set
     */
    @JsonProperty
    public void setNumKFold(Integer numKFold) {
        this.numKFold = numKFold;
    }

    @Override
    public boolean equals(Object obj) {
        if(obj == null || !(obj instanceof ModelTrainConf)) {
            return false;
        }

        ModelTrainConf other = (ModelTrainConf) obj;
        if(this == other) {
            return true;
        }

        return this.algorithm.equals(other.getAlgorithm()) && this.baggingNum.equals(other.getBaggingNum())
                && this.getNumTrainEpochs().equals(other.getNumTrainEpochs())
                && this.validSetRate.equals(other.getValidSetRate());
    }

    @Override
    public ModelTrainConf clone() {
        ModelTrainConf other = new ModelTrainConf();
        other.setAlgorithm(algorithm);
        other.setBaggingNum(baggingNum);
        other.setBaggingSampleRate(baggingSampleRate);
        other.setBaggingSampleSeed(baggingSampleSeed);
        other.setConvergenceThreshold(convergenceThreshold);
        if(customPaths != null) {
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

    public static Map<String, Object> createParamsByAlg(ALGORITHM alg, ModelTrainConf trainConf) {
        Map<String, Object> params = new HashMap<String, Object>();

        if(ALGORITHM.NN.equals(alg)) {
            params.put(CommonConstants.PROPAGATION, "R");
            params.put(CommonConstants.LEARNING_RATE, 0.1);
            params.put(CommonConstants.NUM_HIDDEN_LAYERS, 1);

            List<Integer> nodes = new ArrayList<Integer>();
            nodes.add(50);
            params.put(CommonConstants.NUM_HIDDEN_NODES, nodes);

            List<String> func = new ArrayList<String>();
            func.add("tanh");
            params.put(CommonConstants.ACTIVATION_FUNC, func);
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
        } else if(ALGORITHM.GBT.equals(alg)) {
            params.put("TreeNum", "100");
            params.put("FeatureSubsetStrategy", "TWOTHIRDS");
            params.put("MaxDepth", 7);
            params.put("MinInstancesPerNode", 5);
            params.put("MinInfoGain", 0.0);
            params.put("DropoutRate", 0.0);
            params.put("Impurity", "variance");
            params.put(CommonConstants.LEARNING_RATE, 0.05);
            params.put("Loss", "squared");
        } else if(ALGORITHM.LR.equals(alg)) {
            params.put(LogisticRegressionTrainer.LEARNING_RATE, 0.1);
            params.put("RegularizedConstant", 0.0);
            params.put("L1orL2", "NONE");
        } else if(ALGORITHM.TF_DNN.equals(alg)) {
            params.put(CommonConstants.LEARNING_RATE, 0.1);
            params.put(CommonConstants.NUM_HIDDEN_LAYERS, 1);

            List<Integer> nodes = new ArrayList<Integer>();
            nodes.add(50);
            params.put(CommonConstants.NUM_HIDDEN_NODES, nodes);

            List<String> func = new ArrayList<String>();
            func.add("relu");
            params.put(CommonConstants.ACTIVATION_FUNC, func);
            params.put(CommonConstants.TF_OPTIMIZER, "Adam");
            params.put(CommonConstants.TF_LOSS, "entropy");

        }
        return params;
    }

}
