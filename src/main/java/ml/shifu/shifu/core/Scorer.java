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

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.concurrent.Callable;

import ml.shifu.shifu.column.NSColumn;
import ml.shifu.shifu.container.ScoreObject;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.dtrain.dataset.BasicFloatNetwork;
import ml.shifu.shifu.core.dtrain.nn.NNConstants;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;

import org.apache.commons.collections.CollectionUtils;
import org.encog.ml.BasicML;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.svm.SVM;
import org.encog.neural.networks.BasicNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Scorer, calculate the score for a specify input
 */
public class Scorer {

    private static Logger log = LoggerFactory.getLogger(Scorer.class);
    public static final int DEFAULT_SCORE_SCALE = 1000;

    private String alg;

    private List<BasicML> models;
    private List<ColumnConfig> columnConfigList;
    private double cutoff = 4.0d;
    private ModelConfig modelConfig;
    private int scale = DEFAULT_SCORE_SCALE;

    /**
     * No any variables set to finalSelect=true, we should take all candidate variables as inputs.
     */
    private boolean noVarSelect = false;

    /**
     * For faster query from categorical bins
     */
    private Map<Integer, Map<String, Integer>> binCategoryMap = new HashMap<Integer, Map<String, Integer>>();

    /**
     * For neural network, if output the hidden neurons
     */
    private int outputHiddenLayerIndex = 0;

    public Scorer(List<BasicML> models, List<ColumnConfig> columnConfigList, String algorithm, ModelConfig modelConfig) {
        this(models, columnConfigList, algorithm, modelConfig, 4.0d);
    }

    public Scorer(List<BasicML> models, List<ColumnConfig> columnConfigList, String algorithm, ModelConfig modelConfig,
            Double cutoff) {
        this(models, columnConfigList, algorithm, modelConfig, cutoff, 0);
    }

    public Scorer(List<BasicML> models, List<ColumnConfig> columnConfigList, String algorithm, ModelConfig modelConfig,
            Double cutoff, int outputHiddenLayerIndex) {
        if(modelConfig == null) {
            throw new IllegalArgumentException("modelConfig should not be null");
        }

        this.models = models;

        this.columnConfigList = columnConfigList;
        this.cutoff = cutoff;
        this.alg = algorithm;
        this.modelConfig = modelConfig;

        if(this.columnConfigList != null) {
            int[] inputOutputIndex = DTrainUtils.getInputOutputCandidateCounts(modelConfig.getNormalizeType(),
                    this.columnConfigList);
            int inputNodeCount = inputOutputIndex[0] == 0 ? inputOutputIndex[2] : inputOutputIndex[0];
            int candidateCount = inputOutputIndex[2];
            this.noVarSelect = (inputNodeCount == candidateCount);
        }

        // compute binCategoryMap for all algorithm while only be used in
        if(this.columnConfigList != null) {
            for(ColumnConfig columnConfig: this.columnConfigList) {
                if(columnConfig.isCategorical()) {
                    Map<String, Integer> map = new HashMap<String, Integer>();
                    List<String> categories = columnConfig.getBinCategory();
                    if(categories != null) {
                        for(int i = 0; i < categories.size(); i++) {
                            String categoricalVal = categories.get(i);
                            if(categoricalVal == null) {
                                map.put("", i);
                            } else {
                                List<String> cvals = CommonUtils.flattenCatValGrp(categoricalVal);
                                for(String cval: cvals) {
                                    map.put(cval, i);
                                }
                            }
                            map.put(categories.get(i) == null ? "" : categories.get(i), i);
                        }
                    }
                    this.binCategoryMap.put(columnConfig.getColumnNum(), map);
                }
            }
        }

        this.outputHiddenLayerIndex = outputHiddenLayerIndex;
    }

    public ScoreObject score(Map<String, String> rawDataMap) {
        return scoreNsData(CommonUtils.convertRawMapToNsDataMap(rawDataMap));
    }

    /**
     * Run model against raw NSColumn Data map to get score
     * 
     * @param rawDataNsMap
     *            - raw NSColumn Data map
     * @return ScoreObject - model score
     */
    public ScoreObject scoreNsData(Map<NSColumn, String> rawDataNsMap) {
        return scoreNsData(null, rawDataNsMap);
    }

    public ScoreObject score(final MLDataPair pair, Map<String, String> rawDataMap) {
        return scoreNsData(pair, CommonUtils.convertRawMapToNsDataMap(rawDataMap));
    }

    public ScoreObject scoreNsData(MLDataPair inputPair, Map<NSColumn, String> rawNsDataMap) {
        if(inputPair == null && !this.alg.equalsIgnoreCase(NNConstants.NN_ALG_NAME)) {
            inputPair = CommonUtils.assembleNsDataPair(binCategoryMap, noVarSelect, modelConfig, columnConfigList,
                    rawNsDataMap, cutoff, alg);
        }

        final MLDataPair pair = inputPair;
        List<MLData> modelResults = new ArrayList<MLData>();
        for(final BasicML model: models) {
            // TODO, check if no need 'if' condition and refactor two if for loops please
            if(model instanceof BasicFloatNetwork || model instanceof NNModel) {
                final BasicFloatNetwork network = (model instanceof BasicFloatNetwork) ? (BasicFloatNetwork) model
                        : ((NNModel) model).getIndependentNNModel().getBasicNetworks().get(0);

                final MLDataPair networkPair = CommonUtils.assembleNsDataPair(binCategoryMap, noVarSelect, modelConfig,
                        columnConfigList, rawNsDataMap, cutoff, alg, network.getFeatureSet());

                /*
                 * if(network.getFeatureSet().size() != networkPair.getInput().size()) {
                 * log.error("Network and input size mismatch: Network Size = " + network.getFeatureSet().size()
                 * + "; Input Size = " + networkPair.getInput().size());
                 * continue;
                 * }
                 */
                log.info("Network input count = {}, while input size = {}", network.getInputCount(), networkPair
                        .getInput().size());

                final int fnlOutputHiddenLayerIndex = outputHiddenLayerIndex;
                modelResults.add(new Callable<MLData>() {
                    @Override
                    public MLData call() {
                        MLData finalOutput = network.compute(networkPair.getInput());

                        if(fnlOutputHiddenLayerIndex == 0) {
                            return finalOutput;
                        }

                        // append output values in hidden layer
                        double[] hiddenOutputs = network.getLayerOutput(fnlOutputHiddenLayerIndex);
                        double[] outputs = new double[finalOutput.getData().length + hiddenOutputs.length];

                        System.arraycopy(finalOutput.getData(), 0, outputs, 0, finalOutput.getData().length);
                        System.arraycopy(hiddenOutputs, 0, outputs, finalOutput.getData().length, hiddenOutputs.length);
                        return new BasicMLData(outputs);
                    }
                }.call());
            } else if(model instanceof BasicNetwork) {
                final BasicNetwork network = (BasicNetwork) model;

                final MLDataPair networkPair = CommonUtils.assembleNsDataPair(binCategoryMap, noVarSelect, modelConfig,
                        columnConfigList, rawNsDataMap, cutoff, alg, null);
                modelResults.add(new Callable<MLData>() {
                    @Override
                    public MLData call() {
                        return network.compute(networkPair.getInput());
                    }
                }.call());
            } else if(model instanceof SVM) {
                final SVM svm = (SVM) model;
                if(svm.getInputCount() != pair.getInput().size()) {
                    log.error("SVM and input size mismatch: SVM Size = " + svm.getInputCount() + "; Input Size = "
                            + pair.getInput().size());
                    continue;
                }
                modelResults.add(new Callable<MLData>() {
                    @Override
                    public MLData call() {
                        return svm.compute(pair.getInput());
                    }
                }.call());
            } else if(model instanceof LR) {
                final LR lr = (LR) model;
                if(lr.getInputCount() != pair.getInput().size()) {
                    log.error("LR and input size mismatch: LR Size = " + lr.getInputCount() + "; Input Size = "
                            + pair.getInput().size());
                    continue;
                }
                modelResults.add(new Callable<MLData>() {
                    @Override
                    public MLData call() {
                        return lr.compute(pair.getInput());
                    }
                }.call());
            } else if(model instanceof TreeModel) {
                final TreeModel tm = (TreeModel) model;
                if(tm.getInputCount() != pair.getInput().size()) {
                    throw new RuntimeException("GBDT and input size mismatch: tm input Size = " + tm.getInputCount()
                            + "; data input Size = " + pair.getInput().size());
                }
                modelResults.add(new Callable<MLData>() {
                    @Override
                    public MLData call() {
                        MLData result = tm.compute(pair.getInput());
                        return result;
                    }
                }.call());
            } else {
                throw new RuntimeException("unsupport models");
            }
        }

        List<Double> scores = new ArrayList<Double>();
        List<Integer> rfTreeSizeList = new ArrayList<Integer>();
        SortedMap<String, Double> hiddenOutputs = null;

        if(CollectionUtils.isNotEmpty(modelResults)) {
            if(modelResults.size() != this.models.size()) {
                log.error("Get model results size doesn't match with models size.");
                return null;
            }

            if(this.outputHiddenLayerIndex != 0) {
                hiddenOutputs = new TreeMap<String, Double>(new Comparator<String>() {

                    @Override
                    public int compare(String o1, String o2) {
                        String[] split1 = o1.split("_");
                        String[] split2 = o2.split("_");
                        int model1Index = Integer.parseInt(split1[1]);
                        int model2Index = Integer.parseInt(split2[1]);
                        if(model1Index > model2Index) {
                            return 1;
                        } else if(model1Index < model2Index) {
                            return -1;
                        } else {
                            int hidden1Index = Integer.parseInt(split1[2]);
                            int hidden2Index = Integer.parseInt(split2[2]);
                            if(hidden1Index > hidden2Index) {
                                return 1;
                            } else if(hidden1Index < hidden2Index) {
                                return -1;
                            } else {
                                int hidden11Index = Integer.parseInt(split1[3]);
                                int hidden22Index = Integer.parseInt(split2[3]);
                                return Integer.valueOf(hidden11Index).compareTo(Integer.valueOf(hidden22Index));
                            }
                        }
                    }
                });
            }
            for(int i = 0; i < this.models.size(); i++) {
                BasicML model = this.models.get(i);
                MLData score = modelResults.get(i);

                if(model instanceof BasicNetwork || model instanceof NNModel) {
                    if(modelConfig != null && modelConfig.isRegression()) {
                        scores.add(toScore(score.getData(0)));
                        if(this.outputHiddenLayerIndex != 0) {
                            for(int j = 1; j < score.getData().length; j++) {
                                hiddenOutputs.put("model_" + i + "_" + this.outputHiddenLayerIndex + "_" + (j - 1),
                                        score.getData()[j]);
                            }
                        }
                    } else if(modelConfig != null && modelConfig.isClassification()
                            && modelConfig.getTrain().isOneVsAll()) {
                        // if one vs all classification
                        scores.add(toScore(score.getData(0)));
                    } else {
                        double[] outputs = score.getData();
                        for(double d: outputs) {
                            scores.add(toScore(d));
                        }
                    }
                } else if(model instanceof SVM) {
                    scores.add(toScore(score.getData(0)));
                } else if(model instanceof LR) {
                    scores.add(toScore(score.getData(0)));
                } else if(model instanceof TreeModel) {
                    if(modelConfig.isClassification() && !modelConfig.getTrain().isOneVsAll()) {
                        double[] scoreArray = score.getData();
                        for(double sc: scoreArray) {
                            scores.add(sc);
                        }
                    } else {
                        // if one vs all multiple classification or regression
                        scores.add(toScore(score.getData(0)));
                    }
                    final TreeModel tm = (TreeModel) model;
                    // regression for RF
                    if(!tm.isClassfication() && !tm.isGBDT()) {
                        rfTreeSizeList.add(tm.getTrees().size());
                    }
                } else {
                    throw new RuntimeException("unsupport models");
                }
            }
        }

        Integer tag = Constants.DEFAULT_IDEAL_VALUE;

        if(scores.size() == 0) {
            log.warn("No Scores Calculated...");
        }

        return new ScoreObject(scores, tag, rfTreeSizeList, hiddenOutputs);
    }

    private double toScore(Double d) {
        return d * scale;
    }

    public int getModelCnt() {
        return ((models != null) ? this.models.size() : 0);
    }

    public int getScale() {
        return scale;
    }

    public void setScale(int scale) {
        if(scale > 0) {
            this.scale = scale;
        }
    }
}