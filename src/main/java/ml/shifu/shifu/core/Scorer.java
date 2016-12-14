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
package ml.shifu.shifu.core;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import ml.shifu.shifu.container.ScoreObject;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.util.CommonUtils;

import org.encog.ml.BasicML;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.svm.SVM;
import org.encog.neural.networks.BasicNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Scorer, calculate the score for a specify input
 */
public class Scorer {

    private static Logger log = LoggerFactory.getLogger(Scorer.class);

    private String alg;

    private List<BasicML> models;
    private List<ColumnConfig> columnConfigList;
    private double cutoff = 4.0d;
    private ModelConfig modelConfig;

    /**
     * No any variables set to finalSelect=true, we should take all candidate variables as inputs.
     */
    private boolean noVarSelect = false;

    /**
     * For faster query from categorical bins
     */
    private Map<Integer, Map<String, Integer>> binCategoryMap = new HashMap<Integer, Map<String, Integer>>();

    /**
     * Run model in parallel. Size is # of models.
     */
    private ExecutorService threadPool;

    public Scorer(List<BasicML> models, List<ColumnConfig> columnConfigList, String algorithm, ModelConfig modelConfig) {
        this(models, columnConfigList, algorithm, modelConfig, 4.0d);
    }

    public Scorer(List<BasicML> models, List<ColumnConfig> columnConfigList, String algorithm, ModelConfig modelConfig,
            Double cutoff) {

        if(modelConfig == null) {
            throw new IllegalArgumentException("modelConfig should not be null");
        }

        this.models = models;
        this.columnConfigList = columnConfigList;
        this.cutoff = cutoff;
        this.alg = algorithm;
        this.modelConfig = modelConfig;

        if(this.columnConfigList != null) {
            int[] inputOutputIndex = DTrainUtils.getInputOutputCandidateCounts(this.columnConfigList);
            int inputNodeCount = inputOutputIndex[0] == 0 ? inputOutputIndex[2] : inputOutputIndex[0];
            int candidateCount = inputOutputIndex[2];
            if(inputNodeCount == candidateCount) {
                this.noVarSelect = true;
            } else {
                this.noVarSelect = false;
            }
        }
        if(CommonUtils.isDesicionTreeAlgorithm(alg)) {
            for(ColumnConfig columnConfig: columnConfigList) {
                if(columnConfig.isCategorical()) {
                    Map<String, Integer> map = new HashMap<String, Integer>();
                    List<String> categories = columnConfig.getBinCategory();
                    for(int i = 0; i < categories.size(); i++) {
                        map.put(categories.get(i) == null ? "" : categories.get(i), i);
                    }
                    this.binCategoryMap.put(columnConfig.getColumnNum(), map);
                }
            }
        }

        this.threadPool = Executors.newFixedThreadPool(Math.min(Runtime.getRuntime().availableProcessors(),
                models.size()));

        // add a shutdown hook as a safe guard if some one not call close
        Runtime.getRuntime().addShutdownHook(new Thread(new Runnable() {
            @Override
            public void run() {
                // shut down thread pool
                Scorer.this.threadPool.shutdownNow();
                try {
                    Scorer.this.threadPool.awaitTermination(2, TimeUnit.SECONDS);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        }));
    }

    /**
     * Cleaning the thread pool resources, must be called at last.
     */
    public void close() {
        // shut down thread pool
        this.threadPool.shutdownNow();
        try {
            this.threadPool.awaitTermination(2, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    public ScoreObject score(Map<String, String> rawDataMap) {
        MLDataPair pair = CommonUtils.assembleDataPair(binCategoryMap, noVarSelect, modelConfig, columnConfigList,
                rawDataMap, cutoff);
        return score(pair, rawDataMap);
    }

    public ScoreObject score(final MLDataPair pair, Map<String, String> rawDataMap) {
        if(pair == null) {
            return null;
        }

        List<Integer> scores = new ArrayList<Integer>();

        CompletionService<MLData> completionService = new ExecutorCompletionService<MLData>(this.threadPool);

        List<Integer> rfTreeSizeList = new ArrayList<Integer>();
        for(final BasicML model: models) {
            // TODO, check if no need 'if' condition and refactor two if for loops please
            if(model instanceof BasicNetwork) {
                final BasicNetwork network = (BasicNetwork) model;
                if(network.getInputCount() != pair.getInput().size()) {
                    log.error("Network and input size mismatch: Network Size = " + network.getInputCount()
                            + "; Input Size = " + pair.getInput().size());
                    continue;
                }
                completionService.submit(new Callable<MLData>() {
                    @Override
                    public MLData call() throws Exception {
                        return network.compute(pair.getInput());
                    }
                });
            } else if(model instanceof SVM) {
                final SVM svm = (SVM) model;
                if(svm.getInputCount() != pair.getInput().size()) {
                    log.error("SVM and input size mismatch: SVM Size = " + svm.getInputCount() + "; Input Size = "
                            + pair.getInput().size());
                    continue;
                }
                completionService.submit(new Callable<MLData>() {
                    @Override
                    public MLData call() throws Exception {
                        return svm.compute(pair.getInput());
                    }
                });
            } else if(model instanceof LR) {
                final LR lr = (LR) model;
                if(lr.getInputCount() != pair.getInput().size()) {
                    log.error("LR and input size mismatch: LR Size = " + lr.getInputCount() + "; Input Size = "
                            + pair.getInput().size());
                    continue;
                }
                completionService.submit(new Callable<MLData>() {
                    @Override
                    public MLData call() throws Exception {
                        return lr.compute(pair.getInput());
                    }
                });
            } else if(model instanceof TreeModel) {
                final TreeModel tm = (TreeModel) model;
                if(tm.getInputCount() != pair.getInput().size()) {
                    throw new RuntimeException("GBDT and input size mismatch: rf Size = " + tm.getInputCount()
                            + "; Input Size = " + pair.getInput().size());
                }
                completionService.submit(new Callable<MLData>() {
                    @Override
                    public MLData call() throws Exception {
                        MLData result = tm.compute(pair.getInput());
                        return result;
                    }
                });
            } else {
                throw new RuntimeException("unsupport models");
            }
        }

        int rCnt = 0;
        while(rCnt < this.models.size()) {
            MLData score = null;
            try {
                score = completionService.take().get();
            } catch (ExecutionException e) {
                throw new RuntimeException(e);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            BasicML model = this.models.get(rCnt);

            if(model instanceof BasicNetwork) {
                if(modelConfig != null && modelConfig.isRegression()) {
                    scores.add(toScore(score.getData(0)));
                } else if(modelConfig.isClassification() && modelConfig.getTrain().isOneVsAll()) {
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
                        scores.add((int) sc);
                    }
                } else {
                    // if one vs all consider
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

            rCnt += 1;
        }

        Integer tag = (int) pair.getIdeal().getData(0);

        if(scores.size() == 0) {
            log.error("No Scores Calculated...");
            return null;
        }

        return new ScoreObject(scores, tag, rfTreeSizeList);
    }

    private Integer toScore(Double d) {
        return (int) Math.round(d * 1000);
    }
}
