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

import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import ml.shifu.shifu.column.NSColumn;
import ml.shifu.shifu.container.CaseScoreResult;
import ml.shifu.shifu.container.ScoreObject;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelTrainConf.ALGORITHM;
import ml.shifu.shifu.core.model.ModelSpec;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.commons.collections.MapUtils;
import org.encog.ml.BasicML;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * ModelRunner class is to load the model and run the model for input data
 * Currently it provides three API: one for UDF input - tuple, one for String input, one for map
 * <p>
 * The output result for ModelRunnder is @CaseScoreResult. In the result, not only max/min/average score will be stored,
 * but also Map of raw input
 * </p>
 * <p>
 * If the elements in the input is not equal with the length of header[], it will return null
 * </p>
 * {@link #close()} must be called at caller to release resources.
 */
public class ModelRunner {

    public static Logger log = LoggerFactory.getLogger(Scorer.class);

    protected ModelConfig modelConfig;
    protected List<ColumnConfig> columnConfigList;

    private String[] header;
    private String dataDelimiter;
    private Scorer scorer;
    private Map<String, Scorer> subScorers;

    public ModelRunner(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, String[] header,
            String dataDelimiter, List<BasicML> models) {
        this(modelConfig, columnConfigList, header, dataDelimiter, models, 0);
    }

    public ModelRunner(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, String[] header,
            String dataDelimiter, List<BasicML> models, int outputHiddenLayerIndex) {
        this.modelConfig = modelConfig;
        this.columnConfigList = columnConfigList;
        this.header = header;
        this.dataDelimiter = dataDelimiter;
        this.scorer = new Scorer(models, columnConfigList, modelConfig.getAlgorithm(), modelConfig,
                modelConfig.getNormalizeStdDevCutOff(), outputHiddenLayerIndex);
    }

    /**
     * Constructor for Integration API, if user use this constructor to construct @ModelRunner,
     * only compute(Map(String, String) rawDataMap) is supported to call.
     * That means client is responsible for preparing the input data map.
     * <p>
     * Notice, the Standard deviation Cutoff will be default - Normalizer.STD_DEV_CUTOFF
     * 
     * @param modelConfig
     *            model config
     * @param columnConfigList
     *            - @ColumnConfig list for Model
     * @param models
     *            - models
     */
    public ModelRunner(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, List<BasicML> models) {
        this(modelConfig, columnConfigList, models, Normalizer.STD_DEV_CUTOFF);
    }

    /**
     * Constructor for Integration API, if user use this constructor to construct @ModelRunner,
     * only compute(Map(String, String) rawDataMap) is supported to call.
     * That means client is responsible for preparing the input data map.
     * 
     * @param modelConfig
     *            the modelconfig
     * @param columnConfigList
     *            - @ColumnConfig list for Model
     * @param models
     *            - models
     * @param stdDevCutoff
     *            - the standard deviation cutoff to normalize data
     */
    public ModelRunner(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, List<BasicML> models,
            double stdDevCutoff) {
        this.columnConfigList = columnConfigList;
        this.modelConfig = modelConfig;
        this.scorer = new Scorer(models, columnConfigList, ALGORITHM.NN.name(), modelConfig, stdDevCutoff);
    }

    /**
     * Run model to compute score for inputData
     * 
     * @param inputData
     *            - the whole original input data as String
     * @return CaseScoreResult
     */
    public CaseScoreResult compute(String inputData) {
        if(dataDelimiter == null || header == null) {
            throw new UnsupportedOperationException(
                    "The dataDelimiter and header are null, please use right constructor!");
        }

        Map<String, String> rawDataMap = CommonUtils.convertDataIntoMap(inputData, dataDelimiter, header);

        if(MapUtils.isEmpty(rawDataMap)) {
            return null;
        }
        return compute(rawDataMap);
    }

    /**
     * Run model to compute score for input data map
     * 
     * @param rawDataMap
     *            - the whole original input data as map
     * @return CaseScoreResult
     */
    public CaseScoreResult compute(Map<String, String> rawDataMap) {
        return computeNsData(CommonUtils.convertRawMapToNsDataMap(rawDataMap));
    }

    /**
     * Run model to compute score for input NS Data map
     * 
     * @param rawDataNsMap
     *            - the original input, but key is wrapped by NSColumn
     * @return CaseScoreResult - model score
     */
    public CaseScoreResult computeNsData(Map<NSColumn, String> rawDataNsMap) {
        if(MapUtils.isEmpty(rawDataNsMap)) {
            return null;
        }

        CaseScoreResult scoreResult = new CaseScoreResult();

        if(this.scorer != null) {
            ScoreObject so = scorer.scoreNsData(rawDataNsMap);
            if(so == null) {
                return null;
            }

            scoreResult.setScores(so.getScores());
            scoreResult.setMaxScore(so.getMaxScore());
            scoreResult.setMinScore(so.getMinScore());
            scoreResult.setAvgScore(so.getMeanScore());
            scoreResult.setMedianScore(so.getMedianScore());
            scoreResult.setHiddenLayerScores(so.getHiddenLayerScores());
        }

        if(MapUtils.isNotEmpty(this.subScorers)) {
            Iterator<Map.Entry<String, Scorer>> iterator = this.subScorers.entrySet().iterator();
            while(iterator.hasNext()) {
                Map.Entry<String, Scorer> entry = iterator.next();
                String modelName = entry.getKey();
                Scorer subScorer = entry.getValue();
                ScoreObject so = subScorer.scoreNsData(rawDataNsMap);
                if(so != null) {
                    scoreResult.addSubModelScore(modelName, so);
                }
            }
        }

        return scoreResult;
    }

    /**
     * add @ModelSpec as sub-model. Create scorer for sub-model
     * 
     * @param modelSpec
     *            - model spec for sub model
     */
    public void addSubModels(ModelSpec modelSpec) {
        if(this.subScorers == null) {
            this.subScorers = new TreeMap<String, Scorer>();
        }

        this.subScorers.put(modelSpec.getModelName(), new Scorer(modelSpec.getModels(),
                modelSpec.getColumnConfigList(), modelSpec.getAlgorithm().name(), modelSpec.getModelConfig(), modelSpec
                        .getModelConfig().getNormalizeStdDevCutOff()));
    }

    /**
     * Get the models count of current model
     * 
     * @return - model count
     */
    public int getModelsCnt() {
        return (this.scorer == null ? 0 : this.scorer.getModelCnt());
    }

    /**
     * Get the models count of sub-models
     * 
     * @return - model count of sub-models
     */
    public Map<String, Integer> getSubModelsCnt() {
        if(MapUtils.isNotEmpty(this.subScorers)) {
            Map<String, Integer> subModelsCnt = new TreeMap<String, Integer>();
            Iterator<Map.Entry<String, Scorer>> iterator = this.subScorers.entrySet().iterator();
            while(iterator.hasNext()) {
                Map.Entry<String, Scorer> entry = iterator.next();
                subModelsCnt.put(entry.getKey(), entry.getValue().getModelCnt());
            }
            return subModelsCnt;
        } else {
            return null;
        }
    }

    public void setScoreScale(int scale) {
        this.scorer.setScale(scale);
    }

    /**
     * Cleaning the thread pool resources, must be called at last.
     */
    public void close() {
    }

}
