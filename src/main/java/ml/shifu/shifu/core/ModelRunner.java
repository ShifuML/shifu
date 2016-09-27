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

import java.util.List;
import java.util.Map;

import ml.shifu.shifu.container.CaseScoreResult;
import ml.shifu.shifu.container.ScoreObject;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelTrainConf.ALGORITHM;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.commons.collections.MapUtils;
import org.apache.pig.backend.executionengine.ExecException;
import org.apache.pig.data.Tuple;
import org.encog.ml.BasicML;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * ModelRunner class is to load the model and run the model for input data
 * Currently it provides three API: one for UDF input - tuple, one for String input, one for map
 * <p/>
 * The output result for ModelRunnder is @CaseScoreResult. In the result, not only max/min/average score will be stored,
 * but also Map of raw input
 * <p/>
 * If the elements in the input is not equal with the length of header[], it will return null
 */
public class ModelRunner {

    public static Logger log = LoggerFactory.getLogger(Scorer.class);

    protected ModelConfig modelConfig;
    protected List<ColumnConfig> columnConfigList;

    private String[] header;
    private String dataDelimiter;
    private Scorer scorer;

    public ModelRunner(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, String[] header,
            String dataDelimiter, List<BasicML> models) {
        this.modelConfig = modelConfig;
        this.columnConfigList = columnConfigList;
        this.header = header;
        this.dataDelimiter = dataDelimiter;
        this.scorer = new Scorer(models, columnConfigList, modelConfig.getAlgorithm(), modelConfig,
                modelConfig.getNormalizeStdDevCutOff());
    }

    /**
     * Constructor for Integration API, if user use this constructor to construct @ModelRunner,
     * only compute(Map<String, String> rawDataMap) is supported to call.
     * That means client is responsible for preparing the input data map.
     * <p/>
     * Notice, the Standard deviation Cutoff will be default - Normalizer.STD_DEV_CUTOFF
     * 
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
     * only compute(Map<String, String> rawDataMap) is supported to call.
     * That means client is responsible for preparing the input data map.
     * 
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
     * @return @CaseScoreResult
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
     * Run model to compute score for input tuple
     * 
     * @param tuple
     *            - the whole original input data as @Tuple
     * @return @CaseScoreResult
     */
    public CaseScoreResult compute(Tuple tuple) throws ExecException {
        if(header == null) {
            throw new UnsupportedOperationException("The header are null, please use right constructor!");
        }

        Map<String, String> rawDataMap = CommonUtils.convertDataIntoMap(tuple, header);

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
     * @return @CaseScoreResult
     */
    public CaseScoreResult compute(Map<String, String> rawDataMap) {
        CaseScoreResult scoreResult = new CaseScoreResult();

        ScoreObject so = scorer.score(rawDataMap);
        if(so == null) {
            return null;
        }

        scoreResult.setScores(so.getScores());
        scoreResult.setMaxScore(so.getMaxScore());
        scoreResult.setMinScore(so.getMinScore());
        scoreResult.setAvgScore(so.getMeanScore());
        scoreResult.setMedianScore(so.getMedianScore());

        return scoreResult;
    }

}
