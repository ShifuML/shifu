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

import java.io.File;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Map;

import ml.shifu.shifu.container.obj.ModelBasicConf.RunMode;
import ml.shifu.shifu.container.obj.ModelNormalizeConf.NormType;
import ml.shifu.shifu.container.obj.ModelStatsConf.BinningAlgorithm;
import ml.shifu.shifu.container.obj.ModelStatsConf.BinningMethod;
import ml.shifu.shifu.container.obj.ModelTrainConf.ALGORITHM;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.Environment;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.fs.Path;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.google.common.collect.Lists;

/**
 * ModelConfig class
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ModelConfig {

    private ModelBasicConf basic = new ModelBasicConf();

    private ModelSourceDataConf dataSet = new ModelSourceDataConf();

    private ModelStatsConf stats = new ModelStatsConf();

    private ModelVarSelectConf varSelect = new ModelVarSelectConf();

    private ModelNormalizeConf normalize = new ModelNormalizeConf();

    private ModelTrainConf train = new ModelTrainConf();

    private List<EvalConfig> evals = new ArrayList<EvalConfig>();

    public ModelBasicConf getBasic() {
        return basic;
    }

    public void setBasic(ModelBasicConf basic) {
        this.basic = basic;
    }

    public ModelSourceDataConf getDataSet() {
        return dataSet;
    }

    public void setDataSet(ModelSourceDataConf dataSet) {
        this.dataSet = dataSet;
    }

    public ModelStatsConf getStats() {
        return stats;
    }

    public void setStats(ModelStatsConf stats) {
        this.stats = stats;
    }

    public ModelVarSelectConf getVarSelect() {
        return varSelect;
    }

    public void setVarSelect(ModelVarSelectConf varSelect) {
        this.varSelect = varSelect;
    }

    public ModelNormalizeConf getNormalize() {
        return normalize;
    }

    public void setNormalize(ModelNormalizeConf normalize) {
        this.normalize = normalize;
    }

    public ModelTrainConf getTrain() {
        return train;
    }

    public void setTrain(ModelTrainConf train) {
        this.train = train;
    }

    public List<EvalConfig> getEvals() {
        return evals;
    }

    public void setEvals(List<EvalConfig> evals) {
        this.evals = evals;
    }

    public static ModelConfig createInitModelConfig(String modelName, ALGORITHM alg, String description)
            throws IOException {
        ModelConfig modelConfig = new ModelConfig();

        DateFormat df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");

        // build meta info
        ModelBasicConf basic = new ModelBasicConf();
        basic.setName(modelName);
        basic.setAuthor(Environment.getProperty(Environment.SYSTEM_USER));
        basic.setDescription("Created at " + df.format(new Date()));
        modelConfig.setBasic(basic);

        // build data set info
        ModelSourceDataConf dataSet = new ModelSourceDataConf();
        dataSet.setSource(SourceType.LOCAL);
        dataSet.setDataDelimiter("|");
        dataSet.setDataPath(new File(Environment.getProperty(Environment.SHIFU_HOME), File.separator + "example"
                + File.separator + "cancer-judgement" + File.separator + "DataStore" + File.separator + "DataSet1")
                .toString());
        dataSet.setHeaderPath(new File(Environment.getProperty(Environment.SHIFU_HOME), File.separator + "example"
                + File.separator + "cancer-judgement" + File.separator + "DataStore" + File.separator + "DataSet1"
                + File.separator + ".pig_header").toString());
        dataSet.setTargetColumnName("diagnosis");

        List<String> posTags = new ArrayList<String>();
        posTags.add("M");
        List<String> negTags = new ArrayList<String>();
        negTags.add("B");

        dataSet.setPosTags(posTags);
        dataSet.setNegTags(negTags);

        dataSet.setMissingOrInvalidValues(Lists.asList("", new String[] { "*", "#", "?", "null", "~" }));
        // create empty <ModelName>/meta.column.names
        ShifuFileUtils.createFileIfNotExists(new Path(modelName, Constants.DEFAULT_META_COLUMN_FILE).toString(),
                SourceType.LOCAL);
        dataSet.setMetaColumnNameFile(Constants.DEFAULT_META_COLUMN_FILE);
        // create empty <ModelName>/categorical.column.names
        ShifuFileUtils.createFileIfNotExists(new Path(modelName, Constants.DEFAULT_CATEGORICAL_COLUMN_FILE).toString(),
                SourceType.LOCAL);
        dataSet.setCategoricalColumnNameFile(Constants.DEFAULT_CATEGORICAL_COLUMN_FILE);
        modelConfig.setDataSet(dataSet);

        // build runtime info
        // modelConfig.setRunConf(new ModelRuntimeConf());

        // build stats info
        modelConfig.setStats(new ModelStatsConf());
        modelConfig.setBinningAlgorithm(BinningAlgorithm.SPDTI);

        // build varselect info
        ModelVarSelectConf varselect = new ModelVarSelectConf();
        // create empty <ModelName>/forceselect.column.names
        ShifuFileUtils.createFileIfNotExists(new Path(modelName, Constants.DEFAULT_FORCESELECT_COLUMN_FILE).toString(),
                SourceType.LOCAL);
        varselect.setForceSelectColumnNameFile(Constants.DEFAULT_FORCESELECT_COLUMN_FILE);

        // create empty <ModelName>/forceremove.column.names
        ShifuFileUtils.createFileIfNotExists(new Path(modelName, Constants.DEFAULT_FORCEREMOVE_COLUMN_FILE).toString(),
                SourceType.LOCAL);
        varselect.setForceRemoveColumnNameFile(Constants.DEFAULT_FORCEREMOVE_COLUMN_FILE);
        varselect.setFilterBySE(Boolean.TRUE);
        modelConfig.setVarSelect(varselect);

        // build normalize info
        modelConfig.setNormalize(new ModelNormalizeConf());

        // build train info
        ModelTrainConf trainConf = new ModelTrainConf();

        trainConf.setAlgorithm(alg.name());
        trainConf.setNumTrainEpochs(100);
        trainConf.setEpochsPerIteration(1);
        trainConf.setParams(ModelTrainConf.createParamsByAlg(alg));
        modelConfig.setTrain(trainConf);

        EvalConfig evalConfig = new EvalConfig();
        evalConfig.setName("Eval1");
        RawSourceData evalSet = modelConfig.getDataSet().cloneRawSourceData();
        evalSet.setSource(SourceType.LOCAL);
        evalSet.setDataDelimiter("|");
        evalSet.setDataPath(new File(Environment.getProperty(Environment.SHIFU_HOME), File.separator + "example"
                + File.separator + "cancer-judgement" + File.separator + "DataStore" + File.separator + "EvalSet1")
                .toString());
        evalSet.setHeaderPath(new File(Environment.getProperty(Environment.SHIFU_HOME), File.separator + "example"
                + File.separator + "cancer-judgement" + File.separator + "DataStore" + File.separator + "EvalSet1"
                + File.separator + ".pig_header").toString());
        evalConfig.setDataSet(evalSet);
        // create empty <ModelName>/<EvalSetName>Score.meta.column.names
        ShifuFileUtils.createFileIfNotExists(new Path(modelName, evalConfig.getName()
                + Constants.DEFAULT_EVALSCORE_META_COLUMN_FILE).toString(), SourceType.LOCAL);
        evalConfig.setScoreMetaColumnNameFile(evalConfig.getName() + Constants.DEFAULT_EVALSCORE_META_COLUMN_FILE);
        modelConfig.getEvals().add(evalConfig);

        return modelConfig;
    }

    @JsonIgnore
    public boolean isLocalFileSystem() {
        return SourceType.LOCAL.equals(dataSet.getSource());
    }

    @JsonIgnore
    public boolean isHdfsFileSystem() {
        return SourceType.HDFS.equals(dataSet.getSource());
    }

    @JsonIgnore
    public String getDataSetDelimiter() {
        return dataSet.getDataDelimiter();
    }

    @JsonIgnore
    public int getBaggingNum() {
        return train.getBaggingNum();
    }

    @JsonIgnore
    public Double getNormalizeStdDevCutOff() {
        return normalize.getStdDevCutOff();
    }

    @JsonIgnore
    public boolean isBinningSampleNegOnly() {
        return stats.getSampleNegOnly();
    }

    @JsonIgnore
    public Double getBinningSampleRate() {
        return stats.getSampleRate();
    }

    @JsonIgnore
    public List<String> getPosTags() {
        return dataSet.getPosTags();
    }

    @JsonIgnore
    public List<String> getMissingOrInvalidValues() {
        return dataSet.getMissingOrInvalidValues();
    }

    @JsonIgnore
    public List<String> getPosTags(EvalConfig evalConfig) {
        if(CollectionUtils.isNotEmpty(evalConfig.getDataSet().getPosTags())) {
            return evalConfig.getDataSet().getPosTags();
        } else {
            return dataSet.getPosTags();
        }
    }

    @JsonIgnore
    public List<String> getNegTags() {
        return dataSet.getNegTags();
    }

    @JsonIgnore
    public List<String> getNegTags(EvalConfig evalConfig) {
        if(CollectionUtils.isNotEmpty(evalConfig.getDataSet().getNegTags())) {
            return evalConfig.getDataSet().getNegTags();
        } else {
            return dataSet.getNegTags();
        }
    }

    @JsonIgnore
    public Double getNormalizeSampleRate() {
        return normalize.getSampleRate();
    }
    
    @JsonIgnore
    public NormType getNormalizeType() {
        return normalize.getNormType();
    }

    @JsonIgnore
    public Boolean isNormalizeSampleNegOnly() {
        return normalize.getSampleNegOnly();
    }

    @JsonIgnore
    public int getBinningExpectedNum() {
        return stats.getMaxNumBin();
    }

    @JsonIgnore
    public BinningMethod getBinningMethod() {
        return stats.getBinningMethod();
    }

    @JsonIgnore
    public boolean isTrainOnDisk() {
        return train.getTrainOnDisk();
    }

    @JsonIgnore
    public Double getCrossValidationRate() {
        return train.getValidSetRate();
    }

    @JsonIgnore
    public Double getBaggingSampleRate() {
        return train.getBaggingSampleRate();
    }

    @JsonIgnore
    public boolean isFixInitialInput() {
        return train.getFixInitInput();
    }

    @JsonIgnore
    public boolean isBaggingWithReplacement() {
        return train.getBaggingWithReplacement();
    }

    @JsonIgnore
    public String getAlgorithm() {
        return train.getAlgorithm();
    }

    @JsonIgnore
    public String getModelSetName() {
        return basic.getName();
    }

    @JsonIgnore
    public Integer getAutoTypeThreshold() {
        return stats.getBinningAutoTypeThreshold();
    }

    @JsonIgnore
    public Double getNumericalValueThreshold() {
        return stats.getNumericalValueThreshold();
    }

    @JsonIgnore
    public void setModelSetName(String modelSetName) {
        basic.setName(modelSetName);
    }

    @JsonIgnore
    public void setModelSetCreator(String author) {
        basic.setAuthor(author);
    }

    @JsonIgnore
    public String getDataSetRawPath() {
        return dataSet.getDataPath();
    }

    @JsonIgnore
    public boolean isVariableStoreEnabled() {
        // default - disable
        return false;
    }

    @JsonIgnore
    public String getTargetColumnName() {
        return dataSet.getTargetColumnName();
    }

    @JsonIgnore
    public String getTargetColumnName(EvalConfig evalConfig) {
        if(StringUtils.isNotBlank(evalConfig.getDataSet().getTargetColumnName())) {
            return evalConfig.getDataSet().getTargetColumnName();
        } else {
            return dataSet.getTargetColumnName();
        }
    }

    @JsonIgnore
    public boolean isMapReduceRunMode() {
        return RunMode.mapred.equals(basic.getRunMode());
    }

    @JsonIgnore
    public boolean isLocalRunMode() {
        return RunMode.local.equals(basic.getRunMode());
    }

    @JsonIgnore
    public Boolean getVarSelectWrapperEnabled() {
        return varSelect.getWrapperEnabled();
    }

    @JsonIgnore
    public List<String> getMetaColumnNames() throws IOException {
        return CommonUtils.readConfFileIntoList(dataSet.getMetaColumnNameFile(), SourceType.LOCAL,
                this.getHeaderDelimiter());
    }

    @JsonIgnore
    public List<String> getCategoricalColumnNames() throws IOException {
        return CommonUtils.readConfFileIntoList(dataSet.getCategoricalColumnNameFile(), SourceType.LOCAL,
                this.getHeaderDelimiter());
    }

    @JsonIgnore
    public Map<String, Object> getParams() {
        return train.getParams();
    }

    @JsonIgnore
    public void setParams(Map<String, Object> params) {
        train.setParams(params);
    }

    @JsonIgnore
    public String getFilterExpressions() {
        return dataSet.getFilterExpressions();
    }

    @JsonIgnore
    public Boolean getVarSelectFilterEnabled() {
        return varSelect.getFilterEnable();
    }

    @JsonIgnore
    public Integer getVarSelectFilterNum() {
        return varSelect.getFilterNum();
    }

    @JsonIgnore
    public String getVarSelectFilterBy() {
        return varSelect.getFilterBy();
    }

    @JsonIgnore
    public boolean isCategoricalDisabled() {
        // there is no settings now, always enable
        return false;
    }

    @JsonIgnore
    public String getVarSelectWrapperBy() {
        return varSelect.getWrapperBy();
    }

    @JsonIgnore
    public int getVarSelectWrapperNum() {
        return varSelect.getWrapperNum();
    }

    @JsonIgnore
    public int getNumTrainEpochs() {
        return train.getNumTrainEpochs();
    }

    @JsonIgnore
    public String getMode() {
        return dataSet.getSource().name();
    }

    @JsonIgnore
    public List<String> getListForceRemove() throws IOException {
        return CommonUtils.readConfFileIntoList(varSelect.getForceRemoveColumnNameFile(), SourceType.LOCAL,
                this.getHeaderDelimiter());
    }

    @JsonIgnore
    public List<String> getListForceSelect() throws IOException {
        return CommonUtils.readConfFileIntoList(varSelect.getForceSelectColumnNameFile(), SourceType.LOCAL,
                this.getHeaderDelimiter());
    }

    @JsonIgnore
    public String getModelSetPath() {
        return ".";
    }

    @JsonIgnore
    public Boolean isBinningAutoTypeEnabled() {
        return stats.getBinningAutoTypeEnable();
    }

    @JsonIgnore
    public Integer getBinningAutoTypeThreshold() {
        return stats.getBinningAutoTypeThreshold();
    }

    @JsonIgnore
    public Boolean isBinningMergeEnabled() {
        return stats.getBinningMergeEnable();
    }

    @JsonIgnore
    public String getWeightColumnName() {
        return dataSet.getWeightColumnName();
    }

    @JsonIgnore
    public String getHeaderPath() {
        return dataSet.getHeaderPath();
    }

    @JsonIgnore
    public String getHeaderDelimiter() {
        return dataSet.getHeaderDelimiter();
    }

    @JsonIgnore
    public BinningAlgorithm getBinningAlgorithm() {
        return this.stats.getBinningAlgorithm();
    }

    @JsonIgnore
    public void setBinningAlgorithm(BinningAlgorithm binningAlgorithm) {
        this.stats.setBinningAlgorithm(binningAlgorithm);
    }

    @JsonIgnore
    public EvalConfig getEvalConfigByName(String evalSetName) {
        if(CollectionUtils.isNotEmpty(evals)) {
            for(EvalConfig evalConfig: evals) {
                if(evalConfig.getName().equalsIgnoreCase(evalSetName)) {
                    return evalConfig;
                }
            }
        }

        return null;
    }

    @Override
    public boolean equals(Object obj) {
        if(obj == null || !(obj instanceof ModelConfig)) {
            return false;
        }

        if(obj == this) {
            return true;
        }

        ModelConfig mc = (ModelConfig) obj;
        return mc.getBasic().equals(basic);
    }

    /**
     * Auto generated by eclipse
     */
    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + ((basic == null) ? 0 : basic.hashCode());
        return result;
    }

}
