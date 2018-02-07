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

import java.io.File;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

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
import ml.shifu.shifu.util.HDFSUtils;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.fs.Path;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.google.common.collect.Lists;

/**
 * ModelConfig is for ModelConfig.json configurations.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ModelConfig {

    @JsonIgnore
    private final static Logger LOG = LoggerFactory.getLogger(ModelConfig.class);

    /**
     * Basic information like name, version, author ...
     */
    private ModelBasicConf basic = new ModelBasicConf();

    /**
     * Data configuration like data location, data schema, ...
     */
    private ModelSourceDataConf dataSet = new ModelSourceDataConf();

    /**
     * Stats configuration like sample rate, stats method ...
     */
    private ModelStatsConf stats = new ModelStatsConf();

    /**
     * Var select configuration parameters like filterNum, filterBy ...
     */
    private ModelVarSelectConf varSelect = new ModelVarSelectConf();

    /**
     * Normalizing configurations like norm type, sampleRate ...
     */
    private ModelNormalizeConf normalize = new ModelNormalizeConf();

    /**
     * Model training configurations like baggingNum, algorithm (LR/NN/GBT/RF), training parameters ...
     */
    private ModelTrainConf train = new ModelTrainConf();

    /**
     * Eval configurations listed to evaluated trained models.
     */
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

    /**
     * Create init ModelConfig.json
     * 
     * @param modelName
     *            name of model dataset
     * @param alg
     *            , algorithm used, for LR/NN/RF/GBT, diferent init parameters will be set
     * @param description
     *            data set description
     * @param enableHadoop
     *            if it is distributed Hadoop cluster mode
     * @return ModelConfig instance
     * @throws IOException
     *             if any exception in column configuration file creation
     */
    public static ModelConfig createInitModelConfig(String modelName, ALGORITHM alg, String description,
            boolean enableHadoop) throws IOException {
        ModelConfig modelConfig = new ModelConfig();

        DateFormat df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");

        // build meta info
        ModelBasicConf basic = new ModelBasicConf();
        basic.setName(modelName);

        basic.setAuthor(Environment.getProperty(Environment.SYSTEM_USER));
        basic.setRunMode(enableHadoop ? RunMode.DIST : RunMode.LOCAL);
        basic.setDescription("Created at " + df.format(new Date()));
        modelConfig.setBasic(basic);

        // build data set info
        ModelSourceDataConf dataSet = new ModelSourceDataConf();
        dataSet.setDataDelimiter("|");

        String exampleLocalDSPath = new File(Environment.getProperty(Environment.SHIFU_HOME), File.separator
                + "example" + File.separator + "cancer-judgement" + File.separator + "DataStore" + File.separator
                + "DataSet1").toString();
        if(enableHadoop) {
            Path dst = new Path(File.separator + "user" + File.separator
                    + Environment.getProperty(Environment.SYSTEM_USER) + File.separator + "cancer-judgement");
            HDFSUtils.getFS().delete(dst, true);
            HDFSUtils.getFS().mkdirs(dst);

            HDFSUtils.getFS().copyFromLocalFile(new Path(exampleLocalDSPath), dst);
            dataSet.setSource(SourceType.HDFS);
            dataSet.setDataPath(new File(File.separator + "user" + File.separator
                    + Environment.getProperty(Environment.SYSTEM_USER) + File.separator + "cancer-judgement"
                    + File.separator + "DataSet1").toString());
            dataSet.setHeaderPath(new File(File.separator + "user" + File.separator
                    + Environment.getProperty(Environment.SYSTEM_USER) + File.separator + "cancer-judgement"
                    + File.separator + "DataSet1" + File.separator + ".pig_header").toString());
        } else {
            dataSet.setSource(SourceType.LOCAL);
            dataSet.setDataPath(exampleLocalDSPath);
            dataSet.setHeaderPath(exampleLocalDSPath + File.separator + ".pig_header");
        }

        dataSet.setTargetColumnName("diagnosis");

        List<String> posTags = new ArrayList<String>();
        posTags.add("M");
        List<String> negTags = new ArrayList<String>();
        negTags.add("B");

        dataSet.setPosTags(posTags);
        dataSet.setNegTags(negTags);

        dataSet.setMissingOrInvalidValues(Lists.asList("", new String[] { "*", "#", "?", "null", "~" }));
        // create empty <ModelName>/meta.column.names
        ShifuFileUtils.createFileIfNotExists(new Path(modelName, Constants.COLUMN_META_FOLDER_NAME + File.separator
                + Constants.DEFAULT_META_COLUMN_FILE).toString(), SourceType.LOCAL);
        dataSet.setMetaColumnNameFile(Constants.COLUMN_META_FOLDER_NAME + File.separator
                + Constants.DEFAULT_META_COLUMN_FILE);
        // create empty <ModelName>/categorical.column.names
        ShifuFileUtils.createFileIfNotExists(new Path(modelName, Constants.COLUMN_META_FOLDER_NAME + File.separator
                + Constants.DEFAULT_CATEGORICAL_COLUMN_FILE).toString(), SourceType.LOCAL);
        dataSet.setCategoricalColumnNameFile(Constants.COLUMN_META_FOLDER_NAME + File.separator
                + Constants.DEFAULT_CATEGORICAL_COLUMN_FILE);
        modelConfig.setDataSet(dataSet);

        // build stats info
        modelConfig.setStats(new ModelStatsConf());
        modelConfig.setBinningAlgorithm(BinningAlgorithm.SPDTI);

        // build normalize info
        modelConfig.setNormalize(new ModelNormalizeConf());

        // build varselect info
        ModelVarSelectConf varselect = new ModelVarSelectConf();
        // create empty <ModelName>/forceselect.column.names
        ShifuFileUtils.createFileIfNotExists(new Path(modelName, Constants.COLUMN_META_FOLDER_NAME + File.separator
                + Constants.DEFAULT_FORCESELECT_COLUMN_FILE).toString(), SourceType.LOCAL);
        varselect.setForceSelectColumnNameFile(Constants.COLUMN_META_FOLDER_NAME + File.separator
                + Constants.DEFAULT_FORCESELECT_COLUMN_FILE);

        // create empty <ModelName>/forceremove.column.names
        ShifuFileUtils.createFileIfNotExists(new Path(modelName, Constants.COLUMN_META_FOLDER_NAME + File.separator
                + Constants.DEFAULT_FORCEREMOVE_COLUMN_FILE).toString(), SourceType.LOCAL);
        varselect.setForceRemoveColumnNameFile(Constants.COLUMN_META_FOLDER_NAME + File.separator
                + Constants.DEFAULT_FORCEREMOVE_COLUMN_FILE);
        varselect.setFilterEnable(Boolean.TRUE);
        varselect.setFilterNum(200);
        modelConfig.setVarSelect(varselect);

        // build train info
        ModelTrainConf trainConf = new ModelTrainConf();

        trainConf.setAlgorithm(alg.name());
        trainConf.setEpochsPerIteration(1);
        trainConf.setParams(ModelTrainConf.createParamsByAlg(alg, trainConf));
        trainConf.setNumTrainEpochs(100);
        if(ALGORITHM.NN.equals(alg)) {
            trainConf.setNumTrainEpochs(200);
        } else if(ALGORITHM.SVM.equals(alg)) {
            trainConf.setNumTrainEpochs(100);
        } else if(ALGORITHM.RF.equals(alg)) {
            trainConf.setNumTrainEpochs(20000);
        } else if(ALGORITHM.GBT.equals(alg)) {
            trainConf.setNumTrainEpochs(20000);
        } else if(ALGORITHM.LR.equals(alg)) {
            trainConf.setNumTrainEpochs(100);
        }
        trainConf.setBaggingWithReplacement(false);
        modelConfig.setTrain(trainConf);

        EvalConfig evalConfig = new EvalConfig();
        evalConfig.setName("Eval1");
        RawSourceData evalSet = modelConfig.getDataSet().cloneRawSourceData();
        evalSet.setDataDelimiter("|");
        String exampleLocalESFolder = new File(Environment.getProperty(Environment.SHIFU_HOME), File.separator
                + "example" + File.separator + "cancer-judgement" + File.separator + "DataStore" + File.separator
                + "EvalSet1").toString();
        if(enableHadoop) {
            evalSet.setSource(SourceType.HDFS);
            Path dst = new Path(File.separator + "user" + File.separator
                    + Environment.getProperty(Environment.SYSTEM_USER) + File.separator + "cancer-judgement");
            HDFSUtils.getFS().copyFromLocalFile(new Path(exampleLocalESFolder), dst);

            evalSet.setDataPath(new File(File.separator + "user" + File.separator
                    + Environment.getProperty(Environment.SYSTEM_USER) + File.separator + "cancer-judgement"
                    + File.separator + "EvalSet1").toString());
            evalSet.setHeaderPath(new File(File.separator + "user" + File.separator
                    + Environment.getProperty(Environment.SYSTEM_USER) + File.separator + "cancer-judgement"
                    + File.separator + "EvalSet1" + File.separator + ".pig_header").toString());
        } else {
            evalSet.setSource(SourceType.LOCAL);
            evalSet.setDataPath(exampleLocalESFolder);
            evalSet.setHeaderPath(exampleLocalESFolder + File.separator + ".pig_header");
        }
        // create empty <ModelName>/<EvalSetName>.meta.column.names
        String namesFilePath = Constants.COLUMN_META_FOLDER_NAME + File.separator + evalConfig.getName() + "."
                + Constants.DEFAULT_META_COLUMN_FILE;
        ShifuFileUtils.createFileIfNotExists(new Path(modelName, namesFilePath).toString(), SourceType.LOCAL);
        evalSet.setMetaColumnNameFile(namesFilePath);
        evalConfig.setDataSet(evalSet);

        // create empty <ModelName>/<EvalSetName>Score.meta.column.names
        namesFilePath = Constants.COLUMN_META_FOLDER_NAME + File.separator + evalConfig.getName()
                + Constants.DEFAULT_CHAMPIONSCORE_META_COLUMN_FILE;
        ShifuFileUtils.createFileIfNotExists(new Path(modelName, namesFilePath).toString(), SourceType.LOCAL);
        evalConfig.setScoreMetaColumnNameFile(namesFilePath);

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
    public boolean isRegression() {
        return (CollectionUtils.isNotEmpty(dataSet.getPosTags()) && CollectionUtils.isNotEmpty(dataSet.getNegTags()));
    }

    @JsonIgnore
    public boolean isClassification() {
        return (CollectionUtils.isNotEmpty(dataSet.getPosTags()) && CollectionUtils.isEmpty(dataSet.getNegTags()))
                || (CollectionUtils.isEmpty(dataSet.getPosTags()) && CollectionUtils.isNotEmpty(dataSet.getNegTags()));
    }

    /*
     * Flattened tags for multiple classification. '1', '2|3' will be flattened to '1', '2', '3'. While '2' and '3' are
     * combined to one class.
     */
    @JsonIgnore
    public List<String> getFlattenTags() {
        return getFlattenTags(dataSet.getPosTags(), dataSet.getNegTags());
    }

    @JsonIgnore
    public List<String> getFlattenTags(List<String> tags1, List<String> tags2) {
        List<String> tags = new ArrayList<String>();
        if(CollectionUtils.isNotEmpty(tags1)) {
            for(String tag: tags1) {
                if(tag.contains("|")) {
                    for(String inTag: tag.split("\\|")) {
                        // FIXME, if blank or not
                        if(StringUtils.isNotBlank(inTag)) {
                            tags.add(inTag);
                        }
                    }
                } else {
                    tags.add(tag);
                }
            }
        }
        if(CollectionUtils.isNotEmpty(tags2)) {
            for(String tag: tags2) {
                if(tag.contains("|")) {
                    for(String inTag: tag.split("\\|")) {
                        if(StringUtils.isNotBlank(inTag)) {
                            tags.add(inTag);
                        }
                    }
                } else {
                    tags.add(tag);
                }
            }
        }
        return tags;
    }

    @JsonIgnore
    public List<String> getTags(List<String> tags1, List<String> tags2) {
        List<String> tags = new ArrayList<String>();
        if(CollectionUtils.isNotEmpty(tags1)) {
            for(String tag: tags1) {
                tags.add(tag);
            }
        }
        if(CollectionUtils.isNotEmpty(tags2)) {
            for(String tag: tags2) {
                tags.add(tag);
            }
        }
        return tags;
    }

    @JsonIgnore
    public List<String> getTags() {
        return getTags(dataSet.getPosTags(), dataSet.getNegTags());
    }

    @JsonIgnore
    public List<Set<String>> getSetTags(List<String> tags1, List<String> tags2) {
        List<String> tags = getTags(tags1, tags2);
        List<Set<String>> result = new ArrayList<Set<String>>();
        for(String tag: tags) {
            Set<String> set = new HashSet<String>(16);
            if(tag.contains("|")) {
                for(String inTag: tag.split("\\|")) {
                    if(StringUtils.isNotBlank(inTag)) {
                        set.add(inTag);
                    }
                }
            } else {
                set.add(tag);
            }
            result.add(set);
        }
        return result;
    }

    @JsonIgnore
    public List<Set<String>> getSetTags() {
        return getSetTags(dataSet.getPosTags(), dataSet.getNegTags());
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
    public Double getValidSetRate() {
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
    public String getValidationDataSetRawPath() {
        return dataSet.getValidationDataPath();
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
        return RunMode.MAPRED == basic.getRunMode() || RunMode.DIST == basic.getRunMode();
    }

    @JsonIgnore
    public boolean isDistributedRunMode() {
        return isMapReduceRunMode();
    }

    @JsonIgnore
    public boolean isLocalRunMode() {
        return RunMode.LOCAL.equals(basic.getRunMode());
    }

    @JsonIgnore
    public List<String> getMetaColumnNames() throws IOException {
        String delimiter = StringUtils.isBlank(this.getHeaderDelimiter()) ? this.getDataSetDelimiter() : this
                .getHeaderDelimiter();
        String metaColumnNameFile = dataSet.getMetaColumnNameFile();
        if(StringUtils.isBlank(metaColumnNameFile)) {
            String defaultMetaColumnFileName = Constants.COLUMN_META_FOLDER_NAME + File.separator
                    + Constants.DEFAULT_META_COLUMN_FILE;
            if(ShifuFileUtils.isFileExists(defaultMetaColumnFileName, SourceType.LOCAL)) {
                metaColumnNameFile = defaultMetaColumnFileName;
                LOG.warn(
                        "'dataSet::metaColumnNameFile' is not set while default metaColumnNameFile: {} is found, default meta file will be used.",
                        defaultMetaColumnFileName);
            } else {
                LOG.warn(
                        "'dataSet::metaColumnNameFile' is not set and default metaColumnNameFile: {} is not found, no meta config files, please check and set meta config file in 'dataSet::metaColumnNameFile'.",
                        defaultMetaColumnFileName);
                return new ArrayList<String>();
            }
        }
        return CommonUtils.readConfFileIntoList(metaColumnNameFile, SourceType.LOCAL, delimiter);
    }

    @JsonIgnore
    public List<String> getCategoricalColumnNames() throws IOException {
        String delimiter = StringUtils.isBlank(this.getHeaderDelimiter()) ? this.getDataSetDelimiter() : this
                .getHeaderDelimiter();

        String categoricalColumnNameFile = dataSet.getCategoricalColumnNameFile();
        if(StringUtils.isBlank(categoricalColumnNameFile)) {
            String defaultCategoricalColumnNameFile = Constants.COLUMN_META_FOLDER_NAME + File.separator
                    + Constants.DEFAULT_CATEGORICAL_COLUMN_FILE;
            if(ShifuFileUtils.isFileExists(defaultCategoricalColumnNameFile, SourceType.LOCAL)) {
                categoricalColumnNameFile = defaultCategoricalColumnNameFile;
                LOG.warn(
                        "'dataSet::categoricalColumnNameFile' is not set while default categoricalColumnNameFile: {} is found, default categorical file will be used.",
                        defaultCategoricalColumnNameFile);
            } else {
                LOG.warn(
                        "'dataSet::categoricalColumnNameFile' is not set and default categoricalColumnNameFile: {} is not found, no categorical config files, please check and set categorical config file in 'dataSet::categoricalColumnNameFile'.",
                        defaultCategoricalColumnNameFile);
                return new ArrayList<String>();
            }
        }
        return CommonUtils.readConfFileIntoList(categoricalColumnNameFile, SourceType.LOCAL, delimiter);
    }

    @JsonIgnore
    public List<String> getSegmentFilterExpressions() throws IOException {
        String expressionFile = dataSet.getSegExpressionFile();
        if(StringUtils.isBlank(expressionFile)) {
            String defaultExpressionFile = Constants.COLUMN_META_FOLDER_NAME + File.separator
                    + Constants.DEFAULT_EXPRESSION_COLUMN_FILE;
            if(ShifuFileUtils.isFileExists(defaultExpressionFile, SourceType.LOCAL)) {
                expressionFile = defaultExpressionFile;
                LOG.warn(
                        "'dataSet::segExpressionFile' is not set while default segExpressionFile: {} is found, default expression file will be used.",
                        defaultExpressionFile);
            } else {
                return new ArrayList<String>();
            }
        }
        return CommonUtils.readConfFileIntoList(expressionFile, SourceType.LOCAL,
                Constants.SHIFU_STATS_FILTER_EXPRESSIONS_DELIMETER);
    }

    @JsonIgnore
    public String getSegmentFilterExpressionsAsString() throws IOException {
        List<String> expressions = getSegmentFilterExpressions();
        StringBuilder sb = new StringBuilder(1000);
        for(int i = 0; i < expressions.size(); i++) {
            if(i == expressions.size() - 1) {
                sb.append(expressions.get(i));
            } else {
                sb.append(expressions.get(i)).append(Constants.SHIFU_STATS_FILTER_EXPRESSIONS_DELIMETER);
            }
        }
        return sb.toString();
    }

    @JsonIgnore
    public Map<String, Double> getHybridColumnNames() throws IOException {
        String delimiter = StringUtils.isBlank(this.getHeaderDelimiter()) ? this.getDataSetDelimiter() : this
                .getHeaderDelimiter();

        String hybridColumnNameFile = dataSet.getHybridColumnNameFile();
        if(StringUtils.isBlank(hybridColumnNameFile)) {
            String defaultHybridColumnNameFile = Constants.COLUMN_META_FOLDER_NAME + File.separator
                    + Constants.DEFAULT_HYBRID_COLUMN_FILE;
            if(ShifuFileUtils.isFileExists(defaultHybridColumnNameFile, SourceType.LOCAL)) {
                hybridColumnNameFile = defaultHybridColumnNameFile;
                LOG.warn(
                        "'dataSet::hybridColumnNameFile' is not set while default hybridColumnNameFile: {} is found, default hybrid file will be used.",
                        defaultHybridColumnNameFile);
            } else {
                LOG.warn(
                        "'dataSet::hybridColumnNameFile' is not set and default hybridColumnNameFile: {} is not found, no hybrid configs.",
                        defaultHybridColumnNameFile);
                return new HashMap<String, Double>();
            }
        }
        List<String> list = CommonUtils.readConfFileIntoList(hybridColumnNameFile, SourceType.LOCAL, delimiter);
        Map<String, Double> map = new HashMap<String, Double>();
        for(String string: list) {
            if(string.contains(Constants.DEFAULT_DELIMITER)) {
                String[] splits = CommonUtils.split(string, Constants.DEFAULT_DELIMITER);
                double parNum = CommonUtils.parseNumber(splits[1]);
                if(Double.isNaN(parNum)) {
                    map.put(string, Double.NEGATIVE_INFINITY);
                } else {
                    map.put(string, parNum);
                }
            } else {
                map.put(string, Double.NEGATIVE_INFINITY);
            }
        }
        return map;
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
    public String getVarSelectFilterBy() {
        return varSelect.getFilterBy();
    }

    @JsonIgnore
    public Integer getVarSelectFilterNum() {
        return varSelect.getFilterNum();
    }

    @JsonIgnore
    public boolean isCategoricalDisabled() {
        // there is no settings now, always enable
        return false;
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
    public List<String> getListCandidates() throws IOException {
        String delimiter = StringUtils.isBlank(this.getHeaderDelimiter()) // header delimiter has higher priority
        ? this.getDataSetDelimiter()
                : this.getHeaderDelimiter();

        String candidateColumnNameFile = varSelect.getCandidateColumnNameFile();
        if(StringUtils.isBlank(candidateColumnNameFile)) {
            String defaultCandidateColumnNameFile = Constants.COLUMN_META_FOLDER_NAME + File.separator
                    + Constants.DEFAULT_CANDIDATE_COLUMN_FILE;
            if(ShifuFileUtils.isFileExists(defaultCandidateColumnNameFile, SourceType.LOCAL)) {
                candidateColumnNameFile = defaultCandidateColumnNameFile;
                LOG.warn(
                        "'varSelect::candidateColumnNameFile' is not set while default candidateColumnNameFile: {} is found, default candidate file will be used.",
                        defaultCandidateColumnNameFile);
            } else {
                LOG.warn(
                        "'varSelect::candidateColumnNameFile' is not set and default candidateColumnNameFile: {} is not found, no candidate config files, please check and set candidate columns in 'varSelect::candidateColumnNameFile'.",
                        defaultCandidateColumnNameFile);
                return new ArrayList<String>();
            }
        }
        return CommonUtils.readConfFileIntoList(candidateColumnNameFile, SourceType.LOCAL, delimiter);
    }

    @JsonIgnore
    public List<String> getListForceRemove() throws IOException {
        String delimiter = StringUtils.isBlank(this.getHeaderDelimiter()) ? this.getDataSetDelimiter() : this
                .getHeaderDelimiter();

        String forceRemoveColumnNameFile = varSelect.getForceRemoveColumnNameFile();
        if(StringUtils.isBlank(forceRemoveColumnNameFile)) {
            String defaultForceRemoveColumnNameFile = Constants.COLUMN_META_FOLDER_NAME + File.separator
                    + Constants.DEFAULT_FORCEREMOVE_COLUMN_FILE;
            if(ShifuFileUtils.isFileExists(defaultForceRemoveColumnNameFile, SourceType.LOCAL)) {
                forceRemoveColumnNameFile = defaultForceRemoveColumnNameFile;
                LOG.warn(
                        "'varSelect::forceRemoveColumnNameFile' is not set while default forceRemoveColumnNameFile: {} is found, default force-remove file will be used.",
                        defaultForceRemoveColumnNameFile);
            } else {
                LOG.warn(
                        "'varSelect::forceRemoveColumnNameFile' is not set and default forceRemoveColumnNameFile: {} is not found, no force-remove config files, please check and set force-select config file in 'varSelect::forceRemoveColumnNameFile'.",
                        defaultForceRemoveColumnNameFile);
                return new ArrayList<String>();
            }
        }
        return CommonUtils.readConfFileIntoList(forceRemoveColumnNameFile, SourceType.LOCAL, delimiter);
    }

    @JsonIgnore
    public List<String> getListForceSelect() throws IOException {
        String delimiter = StringUtils.isBlank(this.getHeaderDelimiter()) ? this.getDataSetDelimiter() : this
                .getHeaderDelimiter();

        String forceSelectColumnNameFile = varSelect.getForceSelectColumnNameFile();
        if(StringUtils.isBlank(forceSelectColumnNameFile)) {
            String defaultForceSelectColumnNameFile = Constants.COLUMN_META_FOLDER_NAME + File.separator
                    + Constants.DEFAULT_FORCESELECT_COLUMN_FILE;
            if(ShifuFileUtils.isFileExists(defaultForceSelectColumnNameFile, SourceType.LOCAL)) {
                forceSelectColumnNameFile = defaultForceSelectColumnNameFile;
                LOG.warn(
                        "'varSelect::forceSelectColumnNameFile' is not set while default forceSelectColumnNameFile: {} is found, default force-select file will be used.",
                        defaultForceSelectColumnNameFile);
            } else {
                LOG.warn(
                        "'varSelect::forceSelectColumnNameFile' is not set and default forceSelectColumnNameFile: {} is not found, no force-select config files, please check and set force-select config file in 'varSelect::forceSelectColumnNameFile'.",
                        defaultForceSelectColumnNameFile);
                return new ArrayList<String>();
            }
        }

        return CommonUtils.readConfFileIntoList(forceSelectColumnNameFile, SourceType.LOCAL, delimiter);
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
    public String getPsiColumnName() {
        return this.stats.getPsiColumnName();
    }

    @JsonIgnore
    public void setPsiColumnName(String columnName) {
        this.stats.setPsiColumnName(columnName);
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

    @Override
    public ModelConfig clone() {
        ModelConfig other = new ModelConfig();
        other.setBasic(basic.clone());
        other.setDataSet(dataSet.clone());
        other.setStats(stats.clone());
        other.setVarSelect(varSelect.clone());
        other.setNormalize(normalize.clone());
        other.setTrain(train.clone());

        List<EvalConfig> evalConfigs = new ArrayList<EvalConfig>();
        for(EvalConfig evalConfig: this.evals) {
            evalConfigs.add(evalConfig.clone());
        }
        other.setEvals(evalConfigs);

        return other;
    }

}
