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
package ml.shifu.shifu.core.processor;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;

import ml.shifu.shifu.column.NSColumn;
import ml.shifu.shifu.column.NSColumnUtils;
import ml.shifu.shifu.container.meta.ValidateResult;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelNormalizeConf.NormType;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.shuffle.MapReduceShuffle;
import ml.shifu.shifu.core.validator.ModelInspector;
import ml.shifu.shifu.core.validator.ModelInspector.ModelStep;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.fs.PathFinder;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.pig.PigExecutor;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.Environment;
import ml.shifu.shifu.util.JSONUtils;
import ml.shifu.shifu.util.updater.ColumnConfigUpdater;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.collections.MapUtils;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.fs.Path;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Model Basic Processor, it helps to do basic manipulate in model, including load/save configuration, copy
 * configuration file
 */
public class BasicModelProcessor {

    private final static Logger LOG = LoggerFactory.getLogger(BasicModelProcessor.class);

    protected ModelConfig modelConfig;
    protected List<ColumnConfig> columnConfigList;
    protected PathFinder pathFinder;

    /**
     * If not specified SHIFU_HOME env, some key configurations like pig path or lib path can be configured here
     */
    protected Map<String, Object> otherConfigs;

    /**
     * Params for sub steps
     */
    protected Map<String, Object> params;

    public BasicModelProcessor() {
    }

    public BasicModelProcessor(Map<String, Object> otherConfigs) {
        this.otherConfigs = otherConfigs;
    }

    public BasicModelProcessor(ModelConfig modelConfig, List<ColumnConfig> columnConfigList,
            Map<String, Object> otherConfigs) {
        this.modelConfig = modelConfig;
        this.columnConfigList = columnConfigList;
        this.otherConfigs = otherConfigs;
        this.pathFinder = new PathFinder(modelConfig, otherConfigs);
    }

    /**
     * initialize the config file, pathFinder and other input
     * 
     * @param step
     *            Shifu running step
     * @throws Exception
     *             any exception in setup
     */
    protected void setUp(ModelStep step) throws Exception {
        if(hasInitialized()) {
            return;
        }

        // load model configuration and do validation
        loadModelConfig();
        validateModelConfig(step);

        this.pathFinder = new PathFinder(modelConfig, this.getOtherConfigs());
        checkAlgorithmParam();

        LOG.info(String.format("Training Data Soure Location: %s", modelConfig.getDataSet().getSource()));
        switch(step) {
            case INIT:
                break;
            default:
                loadColumnConfig();
                validateColumnConfig();

                // if in stats but stats -c or stats -p or stats -rebin, column update should be called because of
                // such stats steps should all be called after 'shifu stats', this is actually to call VoidUpdater
                boolean strictCallVoidUpdate = (step == ModelStep.STATS)
                        && (getBooleanParam(this.params, Constants.IS_COMPUTE_CORR)
                                || getBooleanParam(this.params, Constants.IS_COMPUTE_PSI)
                                || getBooleanParam(this.params, Constants.IS_REBIN));

                // update ColumnConfig and save to disk
                ColumnConfigUpdater.updateColumnConfigFlags(modelConfig, columnConfigList, step, strictCallVoidUpdate);

                validateColumnConfigAfterSet();

                saveColumnConfigList();
                break;
        }

        // validate
        switch(step) {
            case NORMALIZE:
            case VARSELECT:
            case TRAIN:
            case EVAL:
                List<String> segs = this.modelConfig.getSegmentFilterExpressions();
                String alg = this.modelConfig.getAlgorithm();
                if(segs.size() > 0 && !(CommonUtils.isNNModel(alg) || CommonUtils.isLRModel(alg))) {
                    throw new IllegalArgumentException(
                            "Segment expression is only supported in NN or LR model, please check train:algrithm setting in ModelConfig.json.");
                }
                break;
            default:
                break;
        }
    }

    private void validateColumnConfigAfterSet() {
        if(this.columnConfigList == null) {
            return;
        }
        NormType normType = this.modelConfig.getNormalizeType();

        for(ColumnConfig config: this.columnConfigList) {
            if(config.isHybrid() && !modelConfig.isRegression()) {
                throw new IllegalArgumentException("Hybrid column " + config.getColumnName()
                        + " is found, but only supported in regression mode, not classfication mode.");
            }
            if(config.isHybrid() && !normType.isWoe()) {
                throw new IllegalArgumentException("Hybrid column " + config.getColumnName()
                        + " is found, but not woe norm type, please set norm#normType to woe related.");
            }
        }
    }

    private void validateColumnConfig() {
        if(this.columnConfigList == null) {
            return;
        }
        Set<NSColumn> names = new HashSet<NSColumn>();
        for(ColumnConfig config: this.columnConfigList) {
            if(StringUtils.isEmpty(config.getColumnName())) {
                throw new IllegalArgumentException("Empty column name, please check your header file.");
            }
            if(names.contains(new NSColumn(config.getColumnName()))) {
                LOG.warn("Duplicated {} in ColumnConfig.json file, later one will be append index to make it unique.",
                        config.getColumnName());
            }
            names.add(new NSColumn(config.getColumnName()));
        }

        if(!names.contains(new NSColumn(modelConfig.getTargetColumnName()))) {
            throw new IllegalArgumentException(
                    "target column " + modelConfig.getTargetColumnName() + " does not exist.");
        }

        if(StringUtils.isNotBlank(modelConfig.getWeightColumnName())
                && !names.contains(new NSColumn(modelConfig.getWeightColumnName()))) {
            throw new IllegalArgumentException(
                    "weight column " + modelConfig.getWeightColumnName() + " does not exist.");
        }
    }

    /**
     * The post-logic after running
     * <p>
     * copy file to hdfs if SourceType is HDFS
     * 
     * @param step
     *            Shifu running step
     * @throws IOException
     *             if any problem happen in copying files to HDFS
     */
    protected void clearUp(ModelStep step) throws IOException {
        // do nothing now
    }

    /**
     * save Model Config
     * 
     * @throws IOException
     *             an exception in saving model config
     */
    public void saveModelConfig() throws IOException {
        LOG.info("Saving ModelConfig...");
        JSONUtils.writeValue(new File(pathFinder.getModelConfigPath(SourceType.LOCAL)), modelConfig);
    }

    /**
     * save the Column Config
     * 
     * @throws IOException
     *             an exception in saving column config
     */
    public void saveColumnConfigList() throws IOException {
        LOG.info("Saving ColumnConfig...");
        JSONUtils.writeValue(new File(pathFinder.getColumnConfigPath(SourceType.LOCAL)), columnConfigList);
    }

    /**
     * validate the modelconfig if it's well written.
     * 
     * @param modelStep
     *            the model step
     * @throws Exception
     *             any exception in validation
     */
    protected void validateModelConfig(ModelStep modelStep) throws Exception {
        ValidateResult result = new ValidateResult(false);

        if(modelConfig == null) {
            result.getCauses().add("The ModelConfig is not loaded!");
        } else {
            result = ModelInspector.getInspector().probe(modelConfig, modelStep);
        }

        if(!result.getStatus()) {
            LOG.error("ModelConfig Validation - Fail! See below:");
            for(String cause: result.getCauses()) {
                LOG.error("\t!!! " + cause);
            }

            throw new ShifuException(ShifuErrorCode.ERROR_MODELCONFIG_NOT_VALIDATION);
        } else {
            LOG.info("ModelConfig Validation - OK");
        }
    }

    /**
     * Close all scanners
     * 
     * @param scanners
     *            the scanners
     */
    public void closeScanners(List<Scanner> scanners) {
        if(CollectionUtils.isNotEmpty(scanners)) {
            for(Scanner scanner: scanners) {
                scanner.close();
            }
        }
    }

    /**
     * Sync data into HDFS for list of EvalConfig,
     * 
     * @param evalConfigList
     *            - EvalConfig list to sync
     * @return true if synced to HDFS
     * @throws IOException
     *             if exception in creating eval folder
     */
    protected boolean syncDataToHdfs(List<EvalConfig> evalConfigList) throws IOException {
        if(CollectionUtils.isNotEmpty(evalConfigList)) {
            // create local eval folder
            for(EvalConfig evalConfig: evalConfigList) {
                String evalSetPath = pathFinder.getEvalSetPath(evalConfig, SourceType.LOCAL);
                FileUtils.forceMkdir(new File(evalSetPath));
            }

            // sync files to HDFS
            for(EvalConfig evalConfig: evalConfigList) {
                if(SourceType.HDFS.equals(evalConfig.getDataSet().getSource())) {
                    return syncDataToHdfs(evalConfig.getDataSet().getSource());
                }
            }
        }
        return false;
    }

    /**
     * Sync data into HDFS if necessary:
     * RunMode == pig and SourceType == HDFS
     * 
     * @param sourceType
     *            source type
     * @return if sync in hdfs or not
     * @throws IOException
     *             any exception in file system io
     */
    public boolean syncDataToHdfs(SourceType sourceType) throws IOException {
        if(SourceType.HDFS.equals(sourceType)) {
            CommonUtils.copyConfFromLocalToHDFS(modelConfig, this.pathFinder);
            return true;
        }

        return false;
    }

    public void copyModelFiles(String sourcePath, String targetPath) throws IOException {
        loadModelConfig(sourcePath + File.separator + "ModelConfig.json", SourceType.LOCAL);
        File targetFile = new File(targetPath);

        this.modelConfig.setModelSetName(targetFile.getName());
        this.modelConfig.setModelSetCreator(Environment.getProperty(Environment.SYSTEM_USER));

        try {
            JSONUtils.writeValue(new File(targetPath + File.separator + "ModelConfig.json"), modelConfig);
        } catch (IOException e) {
            throw new ShifuException(ShifuErrorCode.ERROR_WRITE_MODELCONFIG, e);
        }
    }

    /**
     * get the modelConfig instance
     * 
     * @return the modelConfig
     */
    public ModelConfig getModelConfig() {
        return modelConfig;
    }

    /**
     * get the columnConfigList instance
     * 
     * @return the columnConfigList
     */
    public List<ColumnConfig> getColumnConfigList() {
        return columnConfigList;
    }

    /**
     * get the pathFinder instance
     * 
     * @return the pathFinder
     */
    public PathFinder getPathFinder() {
        return pathFinder;
    }

    /**
     * check algorithm parameter
     * 
     * @throws Exception
     *             modelConfig is not loaded or save ModelConfig.json file error
     */
    public void checkAlgorithmParam() throws Exception {

        String alg = modelConfig.getAlgorithm();
        Map<String, Object> param = modelConfig.getParams();

        if(alg.equalsIgnoreCase("LR")) {
            if(!param.containsKey("LearningRate")) {
                param = new LinkedHashMap<String, Object>();
                param.put("LearningRate", 0.1);
                modelConfig.setParams(param);
                saveModelConfig();
            }
        } else if(alg.equalsIgnoreCase("NN")) {
            if(!param.containsKey("Propagation")) {
                param = new LinkedHashMap<String, Object>();

                param.put("Propagation", "R");
                param.put("LearningRate", 0.1);
                param.put("NumHiddenLayers", 2);

                List<Integer> nodes = new ArrayList<Integer>();
                nodes.add(20);
                nodes.add(10);
                param.put("NumHiddenNodes", nodes);

                List<String> func = new ArrayList<String>();
                func.add("tanh");
                func.add("tanh");
                param.put("ActivationFunc", func);

                modelConfig.setParams(param);
                saveModelConfig();
            }

        } else if(alg.equalsIgnoreCase("SVM")) {
            if(!param.containsKey("Kernel")) {
                param = new LinkedHashMap<String, Object>();

                param.put("Kernel", "linear");
                param.put("Gamma", 1.);
                param.put("Const", 1.);

                modelConfig.setParams(param);
                saveModelConfig();
            }
        } else if(alg.equalsIgnoreCase("DT")) {
            // do nothing
        } else if(alg.equalsIgnoreCase("RF")) {
            if(!param.containsKey("MaxDepth")) {
                param = new LinkedHashMap<String, Object>();

                param.put("TreeNum", 10);
                param.put("FeatureSubsetStrategy", "TWOTHIRDS");
                param.put("MaxDepth", 14);
                param.put("MinInstancesPerNode", 1);
                param.put("MinInfoGain", 0.0);
                param.put("Impurity", "entropy");
                param.put("Loss", "squared");

                modelConfig.setParams(param);
                saveModelConfig();
            }
        } else if(alg.equalsIgnoreCase("GBT")) {
            if(!param.containsKey("MaxDepth")) {
                param = new LinkedHashMap<String, Object>();

                param.put("TreeNum", "100");
                param.put("FeatureSubsetStrategy", "TWOTHIRDS");
                param.put("MaxDepth", 7);
                param.put("MinInstancesPerNode", 5);
                param.put("MinInfoGain", 0.0);
                param.put("DropoutRate", 0.0);
                param.put("Impurity", "variance");
                param.put(CommonConstants.LEARNING_RATE, 0.05);
                param.put("Loss", "squared");
                modelConfig.setParams(param);
                modelConfig.getTrain().setNumTrainEpochs(10000);
                saveModelConfig();
            }
        } else {
            throw new ShifuException(ShifuErrorCode.ERROR_UNSUPPORT_ALG);
        }

        // log.info("Finished: check the algorithm parameter");
    }

    /**
     * load Model Config method
     * 
     * @throws IOException
     *             in load model config
     */
    private void loadModelConfig() throws IOException {
        modelConfig = CommonUtils.loadModelConfig(
                new Path(CommonUtils.getLocalModelSetPath(otherConfigs), Constants.LOCAL_MODEL_CONFIG_JSON).toString(),
                SourceType.LOCAL);
    }

    /**
     * load Model Config method
     * 
     * @throws IOException
     *             in load model config
     */
    private void loadModelConfig(String pathToModel, SourceType source) throws IOException {
        modelConfig = CommonUtils.loadModelConfig(pathToModel, source);
    }

    /**
     * load Column Config
     * 
     * @throws IOException
     *             in load column config
     */
    private void loadColumnConfig() throws IOException {
        columnConfigList = CommonUtils.loadColumnConfigList(
                new Path(CommonUtils.getLocalModelSetPath(otherConfigs), Constants.LOCAL_COLUMN_CONFIG_JSON).toString(),
                SourceType.LOCAL, false);
    }

    /**
     * Check the processor is initialized or not
     * 
     * @return true - if the process is initialized
     *         false - if not
     */
    private boolean hasInitialized() {
        return (null != this.modelConfig && null != this.columnConfigList && null != this.pathFinder);
    }

    /**
     * create HEAD file contain the workspace
     * 
     * @param modelName
     *            model name
     * @throws IOException
     *             any exception in create header
     */
    protected void createHead(String modelName) throws IOException {
        File header = new File(modelName == null ? "" : modelName + "/.HEAD");
        if(header.exists()) {
            LOG.error("File {} already exist.", header.getAbsolutePath());
            return;
        }

        BufferedWriter writer = null;
        try {
            writer = new BufferedWriter(
                    new OutputStreamWriter(new FileOutputStream(header), Constants.DEFAULT_CHARSET));
            writer.write("master");
        } catch (IOException e) {
            LOG.error("Fail to create HEAD file to store the current workspace");
        } finally {
            if(writer != null) {
                writer.close();
            }
        }
    }

    /**
     * @return the otherConfigs
     */
    public Map<String, Object> getOtherConfigs() {
        return otherConfigs;
    }

    /**
     * @param otherConfigs
     *            the otherConfigs to set
     */
    public void setOtherConfigs(Map<String, Object> otherConfigs) {
        this.otherConfigs = otherConfigs;
    }

    protected void runDataClean(boolean isToShuffle) throws IOException {
        SourceType sourceType = modelConfig.getDataSet().getSource();
        String cleanedDataPath = this.pathFinder.getCleanedDataPath();

        LOG.info("Start to generate clean data for tree model ... ");
        if(ShifuFileUtils.isFileExists(cleanedDataPath, sourceType)) {
            ShifuFileUtils.deleteFile(cleanedDataPath, sourceType);
        }

        Map<String, String> paramsMap = new HashMap<String, String>();
        paramsMap.put("sampleRate", modelConfig.getNormalizeSampleRate().toString());
        paramsMap.put("sampleNegOnly", ((Boolean) modelConfig.isNormalizeSampleNegOnly()).toString());
        paramsMap.put("delimiter", CommonUtils.escapePigString(modelConfig.getDataSetDelimiter()));

        try {
            String normPigPath = pathFinder.getScriptPath("scripts/Normalize.pig");
            paramsMap.put(Constants.IS_COMPRESS, "true");
            paramsMap.put(Constants.IS_NORM_FOR_CLEAN, "true");
            paramsMap.put(Constants.PATH_NORMALIZED_DATA, pathFinder.getCleanedDataPath());
            PigExecutor.getExecutor().submitJob(modelConfig, normPigPath, paramsMap, sourceType, this.pathFinder);
            // cleaned validation data
            if(StringUtils.isNotBlank(modelConfig.getValidationDataSetRawPath())) {
                String cleandedValidationDataPath = pathFinder.getCleanedValidationDataPath();
                if(ShifuFileUtils.isFileExists(cleandedValidationDataPath, sourceType)) {
                    ShifuFileUtils.deleteFile(cleandedValidationDataPath, sourceType);
                }
                paramsMap.put(Constants.IS_COMPRESS, "false");
                paramsMap.put(Constants.PATH_RAW_DATA, modelConfig.getValidationDataSetRawPath());
                paramsMap.put(Constants.PATH_NORMALIZED_DATA, pathFinder.getCleanedValidationDataPath());
                PigExecutor.getExecutor().submitJob(modelConfig, normPigPath, paramsMap, sourceType, this.pathFinder);
            }
        } catch (IOException e) {
            throw new ShifuException(ShifuErrorCode.ERROR_RUNNING_PIG_JOB, e);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }

        if(isToShuffle) {
            MapReduceShuffle shuffler = new MapReduceShuffle(this.modelConfig);
            try {
                shuffler.run(pathFinder.getCleanedDataPath());
            } catch (ClassNotFoundException e) {
                throw new RuntimeException("Fail to shuffle the cleaned data.", e);
            } catch (InterruptedException e) {
                throw new RuntimeException("Fail to shuffle the cleaned data.", e);
            }
        }
        LOG.info("Generate clean data for tree model successful.");
    }

    /**
     * Save ModelConfig into some folder
     * 
     * @param folder
     *            - folder to host ModelConfig.json
     * @param modelConfig
     *            model config instance
     * @throws IOException
     *             any io exception
     */
    protected void saveModelConfig(String folder, ModelConfig modelConfig) throws IOException {
        JSONUtils.writeValue(new File(folder + File.separator + Constants.MODEL_CONFIG_JSON_FILE_NAME), modelConfig);
    }

    /**
     * save the Column Config
     * 
     * @param path
     *            path of column config file
     * @param columnConfigs
     *            column config list
     * @throws IOException
     *             an exception in saving column config
     */

    protected void saveColumnConfigList(String path, List<ColumnConfig> columnConfigs) throws IOException {
        LOG.info("Saving ColumnConfig into {} ... ", path);
        JSONUtils.writeValue(new File(path), columnConfigs);
    }

    protected boolean isRequestColumn(List<String> catVariables, ColumnConfig columnConfig) {
        boolean status = false;
        for(String varName: catVariables) {
            if(NSColumnUtils.isColumnEqual(varName, columnConfig.getColumnName())) {
                status = true;
                break;
            }
        }
        return status;
    }

    protected boolean getBooleanParam(Map<String, Object> params, String propKey) {
        if(MapUtils.isNotEmpty(params) && params.get(propKey) instanceof Boolean) {
            return (Boolean) params.get(propKey);
        }
        return false;
    }

    protected List<String> getStringList(Map<String, Object> params, String propKey, String delimiter) {
        if(MapUtils.isNotEmpty(params) && params.get(propKey) instanceof String) {
            String propVal = (String) params.get(propKey);
            if(StringUtils.isNotBlank(propVal)) {
                return Arrays.asList(propVal.split(","));
            }
        }
        return null;
    }

    protected int getIntParam(Map<String, Object> params, String propKey, int defval) {
        if(MapUtils.isNotEmpty(params) && params.get(propKey) instanceof String) {
            String propVal = (String) params.get(propKey);
            try {
                return Integer.parseInt(propVal);
            } catch (Exception e) {
                LOG.warn("Invalid int value for {}. Ignore it...", propKey);
            }
        }
        return defval;
    }

    protected int getIntParam(Map<String, Object> params, String propKey) {
        return getIntParam(params, propKey, 0);
    }

    protected double getDoubleParam(Map<String, Object> params, String propKey, double defval) {
        if(MapUtils.isNotEmpty(params) && params.get(propKey) instanceof String) {
            String propVal = (String) params.get(propKey);
            try {
                return Double.parseDouble(propVal);
            } catch (Exception e) {
                LOG.warn("Invalid double value for {}. Ignore it...", propKey);
            }
        }
        return defval;
    }

    protected long getLongParam(Map<String, Object> params, String propKey) {
        if(MapUtils.isNotEmpty(params) && params.get(propKey) instanceof String) {
            String propVal = (String) params.get(propKey);
            try {
                return Long.parseLong(propVal);
            } catch (Exception e) {
                LOG.warn("Invalid long value for {}. Ignore it...", propKey);
            }
        }
        return 0;
    }

}
