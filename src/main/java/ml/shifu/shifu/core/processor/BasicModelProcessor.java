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
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;

import ml.shifu.shifu.container.meta.ValidateResult;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.validator.ModelInspector;
import ml.shifu.shifu.core.validator.ModelInspector.ModelStep;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.fs.PathFinder;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.Environment;
import ml.shifu.shifu.util.JSONUtils;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Model Basic Processor, it helps to do basic manipulate in model, including load/save configuration, copy
 * configuration file
 */
public class BasicModelProcessor {

    private final static Logger log = LoggerFactory.getLogger(BasicModelProcessor.class);

    protected ModelConfig modelConfig;
    protected List<ColumnConfig> columnConfigList;
    protected PathFinder pathFinder;

    /**
     * If not specified SHIFU_HOME env, some key configurations like pig path or lib path can be configured here
     */
    protected Map<String, Object> otherConfigs;

    public BasicModelProcessor() {
    }

    public BasicModelProcessor(Map<String, Object> otherConfigs) {
        this.otherConfigs = otherConfigs;
    }

    public BasicModelProcessor(ModelConfig modelConfig, List<ColumnConfig> columnConfigList,
            Map<String, Object> otherConfigs) {
        this.modelConfig = modelConfig;
        this.columnConfigList = columnConfigList;
        this.setOtherConfigs(otherConfigs);
        this.pathFinder = new PathFinder(modelConfig, otherConfigs);
    }

    /**
     * initialize the config file, pathFinder and other input
     * 
     * @param step
     *            Shifu running step
     * @throws Exception
     */
    protected void setUp(ModelStep step) throws Exception {
        if(hasInitialized()) {
            return;
        }

        // load model configuration and do validation
        loadModelConfig();
        validateModelConfig(step);

        pathFinder = new PathFinder(modelConfig, this.getOtherConfigs());

        checkAlgorithmParam();

        log.info(String.format("Training Data Soure Location: %s", modelConfig.getDataSet().getSource()));

        switch(step) {
            case INIT:
                break;
            default:
                loadColumnConfig();
                validateColumnConfig();
                break;
        }
    }

    private void validateColumnConfig() {
        if(this.columnConfigList == null) {
            return;
        }
        Set<String> names = new HashSet<String>();
        for(ColumnConfig config: this.columnConfigList) {
            if(StringUtils.isEmpty(config.getColumnName())) {
                throw new IllegalArgumentException("Empry column name, please check your header file.");
            }
            if(names.contains(config.getColumnName())) {
                log.warn("Duplicated {} in ColumnConfig.json file, later one will be append index to make it unique.",
                        config.getColumnName());
            }
            names.add(config.getColumnName());
        }

        if(!names.contains(modelConfig.getTargetColumnName())) {
            throw new IllegalArgumentException("target column " + modelConfig.getTargetColumnName()
                    + " does not exist.");
        }

        if(StringUtils.isNotBlank(modelConfig.getWeightColumnName())
                && !names.contains(modelConfig.getWeightColumnName())) {
            throw new IllegalArgumentException("weight column " + modelConfig.getWeightColumnName()
                    + " does not exist.");
        }
    }

    /**
     * The post-logic after running
     * </p>
     * copy file to hdfs if SourceType is HDFS
     * </p>
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
     */
    public void saveModelConfig() throws IOException {
        log.info("Saving ModelConfig...");
        JSONUtils.writeValue(new File(pathFinder.getModelConfigPath(SourceType.LOCAL)), modelConfig);
    }

    /**
     * save the Column Config
     * 
     * @throws IOException
     */
    public void saveColumnConfigListAndColumnStats(boolean columnStats) throws IOException {
        log.info("Saving ColumnConfig...");
        JSONUtils.writeValue(new File(pathFinder.getColumnConfigPath(SourceType.LOCAL)), columnConfigList);
    }

    /**
     * validate the modelconfig if it's well written.
     * 
     * @return
     * @throws Exception
     */
    protected void validateModelConfig(ModelStep modelStep) throws Exception {
        ValidateResult result = new ValidateResult(false);

        if(modelConfig == null) {
            result.getCauses().add("The ModelConfig is not loaded!");
        } else {
            result = ModelInspector.getInspector().probe(modelConfig, modelStep);
        }

        if(!result.getStatus()) {
            log.error("ModelConfig Validation - Fail! See below:");
            for(String cause: result.getCauses()) {
                log.error("\t!!! " + cause);
            }

            throw new ShifuException(ShifuErrorCode.ERROR_MODELCONFIG_NOT_VALIDATION);
        } else {
            log.info("ModelConfig Validation - OK");
        }
    }

    /**
     * Close all scanners
     * 
     * @param scanners
     */
    public void closeScanners(List<Scanner> scanners) {
        if(CollectionUtils.isNotEmpty(scanners)) {
            for(Scanner scanner: scanners) {
                scanner.close();
            }
        }
    }

    /**
     * Sync data into HDFS if necessary:
     * RunMode == pig && SourceType == HDFS
     * 
     * @param sourceType
     * @return
     * @throws IOException
     */
    public boolean syncDataToHdfs(SourceType sourceType) throws IOException {
        if(SourceType.HDFS.equals(sourceType)) {
            CommonUtils.copyConfFromLocalToHDFS(modelConfig);
            return true;
        }

        return false;
    }

    /**
     * copy model configuration file
     * 
     * @param sourcePath
     * @param targetPath
     * @throws IOException
     */
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
     *             </p>
     *             modelConfig is not loaded or</p>
     *             save ModelConfig.json file error </p>
     */
    public void checkAlgorithmParam() throws Exception {

        String alg = modelConfig.getAlgorithm();
        Map<String, Object> param = modelConfig.getParams();
        log.info("Check algorithm parameter");

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

                param.put("Propagation", "Q");
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
            if(!param.containsKey("FeatureSubsetStrategy")) {
                param = new LinkedHashMap<String, Object>();

                param.put("FeatureSubsetStrategy", "all");
                param.put("MaxDepth", 10);
                param.put("MaxStatsMemoryMB", 256);
                param.put("Impurity", "entropy");

                modelConfig.setParams(param);
                saveModelConfig();
            }
        } else if(alg.equalsIgnoreCase("GBT")) {
            if(!param.containsKey("FeatureSubsetStrategy")) {
                param = new LinkedHashMap<String, Object>();

                param.put("FeatureSubsetStrategy", "all");
                param.put("MaxDepth", 10);
                param.put("MaxStatsMemoryMB", 256);
                param.put("Impurity", "entropy");
                param.put("Loss", "squared");

                modelConfig.setParams(param);
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
     */
    private void loadModelConfig() throws IOException {
        modelConfig = CommonUtils.loadModelConfig();
    }

    /**
     * load Model Config method
     * 
     * @throws IOException
     */
    private void loadModelConfig(String pathToModel, SourceType source) throws IOException {
        modelConfig = CommonUtils.loadModelConfig(pathToModel, source);
    }

    /**
     * load Column Config
     * 
     * @throws IOException
     */
    private void loadColumnConfig() throws IOException {
        columnConfigList = CommonUtils.loadColumnConfigList();
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
     * @throws IOException
     */
    protected void createHead(String modelName) throws IOException {
        File header = new File(modelName == null ? "" : modelName + "/.HEAD");
        if(header.exists()) {
            log.error("File {} already exist.", header.getAbsolutePath());
            return;
        }

        BufferedWriter writer = null;
        try {
            writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(header), Constants.DEFAULT_CHARSET));
            writer.write("master");
        } catch (IOException e) {
            log.error("Fail to create HEAD file to store the current workspace");
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

}
