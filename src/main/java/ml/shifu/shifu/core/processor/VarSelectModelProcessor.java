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

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.collections.ListUtils;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.jexl2.JexlException;
import org.apache.commons.lang.StringUtils;
import org.apache.commons.lang3.tuple.MutablePair;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.map.MultithreadedMapper;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.pig.data.Tuple;
import org.apache.pig.impl.util.JarManager;
import org.apache.zookeeper.ZooKeeper;
import org.encog.ml.BasicML;
import org.encog.ml.data.MLDataSet;
import org.jboss.netty.bootstrap.ServerBootstrap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.base.Splitter;

import ml.shifu.guagua.GuaguaConstants;
import ml.shifu.guagua.hadoop.util.HDPUtils;
import ml.shifu.guagua.mapreduce.GuaguaMapReduceClient;
import ml.shifu.guagua.mapreduce.GuaguaMapReduceConstants;
import ml.shifu.shifu.column.NSColumn;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ColumnConfig.ColumnFlag;
import ml.shifu.shifu.container.obj.ModelTrainConf;
import ml.shifu.shifu.container.obj.ModelVarSelectConf.PostCorrelationMetric;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.VariableSelector;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.dataset.BasicFloatNetwork;
import ml.shifu.shifu.core.dtrain.dt.IndependentTreeModel;
import ml.shifu.shifu.core.dtrain.nn.NNConstants;
import ml.shifu.shifu.core.dvarsel.VarSelMaster;
import ml.shifu.shifu.core.dvarsel.VarSelMasterResult;
import ml.shifu.shifu.core.dvarsel.VarSelOutput;
import ml.shifu.shifu.core.dvarsel.VarSelWorker;
import ml.shifu.shifu.core.dvarsel.VarSelWorkerResult;
import ml.shifu.shifu.core.dvarsel.wrapper.CandidateGenerator;
import ml.shifu.shifu.core.dvarsel.wrapper.WrapperMasterConductor;
import ml.shifu.shifu.core.dvarsel.wrapper.WrapperWorkerConductor;
import ml.shifu.shifu.core.history.VarSelDesc;
import ml.shifu.shifu.core.history.VarSelReason;
import ml.shifu.shifu.core.mr.input.CombineInputFormat;
import ml.shifu.shifu.core.validator.ModelInspector.ModelStep;
import ml.shifu.shifu.core.varselect.ColumnInfo;
import ml.shifu.shifu.core.varselect.ColumnScore;
import ml.shifu.shifu.core.varselect.ColumnStatistics;
import ml.shifu.shifu.core.varselect.VarSelectMapper;
import ml.shifu.shifu.core.varselect.VarSelectReducer;
import ml.shifu.shifu.core.varselect.VarSelectSCMapper;
import ml.shifu.shifu.core.varselect.VarSelectSCReducer;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.fs.PathFinder;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.fs.SourceFile;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.Environment;
import ml.shifu.shifu.util.HdfsPartFile;
import ml.shifu.shifu.util.ModelSpecLoaderUtils;
import ml.shifu.shifu.util.ValueVisitor;

/**
 * Variable selection processor, select the variable based on KS/IV value, or
 * 
 * <p>
 * Selection variable based on the wrapper training processor.
 * 
 * <p>
 * For sensitive variable selection, each time wrapperRatio percent of variables will be removed. If continue do
 * variable selection, continue to run varselect command. Current design will do variable selection continuously.
 */
public class VarSelectModelProcessor extends BasicModelProcessor implements Processor {

    private final static Logger LOG = LoggerFactory.getLogger(VarSelectModelProcessor.class);

    @SuppressWarnings("unused")
    private static final double BAD_IV_THRESHOLD = 0.02d;

    private static final String TRAIN_LOG_PREFIX = "vs-train";

    /**
     * SE stats mao for correlation variable selection,if not se, this field will be null.
     */
    private Map<Integer, ColumnStatistics> seStatsMap;

    public VarSelectModelProcessor() {
        // default constructor
    }

    public VarSelectModelProcessor(Map<String, Object> otherConfigs) {
        super.otherConfigs = otherConfigs;
    }

    /**
     * Run for the variable selection
     */
    @Override
    public int run() throws Exception {
        LOG.info("Step Start: varselect");
        long start = System.currentTimeMillis();
        try {
            setUp(ModelStep.VARSELECT);
            validateParameters();

            // reset all selections if user specify or select by absolute number
            if(getIsToReset()) {
                LOG.info("Reset all selections data including type final select etc!");
                if(this.modelConfig.isMultiTask()) {
                    for(int i = 0; i < this.mtlColumnConfigLists.size(); i++) {
                        List<ColumnConfig> ccList = this.mtlColumnConfigLists.get(i);
                        resetAllFinalSelect(ccList);
                        saveColumnConfigList(pathFinder.getMTLColumnConfigPath(SourceType.LOCAL, i), ccList);
                    }
                } else {
                    resetAllFinalSelect(this.columnConfigList);
                    saveColumnConfigList(this.columnConfigList);
                }
            } else if(getIsToList()) {
                LOG.info("Below variables are selected - ");
                if(this.modelConfig.isMultiTask()) {
                    for(int i = 0; i < this.mtlColumnConfigLists.size(); i++) {
                        List<ColumnConfig> ccList = this.mtlColumnConfigLists.get(i);
                        for(ColumnConfig columnConfig: ccList) {
                            if(columnConfig.isFinalSelect()) {
                                LOG.info("MTL {} selected variable {}", i, columnConfig.getColumnName());
                            }
                        }
                    }
                } else {
                    for(ColumnConfig columnConfig: this.columnConfigList) {
                        if(columnConfig.isFinalSelect()) {
                            LOG.info(columnConfig.getColumnName());
                        }
                    }
                }
                LOG.info("-----  Done -----");
            } else if(getIsToAutoFilter()) {
                LOG.info("Start to run variable auto filter.");
                runAutoVarFilter(this.columnConfigList);
                LOG.info("-----  Done -----");
            } else if(getIsRecoverAuto()) {
                String varselHistory = pathFinder.getVarSelHistory();
                if(ShifuFileUtils.isFileExists(varselHistory, SourceType.LOCAL)) {
                    LOG.info("!!! Auto filtered variables will be recovered from history.");
                    recoverVarselStatusFromHist(varselHistory);
                    LOG.info("-----  Done -----");
                } else {
                    LOG.warn("No variables auto filter history is found.");
                }
            } else if(getVarSelFile() != null) {
                String varselFile = getVarSelFile();
                Set<String> toSelectColumnSet = guessSelectColumnSet(varselFile);
                int selectVarsCnt = 0;
                if(CollectionUtils.isNotEmpty(toSelectColumnSet)) {
                    for(ColumnConfig columnConfig: this.columnConfigList) {
                        // reset firstly
                        columnConfig.setFinalSelect(false);

                        // select if specified
                        if(toSelectColumnSet.contains(columnConfig.getColumnName())) {
                            LOG.info("variable {} is selected.", columnConfig.getColumnName());
                            columnConfig.setFinalSelect(true);
                            selectVarsCnt++;
                        }
                    }
                    LOG.info("Totally, there are {} variables are selected based on {}", selectVarsCnt, varselFile);
                } else {
                    LOG.warn("Illegal file  - {}, or there is no variables in it.", varselFile);
                }
            } else {
                // sync to make sure load from hdfs config is consistent with local configuration
                syncDataToHdfs(super.modelConfig.getDataSet().getSource());

                String filterExpressions = super.modelConfig.getSegmentFilterExpressionsAsString();
                Environment.getProperties().put("shifu.segment.expressions", filterExpressions);
                if(StringUtils.isNotBlank(filterExpressions)) {
                    String[] splits = CommonUtils.split(filterExpressions,
                            Constants.SHIFU_STATS_FILTER_EXPRESSIONS_DELIMETER);
                    for(int i = 0; i < super.columnConfigList.size(); i++) {
                        ColumnConfig config = super.columnConfigList.get(i);
                        int rawSize = super.columnConfigList.size() / (1 + splits.length);
                        if(config.isTarget()) {
                            for(int j = 0; j < splits.length; j++) {
                                ColumnConfig otherConfig = super.columnConfigList.get((j + 1) * rawSize + i);
                                otherConfig.setColumnFlag(ColumnFlag.ForceRemove);
                                otherConfig.setFinalSelect(false);
                            }
                            break;
                        }
                    }

                    this.saveColumnConfigList();
                    // sync to make sure load from hdfs config is consistent with local configuration
                    syncDataToHdfs(super.modelConfig.getDataSet().getSource());
                }

                if(modelConfig.isRegression() || this.isLinearSEorST()) {
                    String filterBy = this.modelConfig.getVarSelectFilterBy();
                    if(filterBy.equalsIgnoreCase(Constants.FILTER_BY_KS)
                            || filterBy.equalsIgnoreCase(Constants.FILTER_BY_IV)
                            || filterBy.equalsIgnoreCase(Constants.FILTER_BY_PARETO)
                            || filterBy.equalsIgnoreCase(Constants.FILTER_BY_MIX)) {
                        if(this.modelConfig.isMultiTask()) {
                            for(int i = 0; i < this.mtlColumnConfigLists.size(); i++) {
                                List<ColumnConfig> ccList = this.mtlColumnConfigLists.get(i);
                                VariableSelector selector = new VariableSelector(this.modelConfig, ccList);
                                ccList = selector.selectByFilter();
                                saveColumnConfigList(pathFinder.getMTLColumnConfigPath(SourceType.LOCAL, i), ccList);
                            }
                        } else {
                            VariableSelector selector = new VariableSelector(this.modelConfig, this.columnConfigList);
                            this.columnConfigList = selector.selectByFilter();
                            this.saveColumnConfigList(this.columnConfigList);
                        }
                    } else if(filterBy.equalsIgnoreCase(Constants.FILTER_BY_FI)) {
                        if(!CommonUtils.isTreeModel(modelConfig.getAlgorithm())) {
                            throw new IllegalArgumentException(
                                    "Filter by FI only works well in GBT/RF. Please check your modelconfig::train.");
                        }
                        selectByFeatureImportance();
                    } else if(filterBy.equalsIgnoreCase(Constants.FILTER_BY_SE)
                            || filterBy.equalsIgnoreCase(Constants.FILTER_BY_ST)
                            || filterBy.equalsIgnoreCase(Constants.FILTER_BY_SC)) {
                        if(!Constants.NN.equalsIgnoreCase(modelConfig.getAlgorithm())
                                && !Constants.LR.equalsIgnoreCase(modelConfig.getAlgorithm())
                                && !CommonUtils.isTensorFlowModel(modelConfig.getAlgorithm())) {
                            throw new IllegalArgumentException(
                                    "Filter by SE/ST only works well in NN/LR/TensorFlow. Please check your modelconfig::train.");
                        }
                        int recursiveCnt = getRecursiveCnt();
                        int i = 0;
                        // create varsel directory and write original copy of ColumnConfig.json
                        ShifuFileUtils.createDirIfNotExists(pathFinder.getVarSelDir(), SourceType.LOCAL);
                        super.saveColumnConfigList(pathFinder.getVarSelColumnConfig(i), this.columnConfigList);
                        while((i++) < recursiveCnt) {
                            String trainLogFile = TRAIN_LOG_PREFIX + "-" + (i - 1) + ".log";
                            distributedSEWrapper(trainLogFile);
                            // copy training log to SE train.log
                            ShifuFileUtils.move(trainLogFile,
                                    new File(pathFinder.getVarSelDir(), trainLogFile).getPath(), SourceType.LOCAL);
                            // copy models to varsel directory
                            copyModelSpec(pathFinder.getModelsPath(SourceType.LOCAL), pathFinder.getVarSelDir(),
                                    (i - 1));

                            String varSelectMSEOutputPath = pathFinder
                                    .getVarSelectMSEOutputPath(modelConfig.getDataSet().getSource());
                            // even fail to run SE, still to create an empty se.x file
                            String varSelMSEHistPath = pathFinder.getVarSelMSEHistPath(i - 1);
                            ShifuFileUtils.createFileIfNotExists(varSelMSEHistPath, SourceType.LOCAL);
                            if(filterBy.equalsIgnoreCase(Constants.FILTER_BY_SC)) {
                                ShifuFileUtils.copyToLocal(
                                        new SourceFile(varSelectMSEOutputPath, modelConfig.getDataSet().getSource()),
                                        Constants.HADOOP_PART_PREFIX, varSelMSEHistPath);
                                ShifuFileUtils.sortFile(varSelMSEHistPath, "\t", 2, true);
                            } else {
                                ShifuFileUtils.copyToLocal(
                                        new SourceFile(varSelectMSEOutputPath, modelConfig.getDataSet().getSource()),
                                        Constants.SHIFU_VARSELECT_SE_OUTPUT_NAME, varSelMSEHistPath);
                            }
                            // save as backup
                            super.saveColumnConfigList(pathFinder.getVarSelColumnConfig(i), this.columnConfigList);
                            // save as current copy
                            super.saveColumnConfigList();
                        }
                    } else if(filterBy.equalsIgnoreCase(Constants.FILTER_BY_VOTED)) {
                        votedVariablesSelection();
                    }
                } else {
                    boolean hasCandidates = CommonUtils.hasCandidateColumns(this.columnConfigList);
                    if(this.modelConfig.getVarSelect().getForceEnable()
                            && CollectionUtils.isNotEmpty(this.modelConfig.getListForceSelect())) {
                        LOG.info("Force Selection is enabled ... "
                                + "for multi-classification, currently only use it to selected variables.");
                        for(ColumnConfig config: this.columnConfigList) {
                            if(config.isForceSelect()) {
                                if(!CommonUtils.isGoodCandidate(config, hasCandidates, modelConfig.isRegression())) {
                                    LOG.warn("!! Variable - {} is not a good candidate. But it is in forceselect list",
                                            config.getColumnName());
                                }
                                config.setFinalSelect(true);
                            }
                        }
                        LOG.info("{} variables are selected by force.", this.modelConfig.getListForceSelect().size());
                    } else {
                        // multiple classification, select all candidate at first, TODO add SE for multi-classification
                        for(ColumnConfig config: this.columnConfigList) {
                            if(CommonUtils.isGoodCandidate(config, hasCandidates, modelConfig.isRegression())) {
                                config.setFinalSelect(true);
                            }
                        }
                    }
                }

                // clean shadow targets for multi-segments
                cleanShadowTargetsForSegments();

                if(modelConfig.getVarSelect().getAutoFilterEnable()) {
                    if(this.modelConfig.isMultiTask()) {
                        for(int i = 0; i < this.mtlColumnConfigLists.size(); i++) {
                            List<ColumnConfig> ccList = this.mtlColumnConfigLists.get(i);
                            runAutoVarFilter(ccList);
                        }
                    } else {
                        runAutoVarFilter(this.columnConfigList);
                    }
                }
            }

            // save column config to file and sync to
            clearUp(ModelStep.VARSELECT);
        } catch (ShifuException e) {
            LOG.error("Error:" + e.getError().toString() + "; msg:" + e.getMessage(), e);
            return -1;
        } catch (Exception e) {
            LOG.error("Error:" + e.getMessage(), e);
            return -1;
        }
        LOG.info("Step Finished: varselect with {} ms", (System.currentTimeMillis() - start));
        return 0;
    }

    private boolean isLinearSEorST() {
        boolean isLinearModel = CommonUtils.isLinearTarget(this.modelConfig, this.columnConfigList);
        String filterBy = this.modelConfig.getVarSelectFilterBy();
        return isLinearModel && (filterBy.equalsIgnoreCase(Constants.FILTER_BY_SE)
                || filterBy.equalsIgnoreCase(Constants.FILTER_BY_ST));
    }

    /**
     * Recover auto-filtered variable status from varsel history file
     * 
     * @param varselHistory
     *            - variable selection history file
     * @throws IOException
     */
    private void recoverVarselStatusFromHist(String varselHistory) throws IOException {
        List<VarSelDesc> varSelDescList = loadVarSelDescList(varselHistory);
        for(VarSelDesc varSelDesc: varSelDescList) {
            ColumnConfig columnConfig = this.columnConfigList.get(varSelDesc.getColumnId());
            if(columnConfig.isFinalSelect() == varSelDesc.getNewSelStatus()) {
                LOG.info("Recover column - {} from {} to {}", varSelDesc.getColumnName(), varSelDesc.getNewSelStatus(),
                        varSelDesc.getOldSelStatus());
                columnConfig.setFinalSelect(varSelDesc.getOldSelStatus());
            }
        }
    }

    /**
     * Load variable selection history file into VarSelDesc
     * 
     * @param varselHistory
     *            - variable selection history file file
     */
    private List<VarSelDesc> loadVarSelDescList(String varselHistory) throws IOException {
        Reader reader = ShifuFileUtils.getReader(varselHistory, SourceType.LOCAL);
        List<String> autoFilterList = IOUtils.readLines(reader);
        IOUtils.closeQuietly(reader);

        List<VarSelDesc> varSelDescList = new ArrayList<VarSelDesc>();
        for(String filterDesc: autoFilterList) {
            VarSelDesc varSelDesc = VarSelDesc.fromString(filterDesc);
            if(varSelDesc != null) {
                varSelDescList.add(varSelDesc);
            }
        }

        return varSelDescList;
    }

    private void selectByFeatureImportance() throws Exception {
        List<BasicML> models = null;
        boolean reuseCurrentModel = Environment.getBoolean("shifu.varsel.reuse.model", Boolean.FALSE);
        if(reuseCurrentModel) {
            try {
                models = ModelSpecLoaderUtils.loadBasicModels(this.modelConfig, null);
            } catch (IOException e) {
                LOG.warn("No existing models found. Will try to build new model for FI.");
            }
        }
        if(models == null || models.size() < 1) {
            TrainModelProcessor trainModelProcessor = new TrainModelProcessor();
            trainModelProcessor.setForVarSelect(true);
            trainModelProcessor.run();
            models = ModelSpecLoaderUtils.loadBasicModels(this.modelConfig, null);
        }

        // compute feature importance and write to local file
        Map<Integer, MutablePair<String, Double>> featureImportances = CommonUtils
                .computeTreeModelFeatureImportance(models);
        if(super.modelConfig.getVarSelect().getFilterEnable()) {
            this.postProcessFIVarSelect(featureImportances);
        }
    }

    public boolean getIsToReset() {
        return getBooleanParam(this.otherConfigs, Constants.IS_TO_RESET);
    }

    public int getRecursiveCnt() {
        return getIntParam(this.otherConfigs, Constants.RECURSIVE_CNT, 1);
    }

    public boolean getIsToList() {
        return getBooleanParam(this.otherConfigs, Constants.IS_TO_LIST);
    }

    public boolean getIsToAutoFilter() {
        return getBooleanParam(this.otherConfigs, Constants.IS_TO_FILTER_AUTO);
    }

    public boolean getIsRecoverAuto() {
        return getBooleanParam(this.otherConfigs, Constants.IS_TO_RECOVER_AUTO);
    }

    public String getVarSelFile() {
        return getStringParam(this.otherConfigs, Constants.VAR_SEL_FILE);
    }

    private void validateParameters() throws Exception {
        String filterBy = this.modelConfig.getVarSelectFilterBy();
        if(filterBy.equalsIgnoreCase(Constants.FILTER_BY_SE) || filterBy.equalsIgnoreCase(Constants.FILTER_BY_ST)) {
            validateSEParameters();
            validateNormalize();
        }
    }

    public void resetAllFinalSelect(List<ColumnConfig> columnConfigList) throws IOException {
        LOG.info("!!! Reset all variables finalSelect = false");
        for(ColumnConfig columnConfig: columnConfigList) {
            columnConfig.setFinalSelect(false);
            columnConfig.setColumnFlag(null);
        }
    }

    private void validateNormalize() throws IOException {
        if(!ShifuFileUtils.isFileExists(
                new PathFinder(modelConfig).getNormalizedDataPath(this.modelConfig.getDataSet().getSource()),
                this.modelConfig.getDataSet().getSource())) {
            throw new IllegalStateException("Cannot find normalized data, please do 'Shifu normalize' firstly.");
        }
    }

    private void validateSEParameters() {
        if(!CommonConstants.NN_ALG_NAME.equalsIgnoreCase(super.getModelConfig().getTrain().getAlgorithm())
                && !"LR".equalsIgnoreCase(super.getModelConfig().getTrain().getAlgorithm())
                && !CommonConstants.WDL_ALG_NAME.equalsIgnoreCase(super.getModelConfig().getTrain().getAlgorithm())
                && !CommonUtils.isTensorFlowModel(super.getModelConfig().getTrain().getAlgorithm())) {
            throw new IllegalArgumentException(
                    "Currently we only support NN and LR distributed training to do wrapper by analyzing variable selection.");
        }

        if(super.getModelConfig().getDataSet().getSource() != SourceType.HDFS) {
            throw new IllegalArgumentException(
                    "Currently we only support distributed wrapper by analyzing on HDFS source type.");
        }

        if(!super.getModelConfig().isMapReduceRunMode()) {
            throw new IllegalArgumentException(
                    "Currently we only support distributed wrapper by on MAPRED or DIST mode.");
        }
    }

    private void votedVariablesSelection() throws ClassNotFoundException, IOException, InterruptedException, Exception {
        LOG.info("Start voted variables selection ");
        // sync data back to hdfs
        super.syncDataToHdfs(modelConfig.getDataSet().getSource());

        SourceType sourceType = super.getModelConfig().getDataSet().getSource();
        Configuration conf = new Configuration();

        final List<String> args = new ArrayList<String>();
        // prepare parameter
        prepareVarSelParams(args, sourceType);

        Path columnIdsPath = getVotedSelectionPath(sourceType);
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT,
                ml.shifu.shifu.util.Constants.VAR_SEL_COLUMN_IDS_OUPUT, columnIdsPath.toString()));

        long start = System.currentTimeMillis();

        GuaguaMapReduceClient guaguaClient = new GuaguaMapReduceClient();

        String hdpVersion = HDPUtils.getHdpVersionForHDP224();
        if(StringUtils.isNotBlank(hdpVersion)) {
            // for hdp 2.2.4, hdp.version should be set and configuration files should be add to container class path
            conf.set("hdp.version", hdpVersion);
        }
        guaguaClient.createJob(args.toArray(new String[0])).waitForCompletion(true);

        LOG.info("Voted variables selection finished in {}ms.", System.currentTimeMillis() - start);

        persistColumnIds(columnIdsPath);
        super.syncDataToHdfs(sourceType);
    }

    private int persistColumnIds(Path path) {
        try {
            List<Scanner> scanners = ShifuFileUtils.getDataScanners(path.toString(),
                    modelConfig.getDataSet().getSource());

            List<Integer> ids = null;
            for(Scanner scanner: scanners) {
                while(scanner.hasNextLine()) {
                    String[] raw = scanner.nextLine().trim().split("\\|");

                    @SuppressWarnings("unused")
                    int idSize = Integer.parseInt(raw[0]);

                    ids = CommonUtils.stringToIntegerList(raw[1]);

                }
            }

            // prevent multiply running setting
            for(ColumnConfig config: columnConfigList) {
                if(!config.isForceSelect()) {
                    config.setFinalSelect(Boolean.FALSE);
                }
            }

            for(Integer id: ids) {
                this.columnConfigList.get(id).setFinalSelect(Boolean.TRUE);
            }

            super.saveColumnConfigList();
        } catch (IOException e) {
            LOG.error("Error:", e);
            return -1;
        } catch (IllegalArgumentException e) {
            LOG.error("Error:", e);
            return -1;
        }

        return 0;
    }

    private Path getVotedSelectionPath(SourceType sourceType) {
        Path filePath = new Path(getPathFinder().getVarSelsPath(sourceType), "VarSels");
        return ShifuFileUtils.getFileSystemBySourceType(sourceType, filePath)
                .makeQualified(new Path(getPathFinder().getVarSelsPath(sourceType), "VarSels"));
    }

    @SuppressWarnings("unused")
    private void prepareVarSelParams(final List<String> args, final SourceType sourceType) throws Exception {
        args.add("-libjars");

        args.add(addRuntimeJars());

        args.add("-i");
        Path filePath = new Path(modelConfig.getDataSetRawPath());
        args.add(ShifuFileUtils.getFileSystemBySourceType(sourceType, filePath).makeQualified(filePath).toString());

        String zkServers = Environment.getProperty(Environment.ZOO_KEEPER_SERVERS);
        if(StringUtils.isEmpty(zkServers)) {
            LOG.warn(
                    "No specified zookeeper settings from zookeeperServers in shifuConfig file, Guagua will set embeded zookeeper server in client process. For big data applications, specified zookeeper servers are strongly recommended.");
        } else {
            args.add("-z");
            args.add(zkServers);
        }

        // setting the class
        args.add("-w");
        args.add(VarSelWorker.class.getName());

        args.add("-m");
        args.add(VarSelMaster.class.getName());

        args.add("-c");
        // the reason to add 1 is that the first iteration in D-NN implementation is used for training preparation.
        // FIXME, how to set iteration number
        int forceSelectCount = 0;
        int candidateCount = 0;
        boolean hasCandidates = CommonUtils.hasCandidateColumns(columnConfigList);
        for(ColumnConfig columnConfig: columnConfigList) {
            if(columnConfig.isForceSelect()) {
                forceSelectCount++;
            }
            if(CommonUtils.isGoodCandidate(columnConfig, hasCandidates)) {
                candidateCount++;
            }
        }

        int iterationCnt = (Integer) this.modelConfig.getVarSelect().getParams()
                .get(CandidateGenerator.POPULATION_MULTIPLY_CNT) + 1;
        args.add(Integer.toString(iterationCnt));

        args.add("-mr");
        args.add(VarSelMasterResult.class.getName());

        args.add("-wr");
        args.add(VarSelWorkerResult.class.getName());

        // setting conductor
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT,
                ml.shifu.shifu.util.Constants.VAR_SEL_MASTER_CONDUCTOR,
                Environment.getProperty(Environment.VAR_SEL_MASTER_CONDUCTOR, WrapperMasterConductor.class.getName())));

        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT,
                ml.shifu.shifu.util.Constants.VAR_SEL_WORKER_CONDUCTOR,
                Environment.getProperty(Environment.VAR_SEL_MASTER_CONDUCTOR, WrapperWorkerConductor.class.getName())));

        // setting queue
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, NNConstants.MAPRED_JOB_QUEUE_NAME, Environment
                .getProperty(Environment.HADOOP_JOB_QUEUE, ml.shifu.shifu.util.Constants.DEFAULT_JOB_QUEUE)));

        // MAPRED timeout
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, NNConstants.MAPRED_TASK_TIMEOUT, Environment
                .getInt(NNConstants.MAPRED_TASK_TIMEOUT, ml.shifu.shifu.util.Constants.DEFAULT_MAPRED_TIME_OUT)));

        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, GuaguaConstants.GUAGUA_MASTER_INTERCEPTERS,
                VarSelOutput.class.getName()));

        // setting model config column config
        Path modelConfPath = new Path(super.getPathFinder().getModelConfigPath(sourceType));
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, CommonConstants.SHIFU_MODEL_CONFIG,
                ShifuFileUtils.getFileSystemBySourceType(sourceType, modelConfPath).makeQualified(modelConfPath)));
        Path columnConfPath = new Path(super.getPathFinder().getColumnConfigPath(sourceType));
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, CommonConstants.SHIFU_COLUMN_CONFIG,
                ShifuFileUtils.getFileSystemBySourceType(sourceType, columnConfPath).makeQualified(columnConfPath)));

        // source type
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, CommonConstants.MODELSET_SOURCE_TYPE,
                sourceType));

        // computation time
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT,
                GuaguaConstants.GUAGUA_COMPUTATION_TIME_THRESHOLD, 60 * 60 * 1000l));
        setHeapSizeAndSplitSize(args);

        // one can set guagua conf in shifuconfig
        CommonUtils.injectHadoopShifuEnvironments(new ValueVisitor() {
            @Override
            public void inject(Object key, Object value) {
                args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, key.toString(), value.toString()));
            }
        });
    }

    // GuaguaOptionsParser doesn't to support *.jar currently.
    private String addRuntimeJars() throws ClassNotFoundException, FileNotFoundException, IOException {
        List<String> jars = new ArrayList<String>(16);
        // pig-*.jar
        jars.add(JarManager.findContainingJar(Tuple.class));
        // jackson-databind-*.jar
        jars.add(JarManager.findContainingJar(ObjectMapper.class));
        // jackson-core-*.jar
        jars.add(JarManager.findContainingJar(JsonParser.class));
        // jackson-annotations-*.jar
        jars.add(JarManager.findContainingJar(JsonIgnore.class));
        // commons-compress-*.jar
        jars.add(JarManager.findContainingJar(BZip2CompressorInputStream.class));
        // commons-lang-*.jar
        jars.add(JarManager.findContainingJar(StringUtils.class));
        // commons-collections-*.jar
        jars.add(JarManager.findContainingJar(ListUtils.class));
        // common-io-*.jar
        jars.add(JarManager.findContainingJar(org.apache.commons.io.IOUtils.class));
        // guava-*.jar
        jars.add(JarManager.findContainingJar(Splitter.class));
        // encog-core-*.jar
        jars.add(JarManager.findContainingJar(MLDataSet.class));
        // shifu-*.jar
        jars.add(JarManager.findContainingJar(getClass()));
        // guagua-core-*.jar
        jars.add(JarManager.findContainingJar(GuaguaConstants.class));
        // guagua-mapreduce-*.jar
        jars.add(JarManager.findContainingJar(GuaguaMapReduceConstants.class));
        // zookeeper-*.jar
        jars.add(JarManager.findContainingJar(ZooKeeper.class));
        // netty-*.jar
        jars.add(JarManager.findContainingJar(ServerBootstrap.class));
        // common jexl related jar
        jars.add(JarManager.findContainingJar(JexlException.class));

        if(CommonUtils.isTensorFlowModel(this.modelConfig.getAlgorithm())) {
            jars.add(JarManager.findContainingJar(Class.forName("ml.shifu.shifu.tensorflow.TensorflowModel")));
            String tJar = JarManager.findContainingJar(Class.forName("org.tensorflow.Tensors"));
            LOG.info("TF jar {}.", tJar);
            jars.add(tJar);
            jars.add(tJar.replaceAll("libtensorflow", "libtensorflow_jni"));
            jars.add(tJar.replaceAll("libtensorflow", "tensorflow"));
        }

        String hdpVersion = HDPUtils.getHdpVersionForHDP224();
        if(StringUtils.isNotBlank(hdpVersion)) {
            // for hdp 2.2.4, hdp.version should be set and configuration files should be add to container class path
        }

        return StringUtils.join(jars, NNConstants.LIB_JAR_SEPARATOR);
    }

    /**
     * Wrapper through {@link TrainModelProcessor} and a MapReduce job to analyze biggest sensitivity RMS.
     */
    private void distributedSEWrapper(String trainLogFile) throws Exception {
        // 1. Train a model using current selected variables, if no variables selected, use all candidate variables.
        boolean reuseCurrentModel = Environment.getBoolean("shifu.varsel.reuse.model", Boolean.FALSE);
        SourceType source = this.modelConfig.getDataSet().getSource();

        if(!reuseCurrentModel) {
            TrainModelProcessor trainModelProcessor = new TrainModelProcessor();
            trainModelProcessor.setForVarSelect(true);
            trainModelProcessor.setTrainLogFile(trainLogFile);
            trainModelProcessor.run();
        }

        syncDataToHdfs(source);

        // 2. Submit a MapReduce job to analyze sensitivity RMS.
        Configuration conf = new Configuration();
        // 2.1 prepare se job conf
        prepareSEJobConf(source, conf);
        // 2.2 get output path
        String varSelectMSEOutputPath = super.getPathFinder().getVarSelectMSEOutputPath(source);

        // 2.3 create se job
        Job job = null;
        if(modelConfig.getVarSelect().getFilterBy().equalsIgnoreCase(Constants.FILTER_BY_SC)) {
            job = createSCMapReduceJob(source, conf, varSelectMSEOutputPath);
        } else {
            job = createSEMapReduceJob(source, conf, varSelectMSEOutputPath);
        }

        // 2.4 clean output firstly
        ShifuFileUtils.deleteFile(varSelectMSEOutputPath, source);

        // 2.5 submit job
        if(job.waitForCompletion(true)) {
            // 2.6 post process 4 var select
            if(super.modelConfig.getVarSelect().getFilterEnable()) {
                postProcess4SEVarSelect(source, varSelectMSEOutputPath);
            } else {
                LOG.info("Only print sensitivity analysis report.");
                LOG.info(
                        "Sensitivity analysis report is in {}/{}-* file(s) with format 'column_index\tcolumn_name\tmean\trms\tvariance'.",
                        varSelectMSEOutputPath, Constants.SHIFU_VARSELECT_SE_OUTPUT_NAME);
            }
        } else {
            LOG.error("VarSelect SE hadoop job is failed, please re-try varselect step.");
        }
    }

    private Job createSEMapReduceJob(SourceType source, Configuration conf, String varSelectMSEOutputPath)
            throws IOException {
        @SuppressWarnings("deprecation")
        Job job = new Job(conf, "Shifu: Variable Selection Wrapper Job : " + this.modelConfig.getModelSetName());
        job.setJarByClass(getClass());
        boolean isSEVarSelMulti = Boolean.TRUE.toString().equalsIgnoreCase(
                Environment.getProperty(Constants.SHIFU_VARSEL_SE_MULTI, Constants.SHIFU_DEFAULT_VARSEL_SE_MULTI));
        if(isSEVarSelMulti) {
            job.setMapperClass(MultithreadedMapper.class);
            MultithreadedMapper.setMapperClass(job, VarSelectMapper.class);
            int threads = getMultiThreadCount();
            conf.setInt("mapreduce.map.cpu.vcores", threads);
            MultithreadedMapper.setNumberOfThreads(job, threads);
        } else {
            job.setMapperClass(VarSelectMapper.class);
        }
        job.setMapOutputKeyClass(LongWritable.class);
        job.setMapOutputValueClass(ColumnInfo.class);
        job.setInputFormatClass(CombineInputFormat.class);
        Path filePath = new Path(super.getPathFinder().getNormalizedDataPath());
        FileInputFormat.setInputPaths(job, ShifuFileUtils.getFileSystemBySourceType(source, filePath)
                .makeQualified(filePath));

        job.setReducerClass(VarSelectReducer.class);
        // Only one reducer, no need set combiner because of distinct keys in map outputs.
        job.setNumReduceTasks(1);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        FileOutputFormat.setOutputPath(job, new Path(varSelectMSEOutputPath));
        MultipleOutputs.addNamedOutput(job, Constants.SHIFU_VARSELECT_SE_OUTPUT_NAME, TextOutputFormat.class,
                Text.class, Text.class);
        return job;
    }

    private Job createSCMapReduceJob(SourceType source, Configuration conf, String varSelectMSEOutputPath)
            throws IOException {
        Job job = Job.getInstance(conf);
        job.setJobName("Shifu: Variable Selection Wrapper Job : " + this.modelConfig.getModelSetName());
        job.setJarByClass(getClass());

        // mapper and mapper out
        boolean isSEVarSelMulti = Boolean.TRUE.toString().equalsIgnoreCase(
                Environment.getProperty(Constants.SHIFU_VARSEL_SE_MULTI, Constants.SHIFU_DEFAULT_VARSEL_SE_MULTI));
        if(isSEVarSelMulti) {
            job.setMapperClass(MultithreadedMapper.class);
            MultithreadedMapper.setMapperClass(job, VarSelectSCMapper.class);
            int threads = getMultiThreadCount();
            conf.setInt("mapreduce.map.cpu.vcores", threads);
            MultithreadedMapper.setNumberOfThreads(job, threads);
        } else {
            job.setMapperClass(VarSelectSCMapper.class);
        }
        job.setMapOutputKeyClass(LongWritable.class);
        job.setMapOutputValueClass(ColumnScore.class);

        // input
        job.setInputFormatClass(CombineInputFormat.class);
        Path filePath = new Path(super.getPathFinder().getNormalizedDataPath());
        FileInputFormat.setInputPaths(job, ShifuFileUtils.getFileSystemBySourceType(source, filePath)
                .makeQualified(filePath));

        job.setReducerClass(VarSelectSCReducer.class);
        // Only one reducer, no need set combiner because of distinct keys in map outputs.
        int reduceNum = this.columnConfigList.size() / 3;
        reduceNum = (reduceNum > 999 ? 999 : reduceNum);
        job.setNumReduceTasks(reduceNum == 0 ? 1 : reduceNum);

        job.setOutputKeyClass(LongWritable.class);
        job.setOutputValueClass(Text.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        FileOutputFormat.setOutputPath(job, new Path(varSelectMSEOutputPath));
        return job;
    }

    private int getMultiThreadCount() {
        int threads;
        try {
            threads = Integer.parseInt(Environment.getProperty(Constants.SHIFU_VARSEL_SE_MULTI_THREAD,
                    Constants.SHIFU_DEFAULT_VARSEL_SE_MULTI_THREAD + ""));
        } catch (Exception e) {
            LOG.warn("'shifu.varsel.se.multi.thread' should be a int value, set default value: {}",
                    Constants.SHIFU_DEFAULT_VARSEL_SE_MULTI_THREAD);
            threads = Constants.SHIFU_DEFAULT_VARSEL_SE_MULTI_THREAD;
        }
        return threads;
    }

    private void prepareSEJobConf(SourceType source, final Configuration conf) throws Exception {
        Path modelConfPath = new Path(super.getPathFinder().getModelConfigPath(source));
        String modelConfigPath = ShifuFileUtils.getFileSystemBySourceType(source, modelConfPath)
                .makeQualified(modelConfPath).toString();
        Path columnConfPath = new Path(super.getPathFinder().getColumnConfigPath(source));
        String columnConfigPath = ShifuFileUtils.getFileSystemBySourceType(source, columnConfPath)
                .makeQualified(columnConfPath).toString();
        // only the first model is sued for sensitivity analysis
        String filePath = null;
        if(CommonUtils.isTensorFlowModel(this.modelConfig.getAlgorithm())) {
            filePath = modelConfigPath + "," + columnConfigPath;
        } else {
            Path modelPath = new Path(super.getPathFinder().getModelsPath(),
                    "model0." + modelConfig.getAlgorithm().toLowerCase());
            String seModelPath = ShifuFileUtils.getFileSystemBySourceType(source, modelPath)
                    .makeQualified(modelPath).toString();
            filePath = modelConfigPath + "," + columnConfigPath + "," + seModelPath;
        }

        // add jars and files to hadoop mapper and reducer
        new GenericOptionsParser(conf, new String[] { "-libjars", addRuntimeJars(), "-files", filePath });

        conf.setBoolean(GuaguaMapReduceConstants.MAPRED_MAP_TASKS_SPECULATIVE_EXECUTION, true);
        conf.setBoolean(GuaguaMapReduceConstants.MAPRED_REDUCE_TASKS_SPECULATIVE_EXECUTION, true);
        conf.setBoolean(GuaguaMapReduceConstants.MAPREDUCE_MAP_SPECULATIVE, true);
        conf.setBoolean(GuaguaMapReduceConstants.MAPREDUCE_REDUCE_SPECULATIVE, true);
        conf.set(Constants.SHIFU_MODEL_CONFIG, ShifuFileUtils.getFileSystemBySourceType(source, modelConfPath)
                .makeQualified(modelConfPath).toString());
        conf.set(Constants.SHIFU_COLUMN_CONFIG, ShifuFileUtils.getFileSystemBySourceType(source, columnConfPath)
                .makeQualified(columnConfPath).toString());
        conf.set(NNConstants.MAPRED_JOB_QUEUE_NAME, Environment.getProperty(Environment.HADOOP_JOB_QUEUE, "default"));
        conf.set(Constants.SHIFU_MODELSET_SOURCE_TYPE, source.toString());
        // set mapreduce.job.max.split.locations to 100 to suppress warnings
        conf.setInt(GuaguaMapReduceConstants.MAPREDUCE_JOB_MAX_SPLIT_LOCATIONS, 5000);

        // Tmp false because of some cluster by default use gzip while CombineInputFormat will split gzip file (a bug)
        conf.setBoolean(CombineInputFormat.SHIFU_VS_SPLIT_COMBINABLE, false);
        conf.setBoolean("mapreduce.input.fileinputformat.input.dir.recursive", true);

        conf.set("mapred.reduce.slowstart.completed.maps",
                Environment.getProperty("mapred.reduce.slowstart.completed.maps", "0.9"));
        conf.set(Constants.SHIFU_VARSELECT_FILTEROUT_TYPE, modelConfig.getVarSelectFilterBy());

        Float filterOutRatio = this.modelConfig.getVarSelect().getFilterOutRatio();
        if(filterOutRatio == null) {
            LOG.warn("filterOutRatio in var select is not set. Using default value 0.05.");
            filterOutRatio = 0.05f;
        }

        if(filterOutRatio.compareTo(Float.valueOf(1.0f)) >= 0) {
            throw new IllegalArgumentException("WrapperRatio should be in (0, 1).");
        }
        conf.setFloat(Constants.SHIFU_VARSELECT_FILTEROUT_RATIO, filterOutRatio);
        conf.setInt(Constants.SHIFU_VARSELECT_FILTER_NUM, this.modelConfig.getVarSelectFilterNum());
        String hdpVersion = HDPUtils.getHdpVersionForHDP224();
        if(StringUtils.isNotBlank(hdpVersion)) {
            // for hdp 2.2.4, hdp.version should be set and configuration files should be add to container class path
            conf.set("hdp.version", hdpVersion);
        }
        // one can set guagua conf in shifuconfig
        CommonUtils.injectHadoopShifuEnvironments(new ValueVisitor() {
            @Override
            public void inject(Object key, Object value) {
                conf.set(key.toString(), value.toString());
            }
        });

        // no matter how the mapreduce.task.io.sort.mb is set for sensitivity job, only 1 reducer and each mapper only
        // output column stats, 150MB is enough.
        conf.setInt("mapreduce.task.io.sort.mb", 150);
    }

    private void postProcessFIVarSelect(Map<Integer, MutablePair<String, Double>> importances) throws IOException {
        VariableSelector.setFilterNumberByFilterOutRatio(this.modelConfig, this.columnConfigList);
        int targetCnt = this.modelConfig.getVarSelectFilterNum();
        List<Integer> candidateColumnIdList = new ArrayList<Integer>();
        candidateColumnIdList.addAll(importances.keySet());
        int candidateCount = candidateColumnIdList.size();
        int i = 0;
        int selectCnt = 0;
        // try to select another (targetCnt - selectCnt) variables, but we need to exclude those
        // force-selected variables
        for(ColumnConfig columnConfig: this.columnConfigList) {
            if(columnConfig.isFinalSelect()) {
                columnConfig.setFinalSelect(false);
            }
            if(columnConfig.isForceSelect()) {
                columnConfig.setFinalSelect(true);
                selectCnt++;
                LOG.info("Variable {} is selected, since it is in ForceSelect list.", columnConfig.getColumnName());
            }
        }
        int forceCnt = selectCnt;

        Set<NSColumn> userCandidateColumns = CommonUtils.loadCandidateColumns(modelConfig);

        while(selectCnt < targetCnt && i < targetCnt) {
            if(i >= candidateCount) {
                LOG.warn("Var select finish due to feature importance count {} is less than target var count {}",
                        candidateCount, targetCnt);
                break;
            }
            Integer columnId = candidateColumnIdList.get(i++);
            ColumnConfig columnConfig = this.columnConfigList.get(columnId);
            if(CollectionUtils.isNotEmpty(userCandidateColumns)
                    && !userCandidateColumns.contains(new NSColumn(columnConfig.getColumnName()))) {
                LOG.info("Variable {} is not in user's candidate list. Skip it.", columnConfig.getColumnName());
            } else if(!columnConfig.isForceSelect() && !columnConfig.isForceRemove()) {
                columnConfig.setFinalSelect(true);
                selectCnt++;
                LOG.info("Variable {} is selected.", columnConfig.getColumnName());
            }
        }
        LOG.info("{} variables are selected, while {} are force-selected, and others from {} candidates.", selectCnt,
                forceCnt, candidateCount);
    }

    private void postProcess4SEVarSelect(SourceType source, String varSelectMSEOutputPath) throws IOException {
        String outputFilePattern = varSelectMSEOutputPath + Path.SEPARATOR + "part-r-*";
        if(!ShifuFileUtils.isFileExists(outputFilePattern, source)) {
            throw new RuntimeException("Var select MSE stats output file not exist.");
        }

        int selectCnt = 0;
        for(ColumnConfig config: super.columnConfigList) {
            if(config.isFinalSelect()) {
                config.setFinalSelect(false);
            }

            // enable ForceSelect
            if(config.isForceSelect()) {
                config.setFinalSelect(true);
                selectCnt++;
                LOG.info("Variable {} is selected, since it is in ForceSelect list.", config.getColumnName());
            }
        }

        Set<NSColumn> userCandidateColumns = CommonUtils.loadCandidateColumns(modelConfig);

        List<Scanner> scanners = null;
        try {
            int targetCnt = 0; // total variable count that user want to select
            List<Integer> candidateColumnIdList = null;
            if(modelConfig.getVarSelect().getFilterBy().equalsIgnoreCase(Constants.FILTER_BY_SC)) {
                candidateColumnIdList = getCandidateVariableList(source, varSelectMSEOutputPath);
                targetCnt = (int) (candidateColumnIdList.size()
                        * (1.0f - modelConfig.getVarSelect().getFilterOutRatio()));
            } else {
                // here only works for 1 reducer
                Path filePath = new Path(outputFilePattern);
                FileStatus[] globStatus = ShifuFileUtils.getFileSystemBySourceType(source, filePath)
                        .globStatus(filePath);
                if(globStatus == null || globStatus.length == 0) {
                    throw new RuntimeException("Var select MSE stats output file not exist.");
                }
                scanners = ShifuFileUtils.getDataScanners(globStatus[0].getPath().toString(), source);
                String str = null;
                candidateColumnIdList = new ArrayList<Integer>();
                Scanner scanner = scanners.get(0);
                while(scanner.hasNext()) {
                    ++targetCnt;
                    str = scanner.nextLine().trim();
                    candidateColumnIdList.add(Integer.parseInt(str));
                }
            }

            int i = 0;
            int candidateCount = candidateColumnIdList.size();
            // try to select another (targetCnt - selectCnt) variables, but we need to exclude those
            // force-selected variables
            while(selectCnt < targetCnt && i < targetCnt) {
                if(i >= candidateCount) {
                    LOG.warn("Var select finish due candidate column {} is less than target var count {}",
                            candidateCount, targetCnt);
                    break;
                }
                Integer columnId = candidateColumnIdList.get(i++);
                // after supporting segments, the columns will expansion. the columnId may not the position
                // in columnConfigList. It's safe to columnId to search (make sure columnNum == columnId)
                ColumnConfig columnConfig = CommonUtils.getColumnConfig(this.columnConfigList, columnId);
                if(CollectionUtils.isNotEmpty(userCandidateColumns)
                        && !userCandidateColumns.contains(new NSColumn(columnConfig.getColumnName()))) {
                    LOG.info("Variable {} is not in user's candidate list. Skip it.", columnConfig.getColumnName());
                } else if(!columnConfig.isForceSelect() && !columnConfig.isForceRemove()) {
                    columnConfig.setFinalSelect(true);
                    selectCnt++;
                    LOG.info("Variable {} is selected.", columnConfig.getColumnName());
                }
            }

            LOG.info("{} variables are selected.", selectCnt);
            if(modelConfig.getVarSelect().getFilterBy().equalsIgnoreCase(Constants.FILTER_BY_SC)) {
                LOG.info(
                        "Sensitivity analysis report is in {}/part-* file(s) with format 'column_index\tsensitivity_perf'.",
                        varSelectMSEOutputPath);
            } else {
                LOG.info(
                        "Sensitivity analysis report is in {}/{}-* file(s) with format 'column_index\tcolumn_name\tmean\trms\tvariance'.",
                        varSelectMSEOutputPath, Constants.SHIFU_VARSELECT_SE_OUTPUT_NAME);
                this.seStatsMap = readSEValuesToMap(
                        varSelectMSEOutputPath + Path.SEPARATOR + Constants.SHIFU_VARSELECT_SE_OUTPUT_NAME + "-*",
                        source);
            }
        } finally {
            if(scanners != null) {
                for(Scanner scanner: scanners) {
                    if(scanner != null) {
                        scanner.close();
                    }
                }
            }
        }
    }

    private List<Integer> getCandidateVariableList(SourceType sourceType, String varSelectMSEOutputPath)
            throws IOException {
        HdfsPartFile partFile = new HdfsPartFile(varSelectMSEOutputPath, sourceType);
        List<VarSelPerf> varSelPerfList = new ArrayList<>();
        String line = null;
        while((line = partFile.readLine()) != null) {
            VarSelPerf perf = VarSelPerf.create(line);
            if(perf != null) {
                varSelPerfList.add(perf);
            }
        }
        partFile.close();

        Collections.sort(varSelPerfList, new Comparator<VarSelPerf>() {
            @Override
            public int compare(VarSelPerf from, VarSelPerf to) {
                return Double.compare(from.sensitivityPerf, to.sensitivityPerf);
            }
        });

        System.out.println(varSelPerfList);

        List<Integer> candidateColumnIds = new ArrayList<>();
        for(VarSelPerf perf: varSelPerfList) {
            if(perf.columnId > -1) { // remove overall column
                candidateColumnIds.add(perf.columnId);
            }
        }

        System.out.println(candidateColumnIds);
        return candidateColumnIds;
    }

    private Map<Integer, ColumnStatistics> readSEValuesToMap(String seOutputFiles, SourceType source)
            throws IOException {
        // here only works for 1 reducer
        Path filePath = new Path(seOutputFiles);
        FileStatus[] globStatus = ShifuFileUtils.getFileSystemBySourceType(source, filePath).globStatus(filePath);
        if(globStatus == null || globStatus.length == 0) {
            throw new RuntimeException("Var select MSE stats output file not exist.");
        }
        Map<Integer, ColumnStatistics> map = new HashMap<Integer, ColumnStatistics>();
        List<Scanner> scanners = null;
        try {
            scanners = ShifuFileUtils.getDataScanners(globStatus[0].getPath().toString(), source);
            for(Scanner scanner: scanners) {
                String str = null;
                while(scanner.hasNext()) {
                    str = scanner.nextLine().trim();
                    String[] splits = CommonUtils.split(str, "\t");
                    if(splits.length == 5) {
                        map.put(Integer.parseInt(splits[0].trim()), new ColumnStatistics(Double.parseDouble(splits[2]),
                                Double.parseDouble(splits[3]), Double.parseDouble(splits[4])));
                    }
                }
            }
        } finally {
            if(scanners != null) {
                for(Scanner scanner: scanners) {
                    if(scanner != null) {
                        scanner.close();
                    }
                }
            }
        }
        return map; // should be a bug, if it always return null
    }

    /**
     * Coopy model spec under modelsPath to varSelDir folder
     * 
     * @param modelsPath
     *            - model spec path, like ./models
     * @param varSelDir
     *            - valsel directory
     * @param i
     *            - round of iteration
     * @throws IOException
     */
    private void copyModelSpec(String modelsPath, String varSelDir, int i) throws IOException {
        File sourceFolder = new File(modelsPath);
        File[] modelFiles = sourceFolder.listFiles();
        for(File mf: modelFiles) {
            ShifuFileUtils.copy(mf.getPath(), varSelDir + File.separator + mf.getName() + "." + i, SourceType.LOCAL);
        }
    }

    @Override
    protected void clearUp(ModelStep step) throws IOException {
        try {
            this.saveColumnConfigList();
        } catch (Exception e) {
            throw new ShifuException(ShifuErrorCode.ERROR_WRITE_COLCONFIG, e);
        }
        this.syncDataToHdfs(this.modelConfig.getDataSet().getSource());
    }

    private void cleanShadowTargetsForSegments() throws IOException {
        String filterExpressions = super.modelConfig.getSegmentFilterExpressionsAsString();
        Environment.getProperties().put("shifu.segment.expressions", filterExpressions);
        if(StringUtils.isNotBlank(filterExpressions)) {
            LOG.info("There are segments. Set all target shadow columns to ForceRemove and final select false");
            String[] splits = CommonUtils.split(filterExpressions, Constants.SHIFU_STATS_FILTER_EXPRESSIONS_DELIMETER);
            for(int i = 0; i < super.columnConfigList.size(); i++) {
                ColumnConfig config = super.columnConfigList.get(i);
                int rawSize = super.columnConfigList.size() / (1 + splits.length);
                if(config.isTarget()) {
                    for(int j = 0; j < splits.length; j++) {
                        ColumnConfig otherConfig = super.columnConfigList.get((j + 1) * rawSize + i);
                        otherConfig.setColumnFlag(ColumnFlag.ForceRemove);
                        otherConfig.setFinalSelect(false);
                    }
                    break;
                }
            }
        }
    }

    private void runAutoVarFilter(List<ColumnConfig> columnConfigList) throws IOException {
        if(this.modelConfig.getVarSelect().getPostCorrelationMetric().equals(PostCorrelationMetric.SE)
                && this.seStatsMap == null) {
            SourceType source = this.modelConfig.getDataSet().getSource();
            String varSelectMSEOutputPath = super.getPathFinder().getVarSelectMSEOutputPath(source);
            this.seStatsMap = readSEValuesToMap(
                    varSelectMSEOutputPath + Path.SEPARATOR + Constants.SHIFU_VARSELECT_SE_OUTPUT_NAME + "-*", source);
        }

        List<VarSelDesc> varSelDescList = new ArrayList<VarSelDesc>();
        autoVarSelCondition(varSelDescList, columnConfigList);
        if(CollectionUtils.isNotEmpty(varSelDescList)) {
            String varselHistory = this.pathFinder.getVarSelHistory();
            ShifuFileUtils.writeLines(varSelDescList, varselHistory, SourceType.LOCAL);
        }
    }

    /**
     * To do some auto variable selection like remove ID-like variables, remove variable with high missing rate.
     * 
     * @throws IOException
     *             any IO exception
     */
    private void autoVarSelCondition(List<VarSelDesc> varSelDescList, List<ColumnConfig> columnConfigList)
            throws IOException {
        // here we do loop again as it is not bad for variables less than 100,000
        // 1. check missing rate
        for(ColumnConfig config: columnConfigList) {
            if(!config.isTarget() && !config.isMeta() && !config.isForceSelect() // column needs check
                    && config.isFinalSelect() && isHighMissingRateColumn(config)) {
                LOG.warn("Column {} is with very high missing rate, set final select to false. "
                        + "If not, you can check it manually in ColumnConfig.json", config.getColumnName());
                config.setFinalSelect(false);
                varSelDescList.add(new VarSelDesc(config, VarSelReason.HIGH_MISSING_RATE));
            }
        }

        // 2. check KS and IV min threshold value
        for(ColumnConfig config: columnConfigList) {
            if(!config.isTarget() && !config.isMeta() && !config.isForceSelect() && config.isFinalSelect()) {
                float minIvThreshold = (super.modelConfig.getVarSelect().getMinIvThreshold() == null ? 0f
                        : super.modelConfig.getVarSelect().getMinIvThreshold());
                if(config.getIv() != null && config.getIv() < minIvThreshold) {
                    LOG.warn("IV of column {} is less than minimal IV threshold, set final select to false. "
                            + "If not, you can check it manually in ColumnConfig.json", config.getColumnName());
                    config.setFinalSelect(false);
                    varSelDescList.add(new VarSelDesc(config, VarSelReason.IV_TOO_LOW));
                }

                float minKsThreshold = (super.modelConfig.getVarSelect().getMinKsThreshold() == null ? 0f
                        : super.modelConfig.getVarSelect().getMinKsThreshold());
                if(config.getKs() != null && config.getKs() < minKsThreshold) {
                    LOG.warn("KS of column {} is less than minimal KS threshold, set final select to false. "
                            + "If not, you can check it manually in ColumnConfig.json", config.getColumnName());
                    config.setFinalSelect(false);
                    varSelDescList.add(new VarSelDesc(config, VarSelReason.KS_TOO_LOW));
                }
            }
        }

        // 3. check correlation value:
        if(!ShifuFileUtils.isFileExists(pathFinder.getLocalCorrelationCsvPath(), SourceType.LOCAL)) {
            return;
        }
        varSelectByCorrelation(varSelDescList);
    }

    // TODO refactor me please, bad function
    private void varSelectByCorrelation(List<VarSelDesc> varSelDescList) throws IOException {
        BufferedReader reader = ShifuFileUtils.getReader(pathFinder.getLocalCorrelationCsvPath(), SourceType.LOCAL);
        int lineNum = 0;
        try {
            String line = null;
            while((line = reader.readLine()) != null) {
                lineNum += 1;
                if(lineNum <= 2) {
                    // skip first 2 lines which are indexes and names
                    continue;
                }

                String[] columns = CommonUtils.split(line, ",");
                if(columns != null && columns.length == columnConfigList.size() + 2) {
                    int columnIndex = Integer.parseInt(columns[0].trim());
                    ColumnConfig config = this.columnConfigList.get(columnIndex);

                    // only check final-selected non-meta columns
                    if(config.isFinalSelect() || config.isTarget()) {
                        double[] corrArray = getCorrArray(columns);
                        for(int i = 0; i < corrArray.length; i++) {
                            // only check column larger than current column index and already final selected
                            if(config.getColumnNum() < i && (columnConfigList.get(i).isTarget()
                                    || columnConfigList.get(i).isFinalSelect())) {
                                // * 1.000005d is to avoid some value like 1.0000000002 in correlation value
                                if(Math.abs(corrArray[i]) > (modelConfig.getVarSelect().getCorrelationThreshold()
                                        * 1.000005d)) {
                                    if(config.isForceSelect() && columnConfigList.get(i).isForceSelect()) {
                                        LOG.warn(
                                                "{} and {} has high correlated value but both not to be removed because both are force-selected",
                                                columnIndex, i);
                                    } else if(config.isForceSelect() && !columnConfigList.get(i).isForceSelect()) {
                                        LOG.warn(
                                                "Absolute correlation value {} in column pair ({}, {}) ({}, {}) are larger than correlationThreshold value {} set in VarSelect#correlationThreshold, column {} name {} is not force-selected will not be selected, set finalSelect to false.",
                                                config.getColumnName(), columnConfigList.get(i).getColumnName(),
                                                modelConfig.getVarSelect().getCorrelationThreshold(),
                                                columnConfigList.get(i).getColumnNum(),
                                                columnConfigList.get(i).getColumnName());
                                        columnConfigList.get(i).setFinalSelect(false);
                                        varSelDescList.add(
                                                new VarSelDesc(columnConfigList.get(i), VarSelReason.HIGH_CORRELATED));
                                    } else if(!config.isForceSelect() && columnConfigList.get(i).isForceSelect()) {
                                        LOG.warn(
                                                "Absolute correlation value {} in column pair ({}, {}) ({}, {}) are larger than correlationThreshold value {} set in VarSelect#correlationThreshold, column {} name {} is not force-selected will not be selected, set finalSelect to false.",
                                                config.getColumnName(), columnConfigList.get(i).getColumnName(),
                                                modelConfig.getVarSelect().getCorrelationThreshold(),
                                                config.getColumnNum(), config.getColumnName());
                                        config.setFinalSelect(false);
                                        varSelDescList.add(new VarSelDesc(config, VarSelReason.HIGH_CORRELATED));
                                    } else if(config.isTarget() && columnConfigList.get(i).isFinalSelect()) {
                                        LOG.warn(
                                                "{} and {} has high correlated value while {} is target, {} is set to NOT final-selected no matter it is force-selected or not.",
                                                columnIndex, i, i);
                                        columnConfigList.get(i).setFinalSelect(false);
                                    } else if(config.isFinalSelect() && columnConfigList.get(i).isTarget()) {
                                        LOG.warn(
                                                "{} and {} has high correlated value while {} is target, {} is set to NOT final-selected no matter it is force-selected or not.",
                                                columnIndex, i, columnIndex);
                                        config.setFinalSelect(false);
                                        varSelDescList.add(new VarSelDesc(config, VarSelReason.HIGH_CORRELATED));
                                    } else {
                                        // both columns are not target and all final selected
                                        ColumnConfig dropConfig = null;
                                        PostCorrelationMetric corrMetric = modelConfig.getVarSelect()
                                                .getPostCorrelationMetric();
                                        if(checkCorrelationMetric(config, columnConfigList.get(i), corrMetric)) {
                                            dropConfig = columnConfigList.get(i);
                                        } else {
                                            dropConfig = config;
                                        }

                                        // if SE filterBy and SE postcorrelationMetric, seStatsMap has stats, do
                                        // correlation comparison by SE RMS value
                                        if((this.modelConfig.getVarSelectFilterBy()
                                                .equalsIgnoreCase(Constants.FILTER_BY_SE)
                                                || this.modelConfig.getVarSelectFilterBy()
                                                        .equalsIgnoreCase(Constants.FILTER_BY_ST))
                                                && corrMetric == PostCorrelationMetric.SE && this.seStatsMap != null
                                                && this.seStatsMap.get(config.getColumnNum()) != null && this.seStatsMap
                                                        .get(columnConfigList.get(i).getColumnNum()) != null) {
                                            LOG.warn(
                                                    "Absolute correlation value {} in column pair ({}, {}) ({}, {}) are larger than correlationThreshold value {} set in VarSelect#correlationThreshold, column {} name {} with smaller SE RMS value will not be selected, set finalSelect to false.",
                                                    Math.abs(corrArray[i]), config.getColumnNum(), i,
                                                    config.getColumnName(), columnConfigList.get(i).getColumnName(),
                                                    modelConfig.getVarSelect().getCorrelationThreshold(),
                                                    dropConfig.getColumnNum(), dropConfig.getColumnName());
                                        } else {
                                            LOG.info(
                                                    "Absolute correlation value {} in column pair ({}, {}) ({}, {}) are larger than correlationThreshold value {} set in VarSelect#correlationThreshold, column {} name {} with smaller {} value will not be selected, set finalSelect to false.",
                                                    Math.abs(corrArray[i]), config.getColumnNum(), i,
                                                    config.getColumnName(), columnConfigList.get(i).getColumnName(),
                                                    modelConfig.getVarSelect().getCorrelationThreshold(),
                                                    dropConfig.getColumnNum(), dropConfig.getColumnName(), corrMetric);
                                        }
                                        // de-select column which is dropped in current logic
                                        dropConfig.setFinalSelect(false);
                                        varSelDescList.add(new VarSelDesc(dropConfig, VarSelReason.HIGH_CORRELATED));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } finally {
            IOUtils.closeQuietly(reader);
        }
    }

    private boolean checkCorrelationMetric(ColumnConfig config1, ColumnConfig config2, PostCorrelationMetric metric) {
        if(metric == null) {
            return config1.getIv() > config2.getIv();
        }
        switch(metric) {
            case KS:
                return config1.getKs() > config2.getKs();
            case SE:
                if((this.modelConfig.getVarSelectFilterBy().equalsIgnoreCase(Constants.FILTER_BY_SE)
                        || this.modelConfig.getVarSelectFilterBy().equalsIgnoreCase(Constants.FILTER_BY_ST))
                        && this.seStatsMap != null && this.seStatsMap.get(config1.getColumnNum()) != null
                        && this.seStatsMap.get(config2.getColumnNum()) != null) {
                    // if bigger SE rms, means it is much important column, smaller will be dropped
                    return this.seStatsMap.get(config1.getColumnNum()).getRms() > this.seStatsMap
                            .get(config2.getColumnNum()).getRms();
                } else {
                    // not valid se, take iv
                    return config1.getIv() > config2.getIv();
                }
            case IV:
            default:
                return config1.getIv() > config2.getIv();
        }
    }

    private double[] getCorrArray(String[] columns) {
        double[] corr = new double[columns.length - 2];
        for(int i = 2; i < corr.length; i++) {
            corr[i - 2] = Double.parseDouble(columns[i].trim());
        }
        return corr;
    }

    private Set<String> guessSelectColumnSet(String varselFile) throws IOException {
        String fileName = new File(varselFile).getName();

        Set<String> toSelectedVars = new HashSet<>();

        if(fileName.endsWith("." + Constants.NN.toLowerCase())) {
            // 1. user specify .nn file as input
            List<BasicML> models = ModelSpecLoaderUtils.loadBasicModels(varselFile, ModelTrainConf.ALGORITHM.NN);
            if(CollectionUtils.isNotEmpty(models)) {
                BasicML model = models.get(0);
                if(model instanceof BasicFloatNetwork) {
                    Set<Integer> featureSets = ((BasicFloatNetwork) model).getFeatureSet();
                    for(ColumnConfig columnConfig: this.columnConfigList) {
                        if(featureSets.contains(columnConfig.getColumnNum())) {
                            toSelectedVars.add(columnConfig.getColumnName());
                        }
                    }
                }
            }
        } else if(fileName.endsWith("." + Constants.GBT.toLowerCase())
                || fileName.endsWith("." + Constants.RF.toLowerCase())) {
            // 2. user specify .gbt or .rf file as input
            IndependentTreeModel treeModel = IndependentTreeModel.loadFromStream(new FileInputStream(varselFile));
            toSelectedVars.addAll(treeModel.getNumNameMapping().values());
        } else if(fileName.startsWith(Constants.COLUMN_CONFIG_JSON_FILE_NAME)) {
            List<ColumnConfig> srcColumnConfigs = CommonUtils.loadColumnConfigList(varselFile, SourceType.LOCAL);
            for(ColumnConfig columnConfig: srcColumnConfigs) {
                if(columnConfig.isFinalSelect()) {
                    toSelectedVars.add(columnConfig.getColumnName());
                }
            }
        } else if(fileName.endsWith(".txt") || fileName.endsWith(".vars") || fileName.endsWith(".names")) {
            // 3. user specify .txt, .vars or .names file as input
            List<String> vars = FileUtils.readLines(new File(varselFile));
            for(String var: vars) {
                String fvar = StringUtils.trimToEmpty(var);
                if(fvar.startsWith("#") || fvar.startsWith("//")) {
                    continue; // skip comments
                } else {
                    toSelectedVars.add(fvar);
                }
            }
        }

        return toSelectedVars;
    }

    /**
     * Check is missing rate is over threshold.
     */
    private boolean isHighMissingRateColumn(ColumnConfig config) {
        Double missingPercentage = config.getMissingPercentage();
        return (missingPercentage != null && missingPercentage >= modelConfig.getVarSelect().getMissingRateThreshold());
    }

    /**
     * Check if column is ID-like.
     */
    @SuppressWarnings("unused")
    private boolean isIDLikeVariable(ColumnConfig config) {
        Long distinctCount = config.getColumnStats().getDistinctCount();
        Long totalCount = config.getColumnStats().getTotalCount();
        if(totalCount != null && distinctCount != null && totalCount >= 10000
                && distinctCount * 1.0 / totalCount >= 0.97d) {
            return true;
        }

        return false;
    }

    private void setHeapSizeAndSplitSize(final List<String> args) {
        // args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT, GuaguaMapReduceConstants.MAPRED_CHILD_JAVA_OPTS,
        // "-Xmn128m -Xms1G -Xmx1G -verbose:gc -XX:+PrintGCDetails -XX:+PrintGCTimeStamps"));
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, GuaguaMapReduceConstants.MAPRED_CHILD_JAVA_OPTS,
                "-Xmn128m -Xms1G -Xmx1G"));
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, GuaguaConstants.GUAGUA_SPLIT_COMBINABLE,
                Environment.getProperty(GuaguaConstants.GUAGUA_SPLIT_COMBINABLE, "true")));
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT,
                GuaguaConstants.GUAGUA_SPLIT_MAX_COMBINED_SPLIT_SIZE,
                Environment.getProperty(GuaguaConstants.GUAGUA_SPLIT_MAX_COMBINED_SPLIT_SIZE, "268435456")));
    }

    public static class VarSelPerf {
        public int columnId;
        public double sensitivityPerf;

        public static VarSelPerf create(String line) {
            line = StringUtils.trimToEmpty(line);
            String[] fields = line.split("\t");
            VarSelPerf perf = null;
            if(fields != null && fields.length == 3) {
                perf = new VarSelPerf();
                perf.columnId = Integer.parseInt(fields[0]);
                perf.sensitivityPerf = Double.parseDouble(fields[2]);
            }
            return perf;
        }

        @Override
        public String toString() {
            return this.columnId + "-->" + this.sensitivityPerf;
        }
    }

}
