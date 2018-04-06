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
import ml.shifu.shifu.container.obj.ModelVarSelectConf.PostCorrelationMetric;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.VariableSelector;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.nn.NNConstants;
import ml.shifu.shifu.core.dvarsel.*;
import ml.shifu.shifu.core.dvarsel.wrapper.CandidateGenerator;
import ml.shifu.shifu.core.dvarsel.wrapper.WrapperMasterConductor;
import ml.shifu.shifu.core.dvarsel.wrapper.WrapperWorkerConductor;
import ml.shifu.shifu.core.history.VarSelDesc;
import ml.shifu.shifu.core.history.VarSelReason;
import ml.shifu.shifu.core.mr.input.CombineInputFormat;
import ml.shifu.shifu.core.validator.ModelInspector.ModelStep;
import ml.shifu.shifu.core.varselect.ColumnInfo;
import ml.shifu.shifu.core.varselect.ColumnStatistics;
import ml.shifu.shifu.core.varselect.VarSelectMapper;
import ml.shifu.shifu.core.varselect.VarSelectReducer;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.fs.PathFinder;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.Environment;
import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.collections.ListUtils;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;
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
import org.apache.pig.impl.util.JarManager;
import org.apache.zookeeper.ZooKeeper;
import org.encog.ml.BasicML;
import org.encog.ml.data.MLDataSet;
import org.jboss.netty.bootstrap.ServerBootstrap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Reader;
import java.util.*;

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

    private final static Logger log = LoggerFactory.getLogger(VarSelectModelProcessor.class);

    @SuppressWarnings("unused")
    private static final double BAD_IV_THRESHOLD = 0.02d;

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
        log.info("Step Start: varselect");
        long start = System.currentTimeMillis();
        try {
            setUp(ModelStep.VARSELECT);
            validateParameters();

            // reset all selections if user specify or select by absolute number
            if(getIsToReset()) {
                log.info("Reset all selections data including type final select etc!");
                resetAllFinalSelect();
            } else if(getIsToList()) {
                log.info("Below variables are selected - ");
                for(ColumnConfig columnConfig: this.columnConfigList) {
                    if(columnConfig.isFinalSelect()) {
                        log.info(columnConfig.getColumnName());
                    }
                }
                log.info("-----  Done -----");
            } else if(getIsToAutoFilter()) {
                log.info("Start to run variable auto filter.");
                runAutoVarFilter();
                log.info("-----  Done -----");
            } else if(getIsRecoverAuto()) {
                String varselHistory = pathFinder.getVarSelHistory();
                if(ShifuFileUtils.isFileExists(varselHistory, SourceType.LOCAL)) {
                    log.info("!!! Auto filtered variables will be recovered from history.");
                    recoverVarselStatusFromHist(varselHistory);
                    log.info("-----  Done -----");
                } else {
                    log.warn("No variables auto filter history is found.");
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

                if(modelConfig.isRegression()) {
                    VariableSelector selector = new VariableSelector(this.modelConfig, this.columnConfigList);
                    String filterBy = this.modelConfig.getVarSelectFilterBy();
                    if(filterBy.equalsIgnoreCase(Constants.FILTER_BY_KS)
                            || filterBy.equalsIgnoreCase(Constants.FILTER_BY_IV)
                            || filterBy.equalsIgnoreCase(Constants.FILTER_BY_PARETO)
                            || filterBy.equalsIgnoreCase(Constants.FILTER_BY_MIX)) {
                        this.columnConfigList = selector.selectByFilter();
                    } else if(filterBy.equalsIgnoreCase(Constants.FILTER_BY_FI)) {
                        if(!CommonUtils.isTreeModel(modelConfig.getAlgorithm())) {
                            throw new IllegalArgumentException(
                                    "Filter by FI only works well in GBT/RF. Please check your modelconfig::train.");
                        }
                        selectByFeatureImportance();
                    } else if(filterBy.equalsIgnoreCase(Constants.FILTER_BY_SE)
                            || filterBy.equalsIgnoreCase(Constants.FILTER_BY_ST)) {
                        if(!Constants.NN.equalsIgnoreCase(modelConfig.getAlgorithm())
                                && !Constants.LR.equalsIgnoreCase(modelConfig.getAlgorithm())) {
                            throw new IllegalArgumentException(
                                    "Filter by SE/ST only works well in NN/LR. Please check your modelconfig::train.");
                        }
                        int recursiveCnt = getRecursiveCnt();
                        int i = 0;
                        // create varsel directory and write original copy of ColumnConfig.json
                        ShifuFileUtils.createDirIfNotExists(pathFinder.getVarSelDir(), SourceType.LOCAL);
                        super.saveColumnConfigList(pathFinder.getVarSelColumnConfig(i), this.columnConfigList);
                        while((i++) < recursiveCnt) {
                            distributedSEWrapper();
                            String varSelectMSEOutputPath = pathFinder
                                    .getVarSelectMSEOutputPath(modelConfig.getDataSet().getSource());
                            // even fail to run SE, still to create an empty se.x file
                            String varSelMSEHistPath = pathFinder.getVarSelMSEHistPath(i - 1);
                            ShifuFileUtils.createFileIfNotExists(varSelMSEHistPath, SourceType.LOCAL);
                            ShifuFileUtils.copyToLocal(varSelectMSEOutputPath, Constants.SHIFU_VARSELECT_SE_OUTPUT_NAME,
                                    varSelMSEHistPath);
                            // save as backup
                            super.saveColumnConfigList(pathFinder.getVarSelColumnConfig(i), this.columnConfigList);
                            // save as current copy
                            super.saveColumnConfigList();
                        }
                    } else if(filterBy.equalsIgnoreCase(Constants.FILTER_BY_VOTED)) {
                        votedVariablesSelection();
                    }
                } else {
                    // multiple classification, select all candidate at first, TODO add SE for multi-classification
                    boolean hasCandidates = CommonUtils.hasCandidateColumns(this.columnConfigList);
                    for(ColumnConfig config: this.columnConfigList) {
                        if(CommonUtils.isGoodCandidate(config, hasCandidates, modelConfig.isRegression())) {
                            config.setFinalSelect(true);
                        }
                    }
                }

                // clean shadow targets for multi-segments
                cleanShadowTargetsForSegments();

                if(modelConfig.getVarSelect().getAutoFilterEnable()) {
                    runAutoVarFilter();
                }
            }

            // save column config to file and sync to
            clearUp(ModelStep.VARSELECT);
        } catch (ShifuException e) {
            log.error("Error:" + e.getError().toString() + "; msg:" + e.getMessage(), e);
            return -1;
        } catch (Exception e) {
            log.error("Error:" + e.getMessage(), e);
            return -1;
        }
        log.info("Step Finished: varselect with {} ms", (System.currentTimeMillis() - start));
        return 0;
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
                log.info("Recover column - {} from {} to {}", varSelDesc.getColumnName(), varSelDesc.getNewSelStatus(),
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
     * @return
     * @throws IOException
     */
    private List<VarSelDesc> loadVarSelDescList(String varselHistory) throws IOException {
        Reader reader = ShifuFileUtils.getReader(varselHistory, SourceType.LOCAL);
        List<String> autoFilterList = IOUtils.readLines(reader);
        IOUtils.closeQuietly(reader);;

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
        if(!super.modelConfig.getVarSelect().getFilterEnable()) {
            models = CommonUtils.loadBasicModels(this.modelConfig, this.columnConfigList, null);
        }
        if(models == null || models.size() < 1) {
            TrainModelProcessor trainModelProcessor = new TrainModelProcessor();
            trainModelProcessor.setForVarSelect(true);
            trainModelProcessor.run();
            models = CommonUtils.loadBasicModels(this.modelConfig, this.columnConfigList, null);
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

    private void validateParameters() throws Exception {
        // String alg = super.getModelConfig().getTrain().getAlgorithm();
        String filterBy = this.modelConfig.getVarSelectFilterBy();
        if(filterBy.equalsIgnoreCase(Constants.FILTER_BY_SE) || filterBy.equalsIgnoreCase(Constants.FILTER_BY_ST)) {
            validateSEParameters();
            validateNormalize();
        }
    }

    public void resetAllFinalSelect() throws IOException {
        log.info("!!! Reset all variables finalSelect = false");
        for(ColumnConfig columnConfig: this.columnConfigList) {
            columnConfig.setFinalSelect(false);
            columnConfig.setColumnFlag(null);
        }
        saveColumnConfigList();
    }

    private void validateNormalize() throws IOException {
        if(!ShifuFileUtils.isFileExists(
                new PathFinder(modelConfig).getNormalizedDataPath(this.modelConfig.getDataSet().getSource()),
                this.modelConfig.getDataSet().getSource())) {
            throw new IllegalStateException("Cannot find normalized data, please do 'Shifu normalize' firstly.");
        }
    }

    private void validateSEParameters() {
        if(!NNConstants.NN_ALG_NAME.equalsIgnoreCase(super.getModelConfig().getTrain().getAlgorithm())
                && !"LR".equalsIgnoreCase(super.getModelConfig().getTrain().getAlgorithm())) {
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

    private void votedVariablesSelection() throws ClassNotFoundException, IOException, InterruptedException {
        log.info("Start voted variables selection ");
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
            HDPUtils.addFileToClassPath(HDPUtils.findContainingFile("hdfs-site.xml"), conf);
            HDPUtils.addFileToClassPath(HDPUtils.findContainingFile("core-site.xml"), conf);
            HDPUtils.addFileToClassPath(HDPUtils.findContainingFile("mapred-site.xml"), conf);
            HDPUtils.addFileToClassPath(HDPUtils.findContainingFile("yarn-site.xml"), conf);
        }
        guaguaClient.createJob(args.toArray(new String[0])).waitForCompletion(true);

        log.info("Voted variables selection finished in {}ms.", System.currentTimeMillis() - start);

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
            e.printStackTrace();
            return -1;
        } catch (IllegalArgumentException e) {
            e.printStackTrace();
            return -1;
        }

        return 0;
    }

    private Path getVotedSelectionPath(SourceType sourceType) {
        return ShifuFileUtils.getFileSystemBySourceType(sourceType)
                .makeQualified(new Path(getPathFinder().getVarSelsPath(sourceType), "VarSels"));
    }

    @SuppressWarnings("unused")
    private void prepareVarSelParams(final List<String> args, final SourceType sourceType) {
        args.add("-libjars");

        args.add(addRuntimeJars());

        args.add("-i");
        args.add(ShifuFileUtils.getFileSystemBySourceType(sourceType)
                .makeQualified(new Path(modelConfig.getDataSetRawPath())).toString());

        String zkServers = Environment.getProperty(Environment.ZOO_KEEPER_SERVERS);
        if(StringUtils.isEmpty(zkServers)) {
            log.warn(
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
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, CommonConstants.SHIFU_MODEL_CONFIG,
                ShifuFileUtils.getFileSystemBySourceType(sourceType)
                        .makeQualified(new Path(super.getPathFinder().getModelConfigPath(sourceType)))));
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, CommonConstants.SHIFU_COLUMN_CONFIG,
                ShifuFileUtils.getFileSystemBySourceType(sourceType)
                        .makeQualified(new Path(super.getPathFinder().getColumnConfigPath(sourceType)))));

        // source type
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, CommonConstants.MODELSET_SOURCE_TYPE,
                sourceType));

        // computation time
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT,
                GuaguaConstants.GUAGUA_COMPUTATION_TIME_THRESHOLD, 60 * 60 * 1000l));
        setHeapSizeAndSplitSize(args);

        // one can set guagua conf in shifuconfig
        for(Map.Entry<Object, Object> entry: Environment.getProperties().entrySet()) {
            if(CommonUtils.isHadoopConfigurationInjected(entry.getKey().toString())) {
                args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, entry.getKey().toString(),
                        entry.getValue().toString()));
            }
        }
    }

    // GuaguaOptionsParser doesn't to support *.jar currently.
    private String addRuntimeJars() {
        List<String> jars = new ArrayList<String>(16);
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

        jars.add(JarManager.findContainingJar(JexlException.class));

        String hdpVersion = HDPUtils.getHdpVersionForHDP224();
        if(StringUtils.isNotBlank(hdpVersion)) {
            // for hdp 2.2.4, hdp.version should be set and configuration files should be add to container class path
            jars.add(HDPUtils.findContainingFile("hdfs-site.xml"));
            jars.add(HDPUtils.findContainingFile("core-site.xml"));
            jars.add(HDPUtils.findContainingFile("mapred-site.xml"));
            jars.add(HDPUtils.findContainingFile("yarn-site.xml"));
        }

        return StringUtils.join(jars, NNConstants.LIB_JAR_SEPARATOR);
    }

    /**
     * Wrapper through {@link TrainModelProcessor} and a MapReduce job to analyze biggest sensitivity RMS.
     */
    private void distributedSEWrapper() throws Exception {
        // 1. Train a model using current selected variables, if no variables selected, use all candidate variables.
        boolean reuseCurrentModel = Environment.getBoolean("shifu.varsel.se.reuse", Boolean.FALSE);
        SourceType source = this.modelConfig.getDataSet().getSource();

        if(!reuseCurrentModel) {
            TrainModelProcessor trainModelProcessor = new TrainModelProcessor();
            trainModelProcessor.setForVarSelect(true);
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
        Job job = createSEMapReduceJob(source, conf, varSelectMSEOutputPath);

        // 2.4 clean output firstly
        ShifuFileUtils.deleteFile(varSelectMSEOutputPath, source);

        // 2.5 submit job
        if(job.waitForCompletion(true)) {
            // 2.6 post process 4 var select
            if(super.modelConfig.getVarSelect().getFilterEnable()) {
                postProcess4SEVarSelect(source, varSelectMSEOutputPath);
            } else {
                log.info("Only print sensitivity analysis report.");
                log.info(
                        "Sensitivity analysis report is in {}/{}-* file(s) with format 'column_index\tcolumn_name\tmean\trms\tvariance'.",
                        varSelectMSEOutputPath, Constants.SHIFU_VARSELECT_SE_OUTPUT_NAME);
            }
        } else {
            log.error("VarSelect SE hadoop job is failed, please re-try varselect step.");
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
            int threads;
            try {
                threads = Integer.parseInt(Environment.getProperty(Constants.SHIFU_VARSEL_SE_MULTI_THREAD,
                        Constants.SHIFU_DEFAULT_VARSEL_SE_MULTI_THREAD + ""));
            } catch (Exception e) {
                log.warn("'shifu.varsel.se.multi.thread' should be a int value, set default value: {}",
                        Constants.SHIFU_DEFAULT_VARSEL_SE_MULTI_THREAD);
                threads = Constants.SHIFU_DEFAULT_VARSEL_SE_MULTI_THREAD;
            }
            conf.setInt("mapreduce.map.cpu.vcores", threads);
            MultithreadedMapper.setNumberOfThreads(job, threads);
        } else {
            job.setMapperClass(VarSelectMapper.class);
        }
        job.setMapOutputKeyClass(LongWritable.class);
        job.setMapOutputValueClass(ColumnInfo.class);
        job.setInputFormatClass(CombineInputFormat.class);
        FileInputFormat.setInputPaths(job, ShifuFileUtils.getFileSystemBySourceType(source)
                .makeQualified(new Path(super.getPathFinder().getNormalizedDataPath())));

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

    private void prepareSEJobConf(SourceType source, Configuration conf) throws IOException {
        String modelConfigPath = ShifuFileUtils.getFileSystemBySourceType(source)
                .makeQualified(new Path(super.getPathFinder().getModelConfigPath(source))).toString();
        String columnConfigPath = ShifuFileUtils.getFileSystemBySourceType(source)
                .makeQualified(new Path(super.getPathFinder().getColumnConfigPath(source))).toString();
        // only the first model is sued for sensitivity analysis
        String seModelPath = ShifuFileUtils.getFileSystemBySourceType(source).makeQualified(
                new Path(super.getPathFinder().getModelsPath(), "model0." + modelConfig.getAlgorithm().toLowerCase()))
                .toString();
        String filePath = modelConfigPath + "," + columnConfigPath + "," + seModelPath;

        // add jars and files to hadoop mapper and reducer
        new GenericOptionsParser(conf, new String[] { "-libjars", addRuntimeJars(), "-files", filePath });

        conf.setBoolean(GuaguaMapReduceConstants.MAPRED_MAP_TASKS_SPECULATIVE_EXECUTION, true);
        conf.setBoolean(GuaguaMapReduceConstants.MAPRED_REDUCE_TASKS_SPECULATIVE_EXECUTION, true);
        conf.setBoolean(GuaguaMapReduceConstants.MAPREDUCE_MAP_SPECULATIVE, true);
        conf.setBoolean(GuaguaMapReduceConstants.MAPREDUCE_REDUCE_SPECULATIVE, true);
        conf.set(Constants.SHIFU_MODEL_CONFIG, ShifuFileUtils.getFileSystemBySourceType(source)
                .makeQualified(new Path(super.getPathFinder().getModelConfigPath(source))).toString());
        conf.set(Constants.SHIFU_COLUMN_CONFIG, ShifuFileUtils.getFileSystemBySourceType(source)
                .makeQualified(new Path(super.getPathFinder().getColumnConfigPath(source))).toString());
        conf.set(NNConstants.MAPRED_JOB_QUEUE_NAME, Environment.getProperty(Environment.HADOOP_JOB_QUEUE, "default"));
        conf.set(Constants.SHIFU_MODELSET_SOURCE_TYPE, source.toString());
        // set mapreduce.job.max.split.locations to 100 to suppress warnings
        conf.setInt(GuaguaMapReduceConstants.MAPREDUCE_JOB_MAX_SPLIT_LOCATIONS, 5000);

        // Tmp set to false because of some cluster by default use gzip while CombineInputFormat will split gzip file (a
        // bug)
        conf.setBoolean(CombineInputFormat.SHIFU_VS_SPLIT_COMBINABLE, false);
        conf.setBoolean("mapreduce.input.fileinputformat.input.dir.recursive", true);

        conf.set("mapred.reduce.slowstart.completed.maps",
                Environment.getProperty("mapred.reduce.slowstart.completed.maps", "0.9"));
        conf.set(Constants.SHIFU_VARSELECT_FILTEROUT_TYPE, modelConfig.getVarSelectFilterBy());

        Float filterOutRatio = this.modelConfig.getVarSelect().getFilterOutRatio();
        if(filterOutRatio == null) {
            log.warn("filterOutRatio in var select is not set. Using default value 0.05.");
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
            HDPUtils.addFileToClassPath(HDPUtils.findContainingFile("hdfs-site.xml"), conf);
            HDPUtils.addFileToClassPath(HDPUtils.findContainingFile("core-site.xml"), conf);
            HDPUtils.addFileToClassPath(HDPUtils.findContainingFile("mapred-site.xml"), conf);
            HDPUtils.addFileToClassPath(HDPUtils.findContainingFile("yarn-site.xml"), conf);
        }
        // one can set guagua conf in shifuconfig
        for(Map.Entry<Object, Object> entry: Environment.getProperties().entrySet()) {
            if(CommonUtils.isHadoopConfigurationInjected(entry.getKey().toString())) {
                conf.set(entry.getKey().toString(), entry.getValue().toString());
            }
        }
        // no matter how the mapreduce.task.io.sort.mb is set for sensitivity job, only 1 reducer and each mapper only
        // output column stats, 150MB is enough.
        conf.setInt("mapreduce.task.io.sort.mb", 150);
    }

    private void postProcessFIVarSelect(Map<Integer, MutablePair<String, Double>> importances) throws IOException {
        int selectCnt = 0;
        for(ColumnConfig config: super.columnConfigList) {
            // enable ForceSelect
            if(config.isForceSelect()) {
                config.setFinalSelect(true);
                selectCnt++;
                log.info("Variable {} is selected, since it is in ForceSelect list.", config.getColumnName());
            }
        }
        VariableSelector.setFilterNumberByFilterOutRatio(this.modelConfig, this.columnConfigList);
        int targetCnt = this.modelConfig.getVarSelectFilterNum();
        List<Integer> candidateColumnIdList = new ArrayList<Integer>();
        candidateColumnIdList.addAll(importances.keySet());
        int i = 0;
        int candidateCount = candidateColumnIdList.size();
        // try to select another (targetCnt - selectCnt) variables, but we need to exclude those
        // force-selected variables
        for(ColumnConfig columnConfig: this.columnConfigList) {
            if(columnConfig.isFinalSelect()) {
                columnConfig.setFinalSelect(false);
            }
        }

        Set<NSColumn> userCandidateColumns = CommonUtils.loadCandidateColumns(modelConfig);

        while(selectCnt < targetCnt && i < targetCnt) {
            if(i >= candidateCount) {
                log.warn("Var select finish due to feature importance count {} is less than target var count {}",
                        candidateCount, targetCnt);
                break;
            }
            Integer columnId = candidateColumnIdList.get(i++);
            ColumnConfig columnConfig = this.columnConfigList.get(columnId);
            if(CollectionUtils.isNotEmpty(userCandidateColumns)
                    && !userCandidateColumns.contains(new NSColumn(columnConfig.getColumnName()))) {
                log.info("Variable {} is not in user's candidate list. Skip it.", columnConfig.getColumnName());
            } else if(!columnConfig.isForceSelect() && !columnConfig.isForceRemove()) {
                columnConfig.setFinalSelect(true);
                selectCnt++;
                log.info("Variable {} is selected.", columnConfig.getColumnName());
            }
        }
        log.info("{} variables are selected.", selectCnt);
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
                log.info("Variable {} is selected, since it is in ForceSelect list.", config.getColumnName());
            }
        }

        Set<NSColumn> userCandidateColumns = CommonUtils.loadCandidateColumns(modelConfig);

        List<Scanner> scanners = null;
        try {
            // here only works for 1 reducer
            FileStatus[] globStatus = ShifuFileUtils.getFileSystemBySourceType(source)
                    .globStatus(new Path(outputFilePattern));
            if(globStatus == null || globStatus.length == 0) {
                throw new RuntimeException("Var select MSE stats output file not exist.");
            }
            scanners = ShifuFileUtils.getDataScanners(globStatus[0].getPath().toString(), source);
            String str = null;
            int targetCnt = 0; // total variable count that user want to select
            List<Integer> candidateColumnIdList = new ArrayList<Integer>();
            Scanner scanner = scanners.get(0);
            while(scanner.hasNext()) {
                ++targetCnt;
                str = scanner.nextLine().trim();
                candidateColumnIdList.add(Integer.parseInt(str));
            }

            int i = 0;
            int candidateCount = candidateColumnIdList.size();
            // try to select another (targetCnt - selectCnt) variables, but we need to exclude those
            // force-selected variables
            while(selectCnt < targetCnt && i < targetCnt) {
                if(i >= candidateCount) {
                    log.warn("Var select finish due candidate column {} is less than target var count {}",
                            candidateCount, targetCnt);
                    break;
                }
                Integer columnId = candidateColumnIdList.get(i++);
                // after supporting segments, the columns will expansion. the columnId may not the position
                // in columnConfigList. It's safe to columnId to search (make sure columnNum == columnId)
                ColumnConfig columnConfig = CommonUtils.getColumnConfig(this.columnConfigList, columnId);
                if(CollectionUtils.isNotEmpty(userCandidateColumns)
                        && !userCandidateColumns.contains(new NSColumn(columnConfig.getColumnName()))) {
                    log.info("Variable {} is not in user's candidate list. Skip it.", columnConfig.getColumnName());
                } else if(!columnConfig.isForceSelect() && !columnConfig.isForceRemove()) {
                    columnConfig.setFinalSelect(true);
                    selectCnt++;
                    log.info("Variable {} is selected.", columnConfig.getColumnName());
                }
            }

            log.info("{} variables are selected.", selectCnt);
            log.info(
                    "Sensitivity analysis report is in {}/{}-* file(s) with format 'column_index\tcolumn_name\tmean\trms\tvariance'.",
                    varSelectMSEOutputPath, Constants.SHIFU_VARSELECT_SE_OUTPUT_NAME);
            this.seStatsMap = readSEValuesToMap(
                    varSelectMSEOutputPath + Path.SEPARATOR + Constants.SHIFU_VARSELECT_SE_OUTPUT_NAME + "-*", source);
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

    private Map<Integer, ColumnStatistics> readSEValuesToMap(String seOutputFiles, SourceType source)
            throws IOException {
        // here only works for 1 reducer
        FileStatus[] globStatus = ShifuFileUtils.getFileSystemBySourceType(source).globStatus(new Path(seOutputFiles));
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
            log.info("There are segments. Set all target shadow columns to ForceRemove and final select false");
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

    /**
     *
     * @throws IOException
     */
    private void runAutoVarFilter() throws IOException {
        if(this.modelConfig.getVarSelect().getPostCorrelationMetric().equals(PostCorrelationMetric.SE)
                && this.seStatsMap == null) {
            SourceType source = this.modelConfig.getDataSet().getSource();
            String varSelectMSEOutputPath = super.getPathFinder().getVarSelectMSEOutputPath(source);
            this.seStatsMap = readSEValuesToMap(
                    varSelectMSEOutputPath + Path.SEPARATOR + Constants.SHIFU_VARSELECT_SE_OUTPUT_NAME + "-*", source);
        }

        List<VarSelDesc> varSelDescList = new ArrayList<VarSelDesc>();
        autoVarSelCondition(varSelDescList);
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
    private void autoVarSelCondition(List<VarSelDesc> varSelDescList) throws IOException {
        // here we do loop again as it is not bad for variables less than 100,000
        // 1. check missing rate
        for(ColumnConfig config: columnConfigList) {
            if(!config.isTarget() && !config.isMeta() && !config.isForceSelect() // column needs check
                    && config.isFinalSelect() && isHighMissingRateColumn(config)) {
                log.warn("Column {} is with very high missing rate, set final select to false. "
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
                    log.warn("IV of column {} is less than minimal IV threshold, set final select to false. "
                            + "If not, you can check it manually in ColumnConfig.json", config.getColumnName());
                    config.setFinalSelect(false);
                    varSelDescList.add(new VarSelDesc(config, VarSelReason.IV_TOO_LOW));
                }

                float minKsThreshold = (super.modelConfig.getVarSelect().getMinKsThreshold() == null ? 0f
                        : super.modelConfig.getVarSelect().getMinKsThreshold());
                if(config.getKs() != null && config.getKs() < minKsThreshold) {
                    log.warn("KS of column {} is less than minimal KS threshold, set final select to false. "
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
                                        log.warn(
                                                "{} and {} has high correlated value but both not to be removed because both are force-selected",
                                                columnIndex, i);
                                    } else if(config.isForceSelect() && !columnConfigList.get(i).isForceSelect()) {
                                        log.warn(
                                                "Absolute correlation value {} in column pair ({}, {}) ({}, {}) are larger than correlationThreshold value {} set in VarSelect#correlationThreshold, column {} name {} is not force-selected will not be selected, set finalSelect to false.",
                                                config.getColumnName(), columnConfigList.get(i).getColumnName(),
                                                modelConfig.getVarSelect().getCorrelationThreshold(),
                                                columnConfigList.get(i).getColumnNum(),
                                                columnConfigList.get(i).getColumnName());
                                        columnConfigList.get(i).setFinalSelect(false);
                                        varSelDescList.add(
                                                new VarSelDesc(columnConfigList.get(i), VarSelReason.HIGH_CORRELATED));
                                    } else if(!config.isForceSelect() && columnConfigList.get(i).isForceSelect()) {
                                        log.warn(
                                                "Absolute correlation value {} in column pair ({}, {}) ({}, {}) are larger than correlationThreshold value {} set in VarSelect#correlationThreshold, column {} name {} is not force-selected will not be selected, set finalSelect to false.",
                                                config.getColumnName(), columnConfigList.get(i).getColumnName(),
                                                modelConfig.getVarSelect().getCorrelationThreshold(),
                                                config.getColumnNum(), config.getColumnName());
                                        config.setFinalSelect(false);
                                        varSelDescList.add(new VarSelDesc(config, VarSelReason.HIGH_CORRELATED));
                                    } else if(config.isTarget() && columnConfigList.get(i).isFinalSelect()) {
                                        log.warn(
                                                "{} and {} has high correlated value while {} is target, {} is set to NOT final-selected no matter it is force-selected or not.",
                                                columnIndex, i, i);
                                        columnConfigList.get(i).setFinalSelect(false);
                                    } else if(config.isFinalSelect() && columnConfigList.get(i).isTarget()) {
                                        log.warn(
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
                                            log.warn(
                                                    "Absolute correlation value {} in column pair ({}, {}) ({}, {}) are larger than correlationThreshold value {} set in VarSelect#correlationThreshold, column {} name {} with smaller SE RMS value will not be selected, set finalSelect to false.",
                                                    Math.abs(corrArray[i]), config.getColumnNum(), i,
                                                    config.getColumnName(), columnConfigList.get(i).getColumnName(),
                                                    modelConfig.getVarSelect().getCorrelationThreshold(),
                                                    dropConfig.getColumnNum(), dropConfig.getColumnName());
                                        } else {
                                            log.info(
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

}
