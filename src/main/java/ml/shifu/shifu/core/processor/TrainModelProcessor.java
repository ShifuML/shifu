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
import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.Reader;
import java.lang.Thread.UncaughtExceptionHandler;
import java.lang.reflect.Array;
import java.lang.reflect.Method;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;

import org.antlr.runtime.RecognitionException;
import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.collections.ListUtils;
import org.apache.commons.collections.map.HashedMap;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.commons.lang3.tuple.MutablePair;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.pig.LoadPushDown.RequiredField;
import org.apache.pig.LoadPushDown.RequiredFieldList;
import org.apache.pig.data.DataType;
import org.apache.pig.data.Tuple;
import org.apache.pig.impl.PigContext;
import org.apache.pig.impl.util.JarManager;
import org.apache.pig.impl.util.ObjectSerializer;
import org.apache.zookeeper.ZooKeeper;
import org.encog.engine.network.activation.ActivationFunction;
import org.encog.engine.network.activation.ActivationLOG;
import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationSIN;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.ml.BasicML;
import org.encog.ml.data.MLDataSet;
import org.jboss.netty.bootstrap.ServerBootstrap;
import org.joda.time.ReadableInstant;
import org.reflections.Reflections;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.xerial.snappy.Snappy;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.base.Splitter;

import ml.shifu.guagua.GuaguaConstants;
import ml.shifu.guagua.mapreduce.GuaguaMapReduceClient;
import ml.shifu.guagua.mapreduce.GuaguaMapReduceConstants;
import ml.shifu.shifu.actor.AkkaSystemExecutor;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelNormalizeConf.NormType;
import ml.shifu.shifu.container.obj.ModelTrainConf.MultipleClassification;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.AbstractTrainer;
import ml.shifu.shifu.core.TreeModel;
import ml.shifu.shifu.core.alg.LogisticRegressionTrainer;
import ml.shifu.shifu.core.alg.NNTrainer;
import ml.shifu.shifu.core.alg.SVMTrainer;
import ml.shifu.shifu.core.alg.TensorflowTrainer;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.dtrain.FeatureSubsetStrategy;
import ml.shifu.shifu.core.dtrain.dataset.BasicFloatNetwork;
import ml.shifu.shifu.core.dtrain.dt.DTMaster;
import ml.shifu.shifu.core.dtrain.dt.DTMasterParams;
import ml.shifu.shifu.core.dtrain.dt.DTOutput;
import ml.shifu.shifu.core.dtrain.dt.DTWorker;
import ml.shifu.shifu.core.dtrain.dt.DTWorkerParams;
import ml.shifu.shifu.core.dtrain.gs.GridSearch;
import ml.shifu.shifu.core.dtrain.lr.LogisticRegressionMaster;
import ml.shifu.shifu.core.dtrain.lr.LogisticRegressionOutput;
import ml.shifu.shifu.core.dtrain.lr.LogisticRegressionParams;
import ml.shifu.shifu.core.dtrain.lr.LogisticRegressionWorker;
import ml.shifu.shifu.core.dtrain.mtl.MTLMaster;
import ml.shifu.shifu.core.dtrain.mtl.MTLOutput;
import ml.shifu.shifu.core.dtrain.mtl.MTLParams;
import ml.shifu.shifu.core.dtrain.mtl.MTLWorker;
import ml.shifu.shifu.core.dtrain.nn.ActivationLeakyReLU;
import ml.shifu.shifu.core.dtrain.nn.ActivationPTANH;
import ml.shifu.shifu.core.dtrain.nn.ActivationReLU;
import ml.shifu.shifu.core.dtrain.nn.ActivationSwish;
import ml.shifu.shifu.core.dtrain.nn.NNConstants;
import ml.shifu.shifu.core.dtrain.nn.NNMaster;
import ml.shifu.shifu.core.dtrain.nn.NNOutput;
import ml.shifu.shifu.core.dtrain.nn.NNParams;
import ml.shifu.shifu.core.dtrain.nn.NNParquetWorker;
import ml.shifu.shifu.core.dtrain.nn.NNWorker;
import ml.shifu.shifu.core.dtrain.wdl.WDLMaster;
import ml.shifu.shifu.core.dtrain.wdl.WDLOutput;
import ml.shifu.shifu.core.dtrain.wdl.WDLParams;
import ml.shifu.shifu.core.dtrain.wdl.WDLWorker;
import ml.shifu.shifu.core.validator.ModelInspector.ModelStep;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.fs.PathFinder;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.guagua.GuaguaParquetMapReduceClient;
import ml.shifu.shifu.guagua.ShifuInputFormat;
import ml.shifu.shifu.util.Base64Utils;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.Environment;
import ml.shifu.shifu.util.HDFSUtils;
import ml.shifu.shifu.util.ModelSpecLoaderUtils;
import ml.shifu.shifu.util.NormalizationUtils;
import ml.shifu.shifu.util.ValueVisitor;
import parquet.ParquetRuntimeException;
import parquet.column.ParquetProperties;
import parquet.column.values.bitpacking.Packer;
import parquet.encoding.Generator;
import parquet.format.PageType;
import parquet.hadoop.ParquetRecordReader;
import parquet.org.codehaus.jackson.Base64Variant;

import static ml.shifu.shifu.core.dtrain.CommonConstants.*;

/**
 * Train processor, produce model based on the normalized dataset.
 */
public class TrainModelProcessor extends BasicModelProcessor implements Processor {

    private final static Logger LOG = LoggerFactory.getLogger(TrainModelProcessor.class);

    private static final int VAR_SELECT_TRAINING_DECAY_EPOCHES_THRESHOLD = 400;

    private static final String SHIFU_DEFAULT_DTRAIN_PARALLEL = "true";

    /**
     * Local trainers list.
     */
    private List<AbstractTrainer> trainers = new ArrayList<AbstractTrainer>();

    /**
     * If for variable selection, only using bagging number 1 to train only one model.
     */
    private boolean isForVarSelect;

    /**
     * Will be used as the train log file prefix, when run variable selection
     */
    private String trainLogFile;

    /**
     * Implicit data cleaning is also embedded in GBDT training, shuffle such clean data or not before gbdt training.
     */
    private boolean isToShuffle = false;

    /**
     * Random generator for get sampling features per each iteration.
     */
    private Random featureSamplingRandom = new Random();

    /**
     * Configuration file used in TensorFlow on Shifu training.
     */
    private Path globalDefaultConfFile = new Path(Environment.getProperty(Environment.SHIFU_HOME) + File.separator
            + "conf" + File.separator + "global-default.xml");

    /**
     * Default constructor for model training processor.
     */
    public TrainModelProcessor() {
    }

    /**
     * Constructor for model training processor.
     * 
     * @param otherConfigs
     *            configs not from ModelConfig.json but can be set in such map.
     */
    public TrainModelProcessor(Map<String, Object> otherConfigs) {
        super.otherConfigs = otherConfigs;
    }

    /**
     * Training process entry point.
     */
    @Override
    public int run() throws Exception {
        if(!this.isForVarSelect()) {
            LOG.info("Step Start: train");
        }

        int status = 0;
        long start = System.currentTimeMillis();
        try {
            setUp(ModelStep.TRAIN);
            switch(super.modelConfig.getBasic().getRunMode()) {
                case DIST:
                case MAPRED:
                    validateDistributedTrain();
                    syncDataToHdfs(super.modelConfig.getDataSet().getSource()); // sync to HDFS to ensure consistency
                    checkAndNormDataForModels(this.isToShuffle);
                    if(Constants.TENSORFLOW.equalsIgnoreCase(modelConfig.getAlgorithm())) {
                        status = runDistributedTensorflowTrain();
                    } else {
                        status = runDistributedTrain();
                    }
                    break;
                case LOCAL:
                default:
                    runLocalTrain();
                    break;
            }

            // need sync MC.json, CC.json back to HDFS for further steps.
            syncDataToHdfs(modelConfig.getDataSet().getSource());

            clearUp(ModelStep.TRAIN);
        } catch (ShifuException e) {
            LOG.error("Error:" + e.getError().toString() + "; msg:" + e.getMessage(), e);
            return -1;
        } catch (Exception e) {
            LOG.error("Error:" + e.getMessage(), e);
            return -1;
        }

        if(!this.isForVarSelect()) {
            LOG.info("Step Finished: train with {} ms", (System.currentTimeMillis() - start));
        }
        return status;
    }

    /**
     * Local mode training entry point.
     */
    private void runLocalTrain() throws IOException {
        if(Constants.TENSORFLOW.equalsIgnoreCase(modelConfig.getAlgorithm())) {
            runLocalTensorflowTrain();
        } else {
            runLocalAkkaTrain(isForVarSelect ? 1 : modelConfig.getBaggingNum());
        }
    }

    /**
     * Run local TF script to train model.
     * 
     * @throws IOException
     *             any exception in training.
     */
    private void runLocalTensorflowTrain() throws IOException {
        List<Scanner> scanners = null;
        TensorflowTrainer trainer = new TensorflowTrainer(modelConfig, columnConfigList);
        LOG.info("Normalized data for training {}.", pathFinder.getNormalizedDataPath());
        try {
            scanners = ShifuFileUtils.getDataScanners(pathFinder.getNormalizedDataPath(),
                    modelConfig.getDataSet().getSource());
        } catch (IOException e) {
            throw new ShifuException(ShifuErrorCode.ERROR_INPUT_NOT_FOUND, e, pathFinder.getNormalizedDataPath());
        }
        if(CollectionUtils.isNotEmpty(scanners)) {
            trainer.train();
        }
        closeScanners(scanners);
    }

    /**
     * Run training process with number of bags
     *
     * @param numBags
     *            number of bags, it decide how much trainer will start training
     */
    private void runLocalAkkaTrain(int numBags) throws IOException {
        File models = new File("models");
        FileUtils.deleteDirectory(models);
        FileUtils.forceMkdir(models);

        // init trainers
        trainers.clear();
        for(int i = 0; i < numBags; i++) {
            AbstractTrainer trainer;
            if(modelConfig.getAlgorithm().equalsIgnoreCase("NN")) {
                trainer = new NNTrainer(modelConfig, i, false);
            } else if(modelConfig.getAlgorithm().equalsIgnoreCase("SVM")) {
                trainer = new SVMTrainer(this.modelConfig, i, false);
            } else if(modelConfig.getAlgorithm().equalsIgnoreCase("LR")) {
                trainer = new LogisticRegressionTrainer(this.modelConfig, i, false);
            } else {
                throw new ShifuException(ShifuErrorCode.ERROR_UNSUPPORT_ALG);
            }
            trainers.add(trainer);
        }

        List<Scanner> scanners = null;
        if(modelConfig.getAlgorithm().equalsIgnoreCase("DT")) {
            LOG.info("Raw Data: " + modelConfig.getDataSetRawPath());
            try {
                scanners = ShifuFileUtils.getDataScanners(modelConfig.getDataSetRawPath(),
                        modelConfig.getDataSet().getSource());
            } catch (IOException e) {
                throw new ShifuException(ShifuErrorCode.ERROR_INPUT_NOT_FOUND, e, pathFinder.getNormalizedDataPath());
            }
            if(CollectionUtils.isNotEmpty(scanners)) {
                AkkaSystemExecutor.getExecutor().submitDecisionTreeTrainJob(modelConfig, columnConfigList, scanners,
                        trainers);
            }
        } else {
            LOG.info("Normalized Data: " + pathFinder.getNormalizedDataPath());
            try {
                scanners = ShifuFileUtils.getDataScanners(pathFinder.getNormalizedDataPath(),
                        modelConfig.getDataSet().getSource());
            } catch (IOException e) {
                throw new ShifuException(ShifuErrorCode.ERROR_INPUT_NOT_FOUND, e, pathFinder.getNormalizedDataPath());
            }
            if(CollectionUtils.isNotEmpty(scanners)) {
                AkkaSystemExecutor.getExecutor().submitModelTrainJob(modelConfig, columnConfigList, scanners, trainers);
            }
        }

        closeScanners(scanners); // release
    }

    /**
     * Get the trainer list
     *
     * @return the trainer list
     */
    public List<AbstractTrainer> getTrainers() {
        return trainers;
    }

    /**
     * Get the trainer
     *
     * @param index
     *            the index of trainer
     * @return the trainer
     */
    public AbstractTrainer getTrainer(int index) {
        if(index >= trainers.size())
            throw new RuntimeException("Insufficient models training");
        return trainers.get(index);
    }

    /**
     * Validate if valid distributed training configurations.
     */
    private void validateDistributedTrain() throws IOException {
        String alg = super.getModelConfig().getTrain().getAlgorithm();

        if(!(CommonConstants.NN_ALG_NAME.equalsIgnoreCase(alg) // NN algorithm
                || CommonConstants.LR_ALG_NAME.equalsIgnoreCase(alg) // LR algorithm
                || CommonUtils.isTreeModel(alg) // RF or GBT algortihm
                || Constants.TENSORFLOW.equalsIgnoreCase(alg) || CommonConstants.TF_ALG_NAME.equalsIgnoreCase(alg)
                || Constants.WDL.equalsIgnoreCase(alg) || CommonConstants.MTL_ALG_NAME.equalsIgnoreCase(alg))) {
            throw new IllegalArgumentException(
                    "Currently only NN, LR, RF(RandomForest), WDL, MTL and GBDT(Gradient Boost Desicion Tree) are supported in distributed training.");
        }

        if((CommonConstants.LR_ALG_NAME.equalsIgnoreCase(alg) || CommonConstants.GBT_ALG_NAME.equalsIgnoreCase(alg))
                && modelConfig.isClassification()
                && modelConfig.getTrain().getMultiClassifyMethod() == MultipleClassification.NATIVE) {
            throw new IllegalArgumentException(
                    "Distributed LR, GBDT(Gradient Boost Desicion Tree) only support binary classification, native multiple classification is not supported.");
        }

        if(modelConfig.isClassification() && modelConfig.getTrain().isOneVsAll() && !CommonUtils.isTreeModel(alg)
                && !CommonConstants.NN_ALG_NAME.equalsIgnoreCase(alg)) {
            throw new IllegalArgumentException("Only GBT and RF and NN support OneVsAll multiple classification.");
        }

        if(super.getModelConfig().getDataSet().getSource() != SourceType.HDFS) {
            throw new IllegalArgumentException("Currently distributed training is only supported on HDFS source type.");
        }

        if(Constants.WDL.equalsIgnoreCase(alg) && this.modelConfig.getNormalize().getNormType() != NormType.ZSCALE_INDEX
                && this.modelConfig.getNormalize().getNormType() != NormType.ZSCORE_INDEX
                && this.modelConfig.getNormalize().getNormType() != NormType.WOE_INDEX
                && this.modelConfig.getNormalize().getNormType() != NormType.WOE_ZSCALE_INDEX
                && this.modelConfig.getNormalize().getNormType() != NormType.ZSCALE_APPEND_INDEX
                && this.modelConfig.getNormalize().getNormType() != NormType.ZSCORE_APPEND_INDEX
                && this.modelConfig.getNormalize().getNormType() != NormType.WOE_APPEND_INDEX
                && this.modelConfig.getNormalize().getNormType() != NormType.WOE_ZSCALE_APPEND_INDEX
                ) {
            throw new IllegalArgumentException(
                    "WDL only supports normalize#normType with ZSCALE_INDEX/ZSCORE_INDEX/WOE_INDEX/WOE_ZSCALE_INDEX, please reset and run 'shifu norm' again.");
        }

        // check if parquet format norm output is consistent with current isParquet setting.
        boolean isParquetMetaFileExist = false;
        try {
            Path filePath = new Path(super.getPathFinder().getNormalizedDataPath(), "_common_metadata");
            isParquetMetaFileExist = ShifuFileUtils
                    .getFileSystemBySourceType(super.getModelConfig().getDataSet().getSource(), filePath)
                    .exists(filePath);
        } catch (Exception e) {
            isParquetMetaFileExist = false;
        }
        if(super.modelConfig.getNormalize().getIsParquet() && !isParquetMetaFileExist) {
            throw new IllegalArgumentException("Your normalized input in "
                    + super.getPathFinder().getNormalizedDataPath()
                    + " is not parquet format. Please keep isParquet and re-run norm again and then run training step or change isParquet to false.");
        } else if(!super.modelConfig.getNormalize().getIsParquet() && isParquetMetaFileExist) {
            throw new IllegalArgumentException("Your normalized input in "
                    + super.getPathFinder().getNormalizedDataPath()
                    + " is parquet format. Please keep isParquet and re-run norm again or change isParquet directly to true.");
        }

        GridSearch gridSearch = new GridSearch(modelConfig.getTrain().getParams(),
                modelConfig.getTrain().getGridConfigFileContent());
        if(!CommonConstants.LR_ALG_NAME.equalsIgnoreCase(alg) && !CommonConstants.NN_ALG_NAME.equalsIgnoreCase(alg)
                && !CommonUtils.isTreeModel(alg) && gridSearch.hasHyperParam()) {
            // if grid search but not NN, not RF, not GBT, not LR
            throw new IllegalArgumentException("Grid search only supports NN, GBT and RF algorithms");
        }

        if(gridSearch.hasHyperParam() && super.getModelConfig().getDataSet().getSource() != SourceType.HDFS
                && modelConfig.isDistributedRunMode()) {
            // if grid search but not mapred/dist run mode, not hdfs raw data set
            throw new IllegalArgumentException("Grid search only works in distributed run mode and HDFS source type.");
        }
    }

    protected boolean useTensorFlow2() {
        return TF_V2.equals(super.modelConfig.getTrain().getParams().get(TF_Version));
    }

    protected String getConfigFileName() {
        return useTensorFlow2() ? "global-default-v2.xml" : "global-default.xml";
    }

    protected String getScriptPrefix() {
        return useTensorFlow2() ? "distributed_tf20_" : "distributed_tf_";
    }

    protected int runDistributedTensorflowTrain() throws Exception {
        LOG.info("Started distributed TensorFlow training.");
        globalDefaultConfFile = new Path(
                super.pathFinder.getAbsolutePath(new Path("conf" + File.separator + getConfigFileName()).toString()));
        LOG.info("Shifu tensorflow on yarn global default file is found in: {}.", globalDefaultConfFile);

        if(super.modelConfig.getTrain().getBaggingNum() != 1) {
            LOG.warn("Bagging tmperally is not supported, only one model can be trained even (baggingNum = {}).",
                    super.modelConfig.getTrain().getBaggingNum());
        }
        // if not continuous mode, remove tmp models to not load it in tf python, continuous mode here there is a bug
        cleanModelPath();

        final List<String> args = new ArrayList<String>();
        args.add("-libjars");
        addTensorflowRuntimeJars(args);

        // copy globalconfig example from common conf path to project folder for user to update and modify
        generateGlobalConf();
        args.add("-globalconfig"); // include python env path,
        args.add(globalDefaultConfFile.getName());

        try {
            String clazz = "ml.shifu.shifu.core.yarn.client.TensorflowClient";
            Method main = Class.forName(clazz).getMethod("main",
                    new Class[] { Array.newInstance(String.class, 0).getClass() });
            try {
                main.invoke(null, (Object) args.toArray(new String[0]));
            } catch (Exception e) {
                LOG.error("executing tensorflow client fails", e);
                return -1;
            }

            Path filePath = new Path(super.getPathFinder().getModelsPath(SourceType.HDFS));
            Path modelPath = HDFSUtils.getFS(filePath).makeQualified(filePath);
            if(ShifuFileUtils.getFileSystemBySourceType(SourceType.HDFS, modelPath).exists(modelPath)) {
                Path localModelsPath = new Path(super.getPathFinder().getModelsPath(SourceType.LOCAL));
                if(HDFSUtils.getLocalFS().exists(localModelsPath)) {
                    HDFSUtils.getLocalFS().delete(localModelsPath, true);
                }
                copyModelToLocal(null, modelPath, SourceType.HDFS);
            } else {
                LOG.warn("Model {} isn't there, training job is failed.", modelPath.toString());
            }
        } finally {
            try { // move config to a temp file for next running FIXME, how to configure two ps numbers in two trainings
                FileUtils.moveFile(new File(globalDefaultConfFile.getName().toString()),
                        new File(globalDefaultConfFile.getName() + "_" + System.currentTimeMillis()));
            } catch (Exception e) {
                LOG.warn("Failed to move tf-yarn conf file, such message can be ignored!");
            }
        }

        return 0;
    }

    private void cleanModelPath() {
        // if var select job and not continue model training
        if(this.isForVarSelect || !modelConfig.getTrain().getIsContinuous()) {
            try {
                Path tmpPath = new Path(Constants.TMP, Constants.DEFAULT_MODELS_TMP_FOLDER);
                FileSystem fs = HDFSUtils.getFS(tmpPath);
                // delete all old models if not continuous
                Path srcTmpModelPath = fs.makeQualified(new Path(super.getPathFinder().getPathBySourceType(tmpPath,
                        SourceType.HDFS)));
                Path mvTmpModelPath = new Path(srcTmpModelPath.toString() + "_" + System.currentTimeMillis());
                LOG.info("Tmp tensorflow model path has been moved to folder: {}.", mvTmpModelPath);
                fs.rename(srcTmpModelPath, mvTmpModelPath);
                fs.mkdirs(srcTmpModelPath);

                // delete all old models if not continuous
                String srcModelPath = super.getPathFinder().getModelsPath(SourceType.HDFS);
                String mvModelPath = srcModelPath + "_" + System.currentTimeMillis();
                LOG.info("Old model path has been moved to {}", mvModelPath);
                fs.rename(new Path(srcModelPath), new Path(mvModelPath));
                fs.mkdirs(new Path(srcModelPath));
                FileSystem.getLocal(HDFSUtils.getConf())
                        .delete(new Path(super.getPathFinder().getModelsPath(SourceType.LOCAL)), true);
            } catch (Exception e) {
                LOG.warn("Failed to move tmp HDFS path, such error can be ignored", e);
            }
        }
    }

    private void setSelectedTargetAndWeightColumnNumber(Configuration globalConf) {
        int targetColumnNum = -1, weightColumnNum = -1;
        List<Integer> seletectedColumnNums = new ArrayList<Integer>();
        String weightColumnName = this.modelConfig.getDataSet().getWeightColumnName();

        for(int i = 0; i < columnConfigList.size(); i++) {
            ColumnConfig cc = columnConfigList.get(i);
            if(cc.isTarget()) {
                targetColumnNum = i;
            } else if(cc.isFinalSelect()) {
                seletectedColumnNums.add(i);
            }

            if(weightColumnName.equalsIgnoreCase(cc.getColumnName())) {
                weightColumnNum = i;
            }
        }
        if(seletectedColumnNums.size() == 0) {
            boolean hasCandidate = CommonUtils.hasCandidateColumns(columnConfigList);
            for(int i = 0; i < columnConfigList.size(); i++) {
                ColumnConfig cc = columnConfigList.get(i);
                if(cc.isTarget() || cc.isMeta()) {
                    continue;
                }
                if(CommonUtils.isGoodCandidate(cc, hasCandidate)) {
                    seletectedColumnNums.add(i);
                }
            }
        }

        globalConf.set("shifu.application.target-column-number", Integer.toString(targetColumnNum));
        globalConf.set("shifu.application.weight-column-number", Integer.toString(weightColumnNum));
        globalConf.set("shifu.application.selected-column-numbers", StringUtils.join(seletectedColumnNums, ' '));
    }

    private void setSelectedColumnForWideDeep(Configuration globalConf) {
        int targetColumnNum = -1;
        int weightColumnNum = -1;
        List<Integer> seletectedNumericColumnNums = new ArrayList<Integer>();
        List<Integer> seletectedCategoryColumnNums = new ArrayList<Integer>();
        String weightColumnName = this.modelConfig.getDataSet().getWeightColumnName();

        for(int i = 0; i < columnConfigList.size(); i++) {
            ColumnConfig cc = columnConfigList.get(i);
            if(cc.isTarget()) {
                targetColumnNum = i;
            } else if(cc.isFinalSelect()) {
                if(cc.isCategorical()) {
                    seletectedCategoryColumnNums.add(i);
                } else {
                    seletectedNumericColumnNums.add(i);
                }
            }

            if(weightColumnName.equalsIgnoreCase(cc.getColumnName())) {
                weightColumnNum = i;
            }
        }
        if((seletectedNumericColumnNums.size() + seletectedCategoryColumnNums.size()) == 0) {
            boolean hasCandidate = CommonUtils.hasCandidateColumns(columnConfigList);
            for(int i = 0; i < columnConfigList.size(); i++) {
                ColumnConfig cc = columnConfigList.get(i);
                if(cc.isTarget() || cc.isMeta()) {
                    continue;
                }
                if(CommonUtils.isGoodCandidate(cc, hasCandidate)) {
                    if(cc.isCategorical()) {
                        seletectedCategoryColumnNums.add(i);
                    } else {
                        seletectedNumericColumnNums.add(i);
                    }
                }
            }
        }

        globalConf.set("shifu.application.target-column-number", Integer.toString(targetColumnNum));
        globalConf.set("shifu.application.weight-column-number", Integer.toString(weightColumnNum));
        globalConf.set("shifu.application.selected-numeric-column-numbers",
                StringUtils.join(seletectedNumericColumnNums, ' '));
        globalConf.set("shifu.application.selected-category-column-numbers",
                StringUtils.join(seletectedCategoryColumnNums, ' '));
    }

    /**
     * Configure and update some fields of conf based on current project for TF-on-Yarn training
     */
    private void generateGlobalConf() throws IOException {
        if(HDFSUtils.getLocalFS().exists(new Path(globalDefaultConfFile.getName()))) {
            LOG.warn("Project already has global conf, we will rename it and generate a new one ...");
            HDFSUtils.getLocalFS().moveToLocalFile(new Path(globalDefaultConfFile.getName()),
                    new Path(globalDefaultConfFile.getName() + "_" + System.currentTimeMillis()));
        }

        Configuration globalConf = new Configuration(false);
        globalConf.addResource(globalDefaultConfFile);

        // set training data path
        globalConf.set("shifu.application.training-data-path", super.getPathFinder().getNormalizedDataPath());

        // set workers instance number based on training data files number
        Path filePath = new Path(super.getPathFinder().getNormalizedDataPath());
        int fileNumber = HDFSUtils.getFileNumber(HDFSUtils.getFS(filePath), filePath);
        globalConf.set("shifu.worker.instances", Integer.toString(fileNumber));
        // set backup workers as 1:10
        int backupWorkerNumber = (fileNumber / 10) > 0 ? fileNumber / 10 : 1;
        globalConf.set("shifu.worker.instances.backup", Integer.toString(backupWorkerNumber));
        // set model conf
        globalConf.set("shifu.application.model-conf", super.getPathFinder().getModelConfigPath(SourceType.LOCAL));
        String delimiter = Environment.getProperty(Constants.SHIFU_OUTPUT_DATA_DELIMITER, Constants.DEFAULT_DELIMITER);
        globalConf.set(Constants.SHIFU_OUTPUT_DATA_DELIMITER, Base64Utils.base64Encode(delimiter));
        // set column conf
        globalConf.set("shifu.application.column-conf", super.getPathFinder().getColumnConfigPath(SourceType.LOCAL));

        // set python script
        if(this.modelConfig.getNormalize().getNormType() == NormType.ZSCALE_INDEX) {
            // Running wide and deep // TODO tmp for test
            globalConf.set("shifu.application.python-script-path",
                    super.getPathFinder().getScriptPath("scripts/distributed_tf_wnd_estimator_not_embed.py"));

            setSelectedColumnForWideDeep(globalConf);
        } else {
            // Running normal NN
            Object tfTypeObj = this.modelConfig.getTrain().getParams().get("TF_type");
            String tyType = tfTypeObj == null ? "keras" : tfTypeObj.toString().toLowerCase();
            String scriptPath = getScriptPrefix() + tyType + ".py";
            String currScriptPath = System.getProperty("user.dir") + File.separator + scriptPath;
            String rawScriptPath = super.getPathFinder().getScriptPath("scripts" + File.separator + scriptPath);

            if(!Files.exists(Paths.get(currScriptPath), LinkOption.NOFOLLOW_LINKS)) {
                InputStream inputStream = null;
                try {
                    inputStream = new FileInputStream(rawScriptPath);
                    Files.copy(inputStream, Paths.get(currScriptPath), StandardCopyOption.REPLACE_EXISTING);
                } finally {
                    if(inputStream != null) {
                        inputStream.close();
                    }
                }
                LOG.info(
                        "After copying, using {} for user customization, please edit it if you would like to customize your model definition.",
                        currScriptPath);
            } else {
                LOG.info(
                        "Using existing {} for user customization, please edit it if you would like to customize your model definition.",
                        currScriptPath);
            }
            globalConf.set("shifu.application.python-script-path", currScriptPath);

            // set selected column number; target column number; weight column number
            setSelectedTargetAndWeightColumnNumber(globalConf);

            // set shell to lauch python
            globalConf.set("shifu.application.python-shell-path",
                    super.getPathFinder().getScriptPath("bin/dist_pytrain.sh"));
            // set application name
            globalConf.set("shifu.application.name", "Shifu Tensorflow: " + modelConfig.getBasic().getName());
            // set yarn queue
            globalConf.set("shifu.yarn.queue", Environment.getProperty(Environment.HADOOP_JOB_QUEUE, "default"));
            // set data total count
            globalConf.set("shifu.application.total-training-data-number",
                    Long.toString(columnConfigList.get(0).getTotalCount()));
            globalConf.set("shifu.application.epochs", this.modelConfig.getTrain().getNumTrainEpochs() + "");
            // set hdfs tmp model path
            globalConf.set("shifu.application.tmp-model-path", super.getPathFinder().getTmpModelsPath(SourceType.HDFS));
            // set hdfs final model path
            globalConf.set("shifu.application.final-model-path", super.getPathFinder().getModelsPath(SourceType.HDFS));
            // add all shifuconf, this includes 'shifu train -Dk=v' <k,v> pairs and it will override default keys set
            // above.
            Properties shifuConfigMap = Environment.getProperties();
            for(Map.Entry<Object, Object> entry: shifuConfigMap.entrySet()) {
                globalConf.set(entry.getKey().toString(), entry.getValue().toString());
            }

            OutputStream os = null;
            try {
                // Write user's overridden conf to an xml to be localized.
                os = new FileOutputStream(globalDefaultConfFile.getName());
                globalConf.writeXml(os);
            } catch (IOException e) {
                throw new RuntimeException(
                        "Failed to create " + globalDefaultConfFile.getName() + " conf file. Exiting.", e);
            } finally {
                if(os != null) {
                    os.close();
                }
            }
        }
    }

    /**
     * Jars to start TF Yarn Application as tf-yarn-client role.
     * 
     * @param args
     *            the command line args
     * @throws ClassNotFoundException
     *             if class in that lib not found (class not set well in current class path)
     */
    private void addTensorflowRuntimeJars(List<String> args) throws ClassNotFoundException {
        List<String> jars = new ArrayList<String>(16);
        // zip4j-1.3.2.jar
        jars.add(JarManager.findContainingJar(Class.forName("net.lingala.zip4j.core.ZipFile")));
        // guagua-mapreduce-*.jar
        jars.add(JarManager.findContainingJar(GuaguaMapReduceConstants.class));
        // guagua-core-*.jar
        jars.add(JarManager.findContainingJar(GuaguaConstants.class));
        // shifu-*.jar
        jars.add(JarManager.findContainingJar(getClass()));
        // shifu-tensorflow-on-yarn*.jar, hard code here as core version without shifu-tensorflow won't have such jar
        jars.add(JarManager.findContainingJar(Class.forName("ml.shifu.shifu.core.yarn.client.TensorflowClient")));

        args.add(StringUtils.join(jars, NNConstants.LIB_JAR_SEPARATOR));
    }

    protected int runDistributedTrain() throws IOException, InterruptedException, ClassNotFoundException {
        LOG.info("Started distributed training.");
        int status = 0;

        Configuration conf = new Configuration();

        SourceType sourceType = super.getModelConfig().getDataSet().getSource();
        Path filePath = new Path(Constants.TMP, Constants.DEFAULT_MODELS_TMP_FOLDER);
        FileSystem fileSystem = ShifuFileUtils.getFileSystemBySourceType(sourceType, filePath);
        Path tmpModelsPath = fileSystem.makeQualified(new Path(super.getPathFinder()
                .getPathBySourceType(filePath, sourceType)));

        if(!this.modelConfig.getTrain().getIsContinuous()) {
            cleanOldModels(conf, sourceType, fileSystem, tmpModelsPath);
        }

        final List<String> args = new ArrayList<String>();

        GridSearch gs = new GridSearch(modelConfig.getTrain().getParams(),
                modelConfig.getTrain().getGridConfigFileContent());

        prepareCommonParams(gs.hasHyperParam(), args, sourceType);

        // add tmp models folder to config
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, CommonConstants.SHIFU_TMP_MODELS_FOLDER,
                tmpModelsPath.toString()));
        int baggingNum = checkBaggingNum();

        boolean isKFoldCV = false;
        Integer kCrossValidation = this.modelConfig.getTrain().getNumKFold();
        if(kCrossValidation != null && kCrossValidation > 0) {
            isKFoldCV = true;
            baggingNum = modelConfig.getTrain().getNumKFold();
            if(baggingNum != super.getModelConfig().getBaggingNum() && gs.hasHyperParam()) {
                // if it is grid search mode, then kfold mode is disabled
                LOG.warn(
                        "'train:baggingNum' is set to {} because of k-fold cross validation is enabled by 'numKFold' not -1.",
                        baggingNum);
            }
        }

        long start = System.currentTimeMillis();
        boolean isParallel = Boolean
                .valueOf(Environment.getProperty(Constants.SHIFU_DTRAIN_PARALLEL, SHIFU_DEFAULT_DTRAIN_PARALLEL))
                .booleanValue();
        GuaguaMapReduceClient guaguaClient;
        int[] inputOutputIndex;
        if(modelConfig.isMultiTask()) {
            inputOutputIndex = DTrainUtils.getInputOutputCandidateCounts(modelConfig.getNormalizeType(),
                    this.mtlColumnConfigLists.get(0));
        } else {
            inputOutputIndex = DTrainUtils.getInputOutputCandidateCounts(modelConfig.getNormalizeType(),
                    this.columnConfigList);
        }
        int inputNodeCount = inputOutputIndex[0] == 0 ? inputOutputIndex[2] : inputOutputIndex[0];
        int candidateCount = inputOutputIndex[2];
        boolean isAfterVarSelect = (inputOutputIndex[0] != 0);
        if(modelConfig.getNormalize().getIsParquet()) {
            guaguaClient = new GuaguaParquetMapReduceClient();
            checkParquetParams(args, inputNodeCount, candidateCount);
        } else {
            guaguaClient = new GuaguaMapReduceClient();
        }

        String alg = super.getModelConfig().getTrain().getAlgorithm();
        status = runDistributedBaggingTraining(status, conf, sourceType, fileSystem, args, gs, alg, baggingNum,
                isKFoldCV, isParallel, guaguaClient, inputNodeCount, isAfterVarSelect);

        if(isKFoldCV) {
            status = postProcess4KFoldCV(status, sourceType, fileSystem, kCrossValidation, start);
        } else if(gs.hasHyperParam()) {
            postProcess4HyperParamTunning(status, sourceType, fileSystem, gs, start);
        } else { // if(!gs.hasHyperParam())
            postProcess4Train(status, sourceType, fileSystem, tmpModelsPath, baggingNum, start);
        }

        if(CommonUtils.isTreeModel(modelConfig.getAlgorithm())) {
            postProcess4TreeModelFeatureImportance();
        }

        return status;
    }

    private void postProcess4TreeModelFeatureImportance() throws IOException {
        List<BasicML> models = ModelSpecLoaderUtils.loadBasicModels(this.modelConfig, null);
        // compute feature importance and write to local file after models are trained
        Map<Integer, MutablePair<String, Double>> featureImportances = CommonUtils
                .computeTreeModelFeatureImportance(models);
        String localFsFolder = pathFinder.getLocalFeatureImportanceFolder();
        String localFIPath = pathFinder.getLocalFeatureImportancePath();
        processRollupForFIFiles(localFsFolder, localFIPath);
        CommonUtils.writeFeatureImportance(localFIPath, featureImportances);
    }

    private void postProcess4Train(int status, SourceType sourceType, FileSystem fileSystem, Path tmpModelsPath,
            int baggingNum, long start) throws IOException {
        int totalModels = 0;
        int foundModels = 0;
        totalModels = baggingNum;
        // copy model files at last.
        for(int i = 0; i < baggingNum; i++) {
            String modelName = getModelName(i);
            Path modelPath = fileSystem
                    .makeQualified(new Path(super.getPathFinder().getModelsPath(sourceType), modelName));
            if(ShifuFileUtils.getFileSystemBySourceType(sourceType, modelPath).exists(modelPath)) {
                copyModelToLocal(modelName, modelPath, sourceType);
                foundModels++;
            } else {
                LOG.warn("Model {} isn't there, maybe job is failed, for bagging it can be ignored.",
                        modelPath.toString());
            }
        }

        // copy temp model files, for RF/GBT, not to copy tmp models because of larger space needed, for others
        // by default copy tmp models to local
        boolean copyTmpModelsToLocal = Boolean.TRUE.toString()
                .equalsIgnoreCase(Environment.getProperty(Constants.SHIFU_TMPMODEL_COPYTOLOCAL, "true"));
        if(copyTmpModelsToLocal) {
            copyTmpModelsToLocal(tmpModelsPath, sourceType);
        } else {
            LOG.info("Tmp models are not copied into local, please find them in hdfs path: {}", tmpModelsPath);
        }
        LOG.info("Distributed training finished in {}ms.", System.currentTimeMillis() - start);
        if(status != 0) {
            LOG.error("Error may occurred. {} / {} models are generated. Please check!", totalModels, foundModels);
        }
    }

    private void postProcess4HyperParamTunning(int status, SourceType sourceType, FileSystem fileSystem, GridSearch gs,
            long start) throws IOException {
        int totalModels = 0;
        int foundModels = 0;
        totalModels = gs.getFlattenParams().size();
        // select the best parameter composite in grid search
        LOG.info("Original grid search params: {}", modelConfig.getParams());
        Map<String, Object> params = findBestParams(sourceType, fileSystem, gs);
        // temp copy all models for evaluation
        for(int i = 0; i < totalModels; i++) {
            String modelName = getModelName(i);
            Path modelPath = fileSystem
                    .makeQualified(new Path(super.getPathFinder().getModelsPath(sourceType), modelName));
            if(ShifuFileUtils.getFileSystemBySourceType(sourceType, modelPath).exists(modelPath)) {
                copyModelToLocal(modelName, modelPath, sourceType);
                foundModels++;
            } else {
                LOG.warn("Model {} isn't there, maybe job is failed, for bagging it can be ignored.",
                        modelPath.toString());
            }
        }
        LOG.info("The best parameters in grid search is {}", params);
        LOG.info("Grid search on distributed training finished in {}ms.", System.currentTimeMillis() - start);
        if(status != 0) {
            LOG.error("Error may occurred. {} / {} models are generated. Please check!", totalModels, foundModels);
        }
    }

    private int postProcess4KFoldCV(int status, SourceType sourceType, FileSystem fileSystem, Integer kCrossValidation,
            long start) throws IOException {
        int totalModels = 0;
        int foundModels = 0;
        totalModels = kCrossValidation;
        // k-fold we also copy model files at last, such models can be used for evaluation
        for(int i = 0; i < kCrossValidation; i++) {
            String modelName = getModelName(i);
            Path modelPath = fileSystem
                    .makeQualified(new Path(super.getPathFinder().getModelsPath(sourceType), modelName));
            if(ShifuFileUtils.getFileSystemBySourceType(sourceType, modelPath).exists(modelPath)) {
                copyModelToLocal(modelName, modelPath, sourceType);
                foundModels++;
            } else {
                LOG.warn("Model {} isn't there, maybe job is failed, for bagging it can be ignored.",
                        modelPath.toString());
                status += 1;
            }
        }

        List<Double> valErrs = readAllValidationErrors(sourceType, fileSystem, kCrossValidation);
        double sum = 0d;
        for(Double err: valErrs) {
            sum += err;
        }
        LOG.info("Average validation error for current k-fold cross validation is {}.", sum / valErrs.size());
        LOG.info("K-fold cross validation on distributed training finished in {}ms.",
                System.currentTimeMillis() - start);
        if(status != 0) {
            LOG.error("Error may occurred. {} / {} models are generated. Please check!", totalModels, foundModels);
        }
        return status;
    }

    private void checkParquetParams(final List<String> args, int inputNodeCount, int candidateCount) {
        // set required field list to make sure we only load selected columns.
        RequiredFieldList requiredFieldList = new RequiredFieldList();
        boolean hasCandidates = CommonUtils.hasCandidateColumns(columnConfigList);
        for(ColumnConfig columnConfig: super.columnConfigList) {
            if(columnConfig.isTarget()) {
                requiredFieldList.add(new RequiredField(columnConfig.getColumnName(), columnConfig.getColumnNum(), null,
                        DataType.FLOAT));
            } else {
                if(inputNodeCount == candidateCount) {
                    // no any variables are selected
                    if(!columnConfig.isMeta() && !columnConfig.isTarget()
                            && CommonUtils.isGoodCandidate(columnConfig, hasCandidates)) {
                        requiredFieldList.add(new RequiredField(columnConfig.getColumnName(),
                                columnConfig.getColumnNum(), null, DataType.FLOAT));
                    }
                } else {
                    if(!columnConfig.isMeta() && !columnConfig.isTarget() && columnConfig.isFinalSelect()) {
                        requiredFieldList.add(new RequiredField(columnConfig.getColumnName(),
                                columnConfig.getColumnNum(), null, DataType.FLOAT));
                    }
                }
            }
        }
        // weight is added manually
        requiredFieldList.add(new RequiredField("weight", columnConfigList.size(), null, DataType.DOUBLE));
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, "parquet.private.pig.required.fields",
                serializeRequiredFieldList(requiredFieldList)));
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, "parquet.private.pig.column.index.access",
                "true"));
    }

    private int checkBaggingNum() {
        int baggingNum = isForVarSelect ? 1 : super.getModelConfig().getBaggingNum();
        if(modelConfig.isClassification()) {
            int classes = modelConfig.getTags().size();
            if(classes == 2) {
                baggingNum = 1; // binary classification, only need one job
            } else {
                if(modelConfig.getTrain().isOneVsAll()) {
                    // one vs all multiple classification, we need multiple bagging jobs to do ONEVSALL
                    baggingNum = modelConfig.getTags().size();
                } else {
                    // native classification, using bagging from setting job, no need set here
                }
            }
            if(baggingNum != super.getModelConfig().getBaggingNum()) {
                LOG.warn("'train:baggingNum' is set to {} because of ONEVSALL multiple classification.", baggingNum);
            }
        }
        return baggingNum;
    }

    private int runDistributedBaggingTraining(int status, Configuration conf, SourceType sourceType,
            FileSystem fileSystem, final List<String> args, GridSearch gs, String alg, int baggingNum,
            boolean isKFoldCV, boolean isParallel, GuaguaMapReduceClient guaguaClient, int inputNodeCount,
            boolean isAfterVarSelect) throws IOException, InterruptedException, ClassNotFoundException {
        int parallelNum = Integer
                .parseInt(Environment.getProperty(CommonConstants.SHIFU_TRAIN_BAGGING_INPARALLEL, "5"));
        int parallelGroups = 1;
        if(gs.hasHyperParam()) {
            parallelGroups = (gs.getFlattenParams().size() % parallelNum == 0
                    ? gs.getFlattenParams().size() / parallelNum
                    : gs.getFlattenParams().size() / parallelNum + 1);
            baggingNum = gs.getFlattenParams().size();
            LOG.warn("'train:baggingNum' is set to {} because of grid search enabled by settings in 'train#params'.",
                    gs.getFlattenParams().size());

        } else {
            parallelGroups = baggingNum % parallelNum == 0 ? baggingNum / parallelNum : baggingNum / parallelNum + 1;
        }

        LOG.info("Distributed trainning with baggingNum: {}", baggingNum);
        List<String> progressLogList = new ArrayList<String>(baggingNum);
        boolean isOneJobNotContinuous = false;
        for(int j = 0; j < parallelGroups; j++) {
            int currBags = getCurrBags(gs, baggingNum, parallelNum, parallelGroups, j);
            for(int k = 0; k < currBags; k++) {
                int i = j * parallelNum + k;
                if(gs.hasHyperParam()) {
                    LOG.info("Start the {}th grid search job with params: {}", i, gs.getParams(i));
                } else if(isKFoldCV) {
                    LOG.info("Start the {}th k-fold cross validation job with params.", i);
                }
                List<String> localArgs = new ArrayList<String>(args);
                // set name for each bagging job.
                localArgs.add("-n");
                localArgs.add(String.format("Shifu Master-Workers %s Training Iteration: %s id:%s", alg,
                        super.getModelConfig().getModelSetName(), i));
                LOG.info("Start trainer with id: {}", i);
                String modelName = getModelName(i);
                Path modelPath = fileSystem
                        .makeQualified(new Path(super.getPathFinder().getModelsPath(sourceType), modelName));
                Path bModelPath = fileSystem
                        .makeQualified(new Path(super.getPathFinder().getBinaryModelsPath(sourceType), modelName));

                // check if job is continuous training, this can be set multiple times and we only get last one
                boolean isContinuous = false;
                if(gs.hasHyperParam()) {
                    isContinuous = false;
                } else {
                    int intContinuous = checkContinuousTraining(fileSystem, localArgs, modelPath,
                            modelConfig.getTrain().getParams());
                    if(intContinuous == -1) {
                        LOG.warn(
                                "Model with index {} with size of trees is over treeNum, such training will not be started.",
                                i);
                        continue;
                    } else {
                        isContinuous = (intContinuous == 1);
                    }
                }

                // gs not support continuous model training, k-fold cross validation is not continuous training
                if(gs.hasHyperParam() || isKFoldCV) {
                    isContinuous = false;
                }
                if(!isContinuous && !isOneJobNotContinuous) {
                    isOneJobNotContinuous = true;
                    // delete all old models if not continuous
                    String srcModelPath = super.getPathFinder().getModelsPath(sourceType);
                    String mvModelPath = srcModelPath + "_" + System.currentTimeMillis();
                    LOG.info("Old model path has been moved to {}", mvModelPath);
                    fileSystem.rename(new Path(srcModelPath), new Path(mvModelPath));
                    fileSystem.mkdirs(new Path(srcModelPath));
                    FileSystem.getLocal(conf).delete(new Path(super.getPathFinder().getModelsPath(SourceType.LOCAL)),
                            true);
                }

                if(CommonConstants.NN_ALG_NAME.equalsIgnoreCase(alg)) {
                    // tree related parameters initialization
                    setDistributedNNFeatureSubsetParams(gs, inputNodeCount, isAfterVarSelect, i, localArgs, modelPath,
                            isContinuous);
                }

                localArgs.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, CommonConstants.GUAGUA_OUTPUT,
                        modelPath.toString()));
                localArgs.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT,
                        Constants.SHIFU_BINARY_MODEL_PATH, bModelPath.toString()));

                if(gs.hasHyperParam() || isKFoldCV) {
                    // k-fold cv need val error
                    Path valErrPath = fileSystem.makeQualified(
                            new Path(super.getPathFinder().getValErrorPath(sourceType), "val_error_" + i));
                    localArgs.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT,
                            CommonConstants.GS_VALIDATION_ERROR, valErrPath.toString()));
                }
                localArgs.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, CommonConstants.SHIFU_TRAINER_ID,
                        String.valueOf(i)));
                final String progressLogFile = getProgressLogFile(i);
                progressLogList.add(progressLogFile);

                Path progressFilePath = new Path(progressLogFile);
                localArgs.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT,
                        CommonConstants.SHIFU_DTRAIN_PROGRESS_FILE,
                        HDFSUtils.getFS(progressFilePath).makeQualified(progressFilePath).toString()));

                if(isParallel) {
                    guaguaClient.addJob(localArgs.toArray(new String[0]));
                } else {
                    TailThread tailThread = startTailThread(new String[] { progressLogFile });
                    boolean ret = guaguaClient.createJob(localArgs.toArray(new String[0])).waitForCompletion(true);
                    status += (ret ? 0 : 1);
                    stopTailThread(tailThread);
                }
            }

            if(isParallel) {
                TailThread tailThread = startTailThread(progressLogList.toArray(new String[0]));
                status += guaguaClient.run();
                stopTailThread(tailThread);
            }
        }
        return status;
    }

    private void setDistributedNNFeatureSubsetParams(GridSearch gs, int inputNodeCount, boolean isAfterVarSelect, int i,
            List<String> localArgs, Path modelPath, boolean isContinuous) throws IOException {
        Map<String, Object> params = gs.hasHyperParam() ? gs.getParams(i) : this.modelConfig.getTrain().getParams();
        Object fssObj = params.get("FeatureSubsetStrategy");
        FeatureSubsetStrategy featureSubsetStrategy = null;
        double featureSubsetRate = 0d;
        if(fssObj != null) {
            try {
                featureSubsetRate = Double.parseDouble(fssObj.toString());
                // no need validate featureSubsetRate is in (0,1], as already validated in ModelInspector
                featureSubsetStrategy = null;
            } catch (NumberFormatException ee) {
                featureSubsetStrategy = FeatureSubsetStrategy.of(fssObj.toString());
            }
        } else {
            LOG.warn("FeatureSubsetStrategy is not set, set to ALL by default.");
            featureSubsetStrategy = FeatureSubsetStrategy.ALL;
            featureSubsetRate = 0;
        }

        // cache all feature list for sampling features
        List<Integer> allFeatures = NormalizationUtils.getAllFeatureList(this.columnConfigList, isAfterVarSelect);

        Set<Integer> subFeatures = null;
        if(isContinuous) {
            BasicFloatNetwork existingModel = (BasicFloatNetwork) ModelSpecLoaderUtils
                    .getBasicNetwork(ModelSpecLoaderUtils.loadModel(modelConfig, modelPath,
                            ShifuFileUtils.getFileSystemBySourceType(this.modelConfig.getDataSet().getSource(), modelPath)));
            if(existingModel == null) {
                subFeatures = new HashSet<Integer>(
                        getSubsamplingFeatures(allFeatures, featureSubsetStrategy, featureSubsetRate, inputNodeCount));
            } else {
                subFeatures = existingModel.getFeatureSet();
            }
        } else {
            subFeatures = new HashSet<Integer>(
                    getSubsamplingFeatures(allFeatures, featureSubsetStrategy, featureSubsetRate, inputNodeCount));
        }
        if(subFeatures == null || subFeatures.size() == 0) {
            localArgs.add(
                    String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, CommonConstants.SHIFU_NN_FEATURE_SUBSET, ""));
        } else {
            localArgs.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, CommonConstants.SHIFU_NN_FEATURE_SUBSET,
                    StringUtils.join(subFeatures, ',')));
            LOG.debug("Size: {}, list: {}.", subFeatures.size(), StringUtils.join(subFeatures, ','));
        }
    }

    private int getCurrBags(GridSearch gs, int baggingNum, int parallelNum, int parallelGroups, int j) {
        int currBags = baggingNum;
        if(gs.hasHyperParam()) {
            if(j == parallelGroups - 1) {
                currBags = gs.getFlattenParams().size() % parallelNum == 0 ? parallelNum
                        : gs.getFlattenParams().size() % parallelNum;
            } else {
                currBags = parallelNum;
            }
        } else {
            if(j == parallelGroups - 1) {
                currBags = baggingNum % parallelNum == 0 ? parallelNum : baggingNum % parallelNum;
            } else {
                currBags = parallelNum;
            }
        }
        return currBags;
    }

    private void cleanOldModels(Configuration conf, SourceType sourceType, FileSystem fileSystem, Path tmpModelsPath)
            throws IOException {
        // mv all old models if not continuous (move for backup, if delete, no any backups)
        String srcModelPath = super.getPathFinder().getModelsPath(sourceType);
        String mvModelPath = srcModelPath + "_" + System.currentTimeMillis();
        LOG.info("Old model path has been moved to {}", mvModelPath);
        fileSystem.rename(new Path(srcModelPath), new Path(mvModelPath));
        fileSystem.mkdirs(new Path(srcModelPath));
        FileSystem.getLocal(conf).delete(new Path(super.getPathFinder().getModelsPath(SourceType.LOCAL)), true);
        // delete tmp model folder
        fileSystem.delete(tmpModelsPath, true);
        fileSystem.mkdirs(tmpModelsPath);
    }

    /**
     * Rollup feature importance file to keep latest one and old ones.
     */
    private void processRollupForFIFiles(String localFsFolder, String fiFile) {
        try {
            Path filePath = new Path(localFsFolder);
            FileSystem fs = ShifuFileUtils.getFileSystemBySourceType(SourceType.LOCAL, filePath);
            if(!fs.isDirectory(filePath)) {
                return;
            }
            FileStatus[] fss = fs.listStatus(new Path(localFsFolder));
            if(fss != null && fss.length > 0) {
                Arrays.sort(fss, new Comparator<FileStatus>() {
                    @Override
                    public int compare(FileStatus o1, FileStatus o2) {
                        return o2.getPath().toString().compareTo(o1.getPath().toString());
                    }
                });
                for(FileStatus fileStatus: fss) {
                    String strPath = fileStatus.getPath().getName();
                    if(strPath.endsWith(PathFinder.FEATURE_IMPORTANCE_FILE)) {
                        fs.rename(fileStatus.getPath(), new Path(fileStatus.getPath() + ".1"));
                    } else if(strPath.contains(PathFinder.FEATURE_IMPORTANCE_FILE)) {
                        int lastDotIndex = strPath.lastIndexOf(".");
                        String lastIndexStr = strPath.substring(lastDotIndex + 1, strPath.length());
                        int index = Integer.parseInt(lastIndexStr);
                        fs.rename(fileStatus.getPath(), new Path(fiFile + "." + (index + 1)));
                    }
                }
            }
        } catch (Exception ignore) {
            // any exception we can ignore, just override old all.fi files
        }
    }

    private Map<String, Object> findBestParams(SourceType sourceType, FileSystem fileSystem, GridSearch gs)
            throws IOException {
        // read validation error and find the best one update ModelConfig.
        double minValErr = Double.MAX_VALUE;
        int minIndex = -1;
        for(int i = 0; i < gs.getFlattenParams().size(); i++) {
            Path valErrPath = fileSystem
                    .makeQualified(new Path(super.getPathFinder().getValErrorPath(sourceType), "val_error_" + i));
            if(ShifuFileUtils.isFileExists(valErrPath.toString(), sourceType)) {
                double valErr;
                BufferedReader reader = null;
                try {
                    reader = ShifuFileUtils.getReader(valErrPath.toString(), sourceType);
                    String line = reader.readLine();
                    if(line == null) {
                        continue;
                    }
                    String valErrStr = line.toString();
                    LOG.debug("valErrStr is {}", valErrStr);
                    valErr = Double.valueOf(valErrStr);
                } catch (NumberFormatException e) {
                    LOG.warn("Parse val error failed, ignore such error. Message: {}", e.getMessage());
                    continue;
                } finally {
                    if(reader != null) {
                        reader.close();
                    }
                }
                if(valErr < minValErr) {
                    minValErr = valErr;
                    minIndex = i;
                }
            }
        }
        Map<String, Object> params = gs.getParams(minIndex);
        LOG.info(
                "The {} params is selected by grid search with params {}, please use it and set it in ModelConfig.json.",
                minIndex, params);
        return params;
    }

    private List<Double> readAllValidationErrors(SourceType sourceType, FileSystem fileSystem, int k)
            throws IOException {
        List<Double> valErrs = new ArrayList<Double>();
        for(int i = 0; i < k; i++) {
            Path valErrPath = fileSystem
                    .makeQualified(new Path(super.getPathFinder().getValErrorPath(sourceType), "val_error_" + i));
            if(ShifuFileUtils.isFileExists(valErrPath.toString(), sourceType)) {
                double valErr;
                BufferedReader reader = null;
                try {
                    reader = ShifuFileUtils.getReader(valErrPath.toString(), sourceType);
                    String line = reader.readLine();
                    if(line == null) {
                        continue;
                    }
                    String valErrStr = line.toString();
                    LOG.debug("valErrStr is {}", valErrStr);
                    valErr = Double.valueOf(valErrStr);
                    valErrs.add(valErr);
                } catch (NumberFormatException e) {
                    LOG.warn("Parse val error failed, ignore such error. Message: {}", e.getMessage());
                    continue;
                } finally {
                    if(reader != null) {
                        reader.close();
                    }
                }
            }
        }
        return valErrs;
    }

    private static String serializeRequiredFieldList(RequiredFieldList requiredFieldList) {
        try {
            return ObjectSerializer.serialize(requiredFieldList);
        } catch (IOException e) {
            throw new RuntimeException("Failed to searlize required fields.", e);
        }
    }

    /*
     * Return 1, continuous training, 0, not continuous training, -1 GBT existing trees is over treeNum
     */
    private int checkContinuousTraining(FileSystem fileSystem, List<String> localArgs, Path modelPath,
            Map<String, Object> modelParams) throws IOException {
        int finalContinuous = 0;
        if(Boolean.TRUE.toString().equals(this.modelConfig.getTrain().getIsContinuous().toString())) {
            // if varselect d-training or no such existing models, directly to disable continuous training.
            if(this.isForVarSelect) {
                finalContinuous = 0;
                LOG.warn("For varSelect step, continous model training is always disabled.");
            } else if(!fileSystem.exists(modelPath)) {
                finalContinuous = 0;
                LOG.info("No existing model, model training will start from scratch.");
            } else if(CommonConstants.NN_ALG_NAME.equalsIgnoreCase(modelConfig.getAlgorithm())
                    && !inputOutputModelCheckSuccess(fileSystem, modelPath, modelParams)) {
                finalContinuous = 0; // TODO hidden layer size and activation functions should also be validated
                LOG.warn(
                        "!!! Model training parameters like hidden nodes, activation and others  are not consistent with settings, model training will start from scratch.");
            } else if(CommonConstants.GBT_ALG_NAME.equalsIgnoreCase(modelConfig.getAlgorithm())) {
                TreeModel model = (TreeModel) ModelSpecLoaderUtils.loadModel(this.modelConfig, modelPath, fileSystem);

                if(!model.getAlgorithm().equalsIgnoreCase(modelConfig.getAlgorithm())) {
                    finalContinuous = 0;
                    LOG.warn("Only GBT supports continuous training, while not GBT, will start from scratch");
                } else if(!model.getLossStr()
                        .equalsIgnoreCase(this.modelConfig.getTrain().getParams().get("Loss").toString())) {
                    finalContinuous = 0;
                    LOG.warn("Loss is changed, continuous training is disabled, will start from scratch");
                } else if(model.getTrees().size() == 0) {
                    finalContinuous = 0;
                } else if(model.getTrees().size() >= Integer
                        .valueOf(modelConfig.getTrain().getParams().get("TreeNum").toString())) {
                    // if over TreeNum, return -1;
                    finalContinuous = -1;
                } else {
                    finalContinuous = 1;
                }
            } else if(CommonConstants.RF_ALG_NAME.equalsIgnoreCase(modelConfig.getAlgorithm())) {
                finalContinuous = 0;
                LOG.warn("RF doesn't support continuous training");
            } else {
                finalContinuous = 1;
            }
        } else {
            finalContinuous = 0;
        }
        localArgs.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, CommonConstants.CONTINUOUS_TRAINING,
                finalContinuous == 1 ? "true" : "false"));
        return finalContinuous;
    }

    @SuppressWarnings("unchecked")
    private boolean inputOutputModelCheckSuccess(FileSystem fileSystem, Path modelPath, Map<String, Object> modelParams)
            throws IOException {
        BasicML basicML = ModelSpecLoaderUtils.loadModel(this.modelConfig, modelPath, fileSystem);
        BasicFloatNetwork model = (BasicFloatNetwork) ModelSpecLoaderUtils.getBasicNetwork(basicML);

        int[] outputCandidateCounts = DTrainUtils.getInputOutputCandidateCounts(modelConfig.getNormalizeType(),
                getColumnConfigList());
        int inputs = outputCandidateCounts[0] == 0 ? outputCandidateCounts[2] : outputCandidateCounts[0];
        boolean isInputOutConsistent = model.getInputCount() <= inputs
                && model.getOutputCount() == outputCandidateCounts[1];

        if(!isInputOutConsistent) {
            return false;
        }

        // same hidden layer ?
        boolean isHasSameHiddenLayer = (model.getLayerCount() - 2) <= (Integer) modelParams
                .get(CommonConstants.NUM_HIDDEN_LAYERS);
        if(!isHasSameHiddenLayer) {
            return false;
        }

        // same hidden nodes; same hidden activiations
        boolean isHasSameHiddenNodes = true, isHasSameHiddenActivation = true;
        List<Integer> hiddenNodeList = (List<Integer>) modelParams.get(CommonConstants.NUM_HIDDEN_NODES);
        List<String> actFuncList = (List<String>) modelParams.get(CommonConstants.ACTIVATION_FUNC);
        for(int i = 1; i < model.getLayerCount() - 1; i++) {
            if(model.getLayerNeuronCount(i) > hiddenNodeList.get(i - 1)) {
                isHasSameHiddenNodes = false;
                break;
            }

            ActivationFunction activation = model.getActivation(i);
            String actFunc = actFuncList.get(i - 1);
            if(actFunc.equalsIgnoreCase(NNConstants.NN_LINEAR)) {
                isHasSameHiddenActivation = ActivationLinear.class == activation.getClass();
            } else if(actFunc.equalsIgnoreCase(NNConstants.NN_SIGMOID)) {
                isHasSameHiddenActivation = ActivationSigmoid.class == activation.getClass();
            } else if(actFunc.equalsIgnoreCase(NNConstants.NN_TANH)) {
                isHasSameHiddenActivation = ActivationTANH.class == activation.getClass();
            } else if(actFunc.equalsIgnoreCase(NNConstants.NN_LOG)) {
                isHasSameHiddenActivation = ActivationLOG.class == activation.getClass();
            } else if(actFunc.equalsIgnoreCase(NNConstants.NN_SIN)) {
                isHasSameHiddenActivation = ActivationSIN.class == activation.getClass();
            } else if(actFunc.equalsIgnoreCase(NNConstants.NN_RELU)) {
                isHasSameHiddenActivation = ActivationReLU.class == activation.getClass();
            } else if(actFunc.equalsIgnoreCase(NNConstants.NN_LEAKY_RELU)) {
                isHasSameHiddenActivation = ActivationLeakyReLU.class == activation.getClass();
            } else if(actFunc.equalsIgnoreCase(NNConstants.NN_SWISH)) {
                isHasSameHiddenActivation = ActivationSwish.class == activation.getClass();
            } else if(actFunc.equalsIgnoreCase(NNConstants.NN_PTANH)) {
                isHasSameHiddenActivation = ActivationPTANH.class == activation.getClass();
            } else {
                isHasSameHiddenActivation = ActivationSigmoid.class == activation.getClass();
            }
            if(!isHasSameHiddenActivation) {
                break;
            }
        }
        return isHasSameHiddenNodes && isHasSameHiddenActivation;
    }

    private String getProgressLogFile(int i) {
        long currTimeMillis = System.currentTimeMillis();
        return String.format("tmp_%s/%s_%s.log", currTimeMillis, currTimeMillis, i);
    }

    private void stopTailThread(TailThread thread) throws IOException {
        thread.interrupt();
        try {
            thread.join(NNConstants.DEFAULT_JOIN_TIME);
        } catch (InterruptedException e) {
            LOG.error("Thread stopped!", e);
            Thread.currentThread().interrupt();
        }
        // delete progress file at last
        thread.deleteProgressFiles(trainLogFile);
    }

    private TailThread startTailThread(final String[] progressLog) {
        TailThread thread = new TailThread(progressLog);
        thread.setName("Training Progress");
        thread.setDaemon(true);
        thread.setUncaughtExceptionHandler(new UncaughtExceptionHandler() {
            @Override
            public void uncaughtException(Thread t, Throwable e) {
                LOG.warn(String.format("Error in thread %s: %s", t.getName(), e.getMessage()));
            }
        });
        thread.start();
        return thread;
    }

    private void copyTmpModelsToLocal(final Path tmpModelsDir, final SourceType sourceType) throws IOException {
        // copy all tmp nn to local, these tmp nn are outputs from
        if(ShifuFileUtils.getFileSystemBySourceType(sourceType, tmpModelsDir).exists(tmpModelsDir)) {
            Path localTmpModelsFolder = new Path(Constants.MODELS_TMP);
            HDFSUtils.getLocalFS().delete(localTmpModelsFolder, true);
            HDFSUtils.getLocalFS().mkdirs(localTmpModelsFolder);
            ShifuFileUtils.getFileSystemBySourceType(sourceType, tmpModelsDir)
                    .copyToLocalFile(tmpModelsDir, localTmpModelsFolder);
        }
    }

    private void prepareDTParams(final List<String> args, final SourceType sourceType) {
        args.add("-w");
        args.add(DTWorker.class.getName());
        args.add("-m");
        args.add(DTMaster.class.getName());
        args.add("-mr");
        args.add(DTMasterParams.class.getName());
        args.add("-wr");
        args.add(DTWorkerParams.class.getName());
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, GuaguaConstants.GUAGUA_MASTER_INTERCEPTERS,
                DTOutput.class.getName()));
    }

    private void prepareLRParams(final List<String> args, final SourceType sourceType) {
        args.add("-w");
        args.add(LogisticRegressionWorker.class.getName());
        args.add("-m");
        args.add(LogisticRegressionMaster.class.getName());
        args.add("-mr");
        args.add(LogisticRegressionParams.class.getName());
        args.add("-wr");
        args.add(LogisticRegressionParams.class.getName());
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, GuaguaConstants.GUAGUA_MASTER_INTERCEPTERS,
                LogisticRegressionOutput.class.getName()));
    }

    private void prepareNNParams(final List<String> args, final SourceType sourceType) {
        args.add("-w");
        if(modelConfig.getNormalize().getIsParquet()) {
            args.add(NNParquetWorker.class.getName());
        } else {
            args.add(NNWorker.class.getName());
        }

        args.add("-m");
        args.add(NNMaster.class.getName());

        args.add("-mr");
        args.add(NNParams.class.getName());

        args.add("-wr");
        args.add(NNParams.class.getName());
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, GuaguaConstants.GUAGUA_MASTER_INTERCEPTERS,
                NNOutput.class.getName()));
    }

    private void prepareCommonParams(boolean isGsMode, final List<String> args, final SourceType sourceType)
            throws IOException {
        String alg = super.getModelConfig().getTrain().getAlgorithm();

        args.add("-libjars");
        addRuntimeJars(args);

        args.add("-i");
        if(CommonUtils.isTreeModel(alg)) {
            Path filePath = new Path(super.getPathFinder().getCleanedDataPath());
            args.add(ShifuFileUtils.getFileSystemBySourceType(sourceType, filePath)
                    .makeQualified(filePath).toString());
        } else {
            Path filePath = new Path(super.getPathFinder().getNormalizedDataPath());
            args.add(ShifuFileUtils.getFileSystemBySourceType(sourceType, filePath)
                    .makeQualified(filePath).toString());
        }

        if(StringUtils.isNotBlank(modelConfig.getValidationDataSetRawPath())) {
            args.add("-inputformat");
            args.add(ShifuInputFormat.class.getName());
        }

        String zkServers = Environment.getProperty(Environment.ZOO_KEEPER_SERVERS);
        if(StringUtils.isEmpty(zkServers)) {
            LOG.warn(
                    "No specified zookeeper settings from zookeeperServers in shifuConfig file, Guagua will set embedded zookeeper server in client process or master node. For fail-over zookeeper applications, specified zookeeper servers are strongly recommended.");
        } else {
            args.add("-z");
            args.add(zkServers);
        }

        if(CommonConstants.LR_ALG_NAME.equalsIgnoreCase(alg)) {
            this.prepareLRParams(args, sourceType);
        } else if(CommonConstants.NN_ALG_NAME.equalsIgnoreCase(alg)) {
            this.prepareNNParams(args, sourceType);
        } else if(CommonUtils.isTreeModel(alg)) {
            this.prepareDTParams(args, sourceType);
        } else if(CommonConstants.WDL_ALG_NAME.equalsIgnoreCase(alg)) {
            this.prepareWDLParams(args, sourceType);
        } else if(CommonConstants.MTL_ALG_NAME.equalsIgnoreCase(alg)) {
            this.prepareMTLParams(args, sourceType);
        }

        args.add("-c");
        int numTrainEpoches = super.getModelConfig().getTrain().getNumTrainEpochs();
        // only for NN varselect, use half of epochs for sensitivity analysis
        // if for gs mode, half of iterations are used
        LOG.debug("this.isForVarSelect() - {}, isGsMode - {}", this.isForVarSelect(), isGsMode);
        if(CommonConstants.NN_ALG_NAME.equalsIgnoreCase(alg) && (this.isForVarSelect() || isGsMode)
                && numTrainEpoches >= VAR_SELECT_TRAINING_DECAY_EPOCHES_THRESHOLD) {
            numTrainEpoches = numTrainEpoches / 2;
        }

        // if GBDT or RF, such iteration should be extended to make sure all trees will be executed successfully without
        // maxIteration limitation
        if(CommonUtils.isTreeModel(alg) && numTrainEpoches <= 50000) {
            numTrainEpoches = 50000;
        }
        // the reason to add 1 is that the first iteration in implementation is used for training preparation.
        numTrainEpoches = numTrainEpoches + 1;

        if(CommonConstants.LR_ALG_NAME.equalsIgnoreCase(alg)) {
            LOG.info("Number of train iterations is set to {}.", numTrainEpoches - 1);
        } else if(CommonConstants.NN_ALG_NAME.equalsIgnoreCase(alg)) {
            LOG.info("Number of train epochs is set to {}.", numTrainEpoches - 1);
        } else if(CommonUtils.isTreeModel(alg)) {
            LOG.info("Number of train iterations is set to {}.", numTrainEpoches - 1);
        }

        args.add(String.valueOf(numTrainEpoches));

        if(CommonUtils.isTreeModel(alg)) {
            // for tree models, using cleaned validation data path
            Path filePath = new Path(super.getPathFinder().getCleanedValidationDataPath(sourceType));
            args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, CommonConstants.CROSS_VALIDATION_DIR,
                    ShifuFileUtils.getFileSystemBySourceType(sourceType, filePath)
                            .makeQualified(filePath).toString()));
        } else {
            Path filePath = new Path(super.getPathFinder().getNormalizedValidationDataPath(sourceType));
            args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, CommonConstants.CROSS_VALIDATION_DIR,
                    ShifuFileUtils.getFileSystemBySourceType(sourceType, filePath)
                            .makeQualified(filePath).toString()));
        }
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, NNConstants.MAPRED_JOB_QUEUE_NAME,
                Environment.getProperty(Environment.HADOOP_JOB_QUEUE, Constants.DEFAULT_JOB_QUEUE)));
        Path modelConfPath = new Path(super.getPathFinder().getModelConfigPath(sourceType));
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, CommonConstants.SHIFU_MODEL_CONFIG,
                ShifuFileUtils.getFileSystemBySourceType(sourceType, modelConfPath).makeQualified(modelConfPath)));
        Path columnConfPath = new Path(super.getPathFinder().getColumnConfigPath(sourceType));
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, CommonConstants.SHIFU_COLUMN_CONFIG,
                ShifuFileUtils.getFileSystemBySourceType(sourceType, columnConfPath).makeQualified(columnConfPath)));
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, CommonConstants.MODELSET_SOURCE_TYPE,
                sourceType));
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, NNConstants.NN_POISON_SAMPLER,
                Environment.getProperty(NNConstants.NN_POISON_SAMPLER, "true")));
        // hard code set computation threshold for 50s. Can be changed in shifuconfig file
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT,
                GuaguaConstants.GUAGUA_COMPUTATION_TIME_THRESHOLD, 60 * 1000L));
        setHeapSizeAndSplitSize(args);

        // set default embedded zookeeper to client to avoid mapper oom: master mapper embeded zookeeper will use
        // 512M-1G memeory which may cause oom issue.
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, GuaguaConstants.GUAGUA_ZK_EMBEDBED_IS_IN_CLIENT,
                "true"));

        int vcores = vcoresSetting();
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, CommonConstants.MAPREDUCE_MAP_CPU_VCORES,
                vcores));

        // one can set guagua conf in shifuconfig
        CommonUtils.injectHadoopShifuEnvironments(new ValueVisitor() {
            @Override
            public void inject(Object key, Object value) {
                args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, key.toString(), value.toString()));
            }
        });
    }

    private void prepareMTLParams(List<String> args, SourceType sourceType) {
        args.add("-w");
        args.add(MTLWorker.class.getName());

        args.add("-m");
        args.add(MTLMaster.class.getName());

        args.add("-mr");
        args.add(MTLParams.class.getName());

        args.add("-wr");
        args.add(MTLParams.class.getName());

        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, GuaguaConstants.GUAGUA_MASTER_INTERCEPTERS,
                MTLOutput.class.getName()));
    }

    private void prepareWDLParams(List<String> args, SourceType sourceType) {
        args.add("-w");
        args.add(WDLWorker.class.getName());

        args.add("-m");
        args.add(WDLMaster.class.getName());

        args.add("-mr");
        args.add(WDLParams.class.getName());

        args.add("-wr");
        args.add(WDLParams.class.getName());

        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, GuaguaConstants.GUAGUA_MASTER_INTERCEPTERS,
                WDLOutput.class.getName()));
    }

    private int vcoresSetting() {
        // if set in shifuconfig with keyvalue, such value will be used, otherwise, thread number will be used
        String vcoreStr = Environment.getProperty(CommonConstants.MAPREDUCE_MAP_CPU_VCORES);
        int vcores = 1;
        if(vcoreStr == null) {
            vcores = modelConfig.getTrain().getWorkerThreadCount() == null ? 1
                    : modelConfig.getTrain().getWorkerThreadCount();
        } else {
            try {
                vcores = Integer.parseInt(vcoreStr);
            } catch (Exception e) {
                LOG.warn("Error in: {} not a number, will be set to 1", CommonConstants.MAPREDUCE_MAP_CPU_VCORES);
                vcores = 1;
            }
            if(vcores < 1 || vcores > 256) {
                LOG.warn("Error in: {} not in [1, 256], will be set to 6", CommonConstants.MAPREDUCE_MAP_CPU_VCORES);
                vcores = 6;
            }
        }
        return vcores;
    }

    private List<Integer> getSubsamplingFeatures(List<Integer> allFeatures, FeatureSubsetStrategy featureSubsetStrategy,
            double featureSubsetRate, int inputNum) {
        if(featureSubsetStrategy == null) {
            if(Double.compare(1d, featureSubsetRate) == 0) {
                return new ArrayList<Integer>();
            } else {
                return sampleFeaturesForNodeStats(allFeatures, (int) (allFeatures.size() * featureSubsetRate));
            }
        } else {
            switch(featureSubsetStrategy) {
                case HALF:
                    return sampleFeaturesForNodeStats(allFeatures, allFeatures.size() / 2);
                case ONETHIRD:
                    return sampleFeaturesForNodeStats(allFeatures, allFeatures.size() / 3);
                case TWOTHIRDS:
                    return sampleFeaturesForNodeStats(allFeatures, allFeatures.size() * 2 / 3);
                case SQRT:
                    return sampleFeaturesForNodeStats(allFeatures,
                            (int) (allFeatures.size() * Math.sqrt(inputNum) / inputNum));
                case LOG2:
                    return sampleFeaturesForNodeStats(allFeatures,
                            (int) (allFeatures.size() * Math.log(inputNum) / Math.log(2) / inputNum));
                case AUTO:
                case ALL:
                default:
                    return new ArrayList<Integer>();
            }
        }
    }

    private List<Integer> sampleFeaturesForNodeStats(List<Integer> allFeatures, int sample) {
        List<Integer> features = new ArrayList<Integer>(sample);
        for(int i = 0; i < sample; i++) {
            features.add(allFeatures.get(i));
        }

        for(int i = sample; i < allFeatures.size(); i++) {
            int replacementIndex = (int) (featureSamplingRandom.nextDouble() * i);
            if(replacementIndex >= 0 && replacementIndex < sample) {
                features.set(replacementIndex, allFeatures.get(i));
            }
        }
        return features;
    }

    private void setHeapSizeAndSplitSize(final List<String> args) throws IOException {
        // can be override by shifuconfig, ok for hard code
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, GuaguaMapReduceConstants.MAPRED_CHILD_JAVA_OPTS,
                "-Xms2048m -Xmx2048m -verbose:gc -XX:+PrintGCDetails -XX:+PrintGCTimeStamps"));
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, "mapreduce.map.java.opts",
                "-Xms2048m -Xmx2048m -server -XX:+UseParNewGC -XX:+UseConcMarkSweepGC "
                        + "-XX:CMSInitiatingOccupancyFraction=70 -verbose:gc -XX:+PrintGCDetails -XX:+PrintGCTimeStamps"));

        if(super.modelConfig.getNormalize().getIsParquet()) {
            args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, GuaguaConstants.GUAGUA_SPLIT_COMBINABLE,
                    Environment.getProperty(GuaguaConstants.GUAGUA_SPLIT_COMBINABLE, "false")));
        } else {
            args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, GuaguaConstants.GUAGUA_SPLIT_COMBINABLE,
                    Environment.getProperty(GuaguaConstants.GUAGUA_SPLIT_COMBINABLE, "true")));
            long maxCombineSize = computeDynamicCombineSize();
            LOG.info(
                    "Dynamic worker size is tuned to {}. If not good for # of workers, configure it in SHIFU_HOME/conf/shifuconfig::guagua.split.maxCombinedSplitSize",
                    maxCombineSize);
            args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT,
                    GuaguaConstants.GUAGUA_SPLIT_MAX_COMBINED_SPLIT_SIZE, Environment
                            .getProperty(GuaguaConstants.GUAGUA_SPLIT_MAX_COMBINED_SPLIT_SIZE, maxCombineSize + "")));
        }
        // special tuning parameters for shifu, 0.97 means each iteation master wait for 97% workers and then can go to
        // next iteration.
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, GuaguaConstants.GUAGUA_MIN_WORKERS_RATIO, 0.97));
        // 2 seconds if waiting over 10, consider 99% workers; these two can be overrided in shifuconfig
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, GuaguaConstants.GUAGUA_MIN_WORKERS_TIMEOUT,
                2 * 1000L));
    }

    private long computeDynamicCombineSize() throws IOException {
        // how many part-m-*.gz file in for gzip file, norm depends on how many gzip files
        String dataPath = null;
        if(CommonUtils.isTreeModel(super.modelConfig.getAlgorithm())) {
            Path filePath = new Path(super.getPathFinder().getCleanedDataPath());
            dataPath = ShifuFileUtils.getFileSystemBySourceType(modelConfig.getDataSet().getSource(), filePath)
                    .makeQualified(filePath).toString();
        } else {
            Path filePath = new Path(super.getPathFinder().getNormalizedDataPath());
            dataPath = ShifuFileUtils.getFileSystemBySourceType(modelConfig.getDataSet().getSource(), filePath)
                    .makeQualified(filePath).toString();
        }

        int filePartCnt = ShifuFileUtils.getFilePartCount(dataPath, SourceType.HDFS);
        long actualFileSize = ShifuFileUtils.getFileOrDirectorySize(dataPath, SourceType.HDFS);
        boolean isGzip = ShifuFileUtils.isPartFileAllGzip(dataPath, SourceType.HDFS);
        long avgFileSize = actualFileSize / filePartCnt;

        if(isGzip && filePartCnt <= 20) {
            // only 20 files, just set each one as a worker, 1.2 * avgFileSize is to make sure
            return (long) (avgFileSize * 1.2d);
            // otherwise, let dynamic combine size works
        }

        // in shifuconfig; by default it is 200M, consider in some cases user selects only a half of features,
        // this number should be 400m ?

        // int[] inputOutputIndex = DTrainUtils.getInputOutputCandidateCounts(modelConfig.getNormalizeType(),
        //        this.columnConfigList);
        //int candidateCount = (inputOutputIndex[2] == 0 ? inputOutputIndex[0] : inputOutputIndex[2]);
        int candidateCount = (modelConfig.isMultiTask() ?
                DTrainUtils.generateModelFeatureSet(modelConfig, this.mtlColumnConfigLists.get(0)).size()
                : DTrainUtils.generateModelFeatureSet(modelConfig, columnConfigList).size());

        // 1. set benchmark
        long maxCombineSize = CommonUtils.isTreeModel(modelConfig.getAlgorithm()) ? 209715200L : 168435456L;
        if(modelConfig.isClassification()) {
            return maxCombineSize;
        }
        // default 200M for gbt/RF, 150M for NN
        // why nn default is 150M, because of all categorical data is normalized to numeric, which is to save disk
        // for RF/gbt, categorical is still string and so default disk size is 200M
        // 2. according to ratio of ( candidate count / benchmark 600 features), tune combine size, 0.85 is a factor
        double ratio = candidateCount / 600d;
        if(ratio > 2d) {
            // 0.85 is a factor if selected ratio is 0.5 and only be effective if selected ratio over 2
            ratio = 0.85 * ratio;
        }

        long finalCombineSize = Double.valueOf((maxCombineSize * 1d * (ratio))).longValue();

        if(finalCombineSize != 0L && actualFileSize / finalCombineSize < 25) {
            // we can leverage more workers.
            finalCombineSize /= 2;
        }

        if((actualFileSize / finalCombineSize) > 1000L) {
            // auto tunning, no more than 1500 workers
            finalCombineSize = (actualFileSize / 1000L);
        }

        return finalCombineSize;
    }

    private void copyModelToLocal(String modelName, Path modelPath, SourceType sourceType) throws IOException {
        ShifuFileUtils.getFileSystemBySourceType(sourceType, modelPath).copyToLocalFile(modelPath,
                StringUtils.isBlank(modelName) ? new Path(super.getPathFinder().getModelsPath(SourceType.LOCAL))
                        : new Path(super.getPathFinder().getModelsPath(SourceType.LOCAL), modelName));
    }

    // GuaguaOptionsParser doesn't to support *.jar currently.
    private void addRuntimeJars(final List<String> args) {
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
        // common-lang3-*.jar
        jars.add(JarManager.findContainingJar(org.apache.commons.lang3.StringUtils.class));
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

        jars.add(JarManager.findContainingJar(Reflections.class));

        if(modelConfig.getNormalize().getIsParquet()) {
            // this jars are only for parquet format
            // parquet-mr-*.jar
            jars.add(JarManager.findContainingJar(ParquetRecordReader.class));
            // parquet-pig-*.jar
            jars.add(JarManager.findContainingJar(parquet.pig.ParquetLoader.class));
            // pig-*.jar
            jars.add(JarManager.findContainingJar(PigContext.class));
            // parquet-common-*.jar
            jars.add(JarManager.findContainingJar(ParquetRuntimeException.class));
            // parquet-column-*.jar
            jars.add(JarManager.findContainingJar(ParquetProperties.class));
            // parquet-encoding-*.jar
            jars.add(JarManager.findContainingJar(Packer.class));
            // parquet-generator-*.jar
            jars.add(JarManager.findContainingJar(Generator.class));
            // parquet-format-*.jar
            jars.add(JarManager.findContainingJar(PageType.class));
            // snappy-*.jar
            jars.add(JarManager.findContainingJar(Snappy.class));
            // parquet-jackson-*.jar
            jars.add(JarManager.findContainingJar(Base64Variant.class));
            // antlr jar
            jars.add(JarManager.findContainingJar(RecognitionException.class));
            // joda-time jar
            jars.add(JarManager.findContainingJar(ReadableInstant.class));
        }

        args.add(StringUtils.join(jars, NNConstants.LIB_JAR_SEPARATOR));
    }

    /**
     * For RF/GBT model, no need do normalizing, but clean and filter data is needed. Before real training, we have to
     * clean and filter data.
     *
     * @param isToShuffle
     *            if shuffle data before training
     * @throws IOException
     *             the io exception
     */
    protected void checkAndNormDataForModels(boolean isToShuffle) throws IOException {
        // check if binBoundaries and binCategories are good and log error
        for(ColumnConfig columnConfig: columnConfigList) {
            if(columnConfig.isFinalSelect() && !columnConfig.isTarget() && !columnConfig.isMeta()) {
                if(columnConfig.isNumerical() && columnConfig.getBinBoundary() == null) {
                    throw new IllegalArgumentException("Final select " + columnConfig.getColumnName()
                            + "column but binBoundary in ColumnConfig.json is null.");
                }
                if(columnConfig.isNumerical() && columnConfig.getBinBoundary().size() <= 1) {
                    LOG.warn(
                            "Column {} {} with only one or zero element in binBounday, such column will be ignored in tree model training.",
                            columnConfig.getColumnNum(), columnConfig.getColumnName());
                }
                if(columnConfig.isCategorical() && columnConfig.getBinCategory() == null) {
                    throw new IllegalArgumentException("Final select " + columnConfig.getColumnName()
                            + "column but binCategory in ColumnConfig.json is null.");
                }
                if(columnConfig.isCategorical() && columnConfig.getBinCategory().size() <= 0) {
                    LOG.warn(
                            "Column {} {} with only zero element in binCategory, such column will be ignored in tree model training.",
                            columnConfig.getColumnNum(), columnConfig.getColumnName());
                }
            }
        }

        // run cleaning data logic for model input
        SourceType sourceType = modelConfig.getDataSet().getSource();
        String needReGen = Environment.getProperty("shifu.tree.regeninput", Boolean.FALSE.toString());

        String alg = this.getModelConfig().getTrain().getAlgorithm();
        // only for tree models
        if(!CommonUtils.isTreeModel(alg)) {
            String normalDataPath = this.pathFinder.getNormalizedDataPath();
            if (Boolean.TRUE.toString().equalsIgnoreCase(needReGen)
                    || !ShifuFileUtils.isFileExists(normalDataPath, sourceType)
                    || (StringUtils.isNotBlank(modelConfig.getValidationDataSetRawPath())
                        && !ShifuFileUtils.isFileExists(pathFinder.getNormalizedValidationDataPath(), sourceType))) {
                LOG.info("The normalized data path {} doesn't exist. Generate it before training.", normalDataPath);
                Map<String, Object> params = new HashedMap();
                params.put(Constants.IS_TO_SHUFFLE_DATA, isToShuffle);
                NormalizeModelProcessor normStep = new NormalizeModelProcessor();
            }
        } else {
            String cleanedDataPath = this.pathFinder.getCleanedDataPath();

            // 1. shifu.tree.regeninput = true, no matter what, will regen;
            // 2. if cleanedDataPath does not exist, generate clean data for tree ensemble model training
            // 3. if validationDataPath is not blank and cleanedValidationDataPath does not exist, generate clean data for
            // tree ensemble model training
            if(Boolean.TRUE.toString().equalsIgnoreCase(needReGen)
                    || !ShifuFileUtils.isFileExists(cleanedDataPath, sourceType)
                    || (StringUtils.isNotBlank(modelConfig.getValidationDataSetRawPath())
                        && !ShifuFileUtils.isFileExists(pathFinder.getCleanedValidationDataPath(), sourceType))) {
                runDataClean(isToShuffle, -1.0, false); // -1.0 means no re-balance
            } else {
                // no need regen data
                LOG.warn("For RF/GBT, training input in {} exists, no need to regenerate it.", cleanedDataPath);
                LOG.warn("Need regen it, please set shifu.tree.regeninput in shifuconfig to true.");
            }
        }
    }

    /**
     * Get model name
     *
     * @param i
     *            index for model name
     * @return the ith model name
     */
    public String getModelName(int i) {
        String alg = super.getModelConfig().getTrain().getAlgorithm();
        return String.format("model%s.%s", i, alg.toLowerCase());
    }

    // d-train part ends here

    public void setToShuffle(boolean toShuffle) {
        isToShuffle = toShuffle;
    }

    /**
     * @return the isForVarSelect
     */
    public boolean isForVarSelect() {
        return isForVarSelect;
    }

    /**
     * @param isForVarSelect
     *            the isForVarSelect to set
     */
    public void setForVarSelect(boolean isForVarSelect) {
        this.isForVarSelect = isForVarSelect;
    }

    /**
     * set the train log file prefix
     *
     * @param trainLogFile
     *            - file to save train log
     */
    public void setTrainLogFile(String trainLogFile) {
        this.trainLogFile = trainLogFile;
    }

    /**
     * A thread used to tail progress log from hdfs log file.
     */
    public static class TailThread extends Thread {
        private long offset[];
        private String[] progressLogs;

        public TailThread(String[] progressLogs) {
            this.progressLogs = progressLogs;
            this.offset = new long[this.progressLogs.length];
            for(String progressLog: progressLogs) {
                try {
                    // delete it firstly, it will be updated from master
                    Path logPath = new Path(progressLog);
                    HDFSUtils.getFS(logPath).delete(new Path(progressLog), true);
                } catch (IOException e) {
                    LOG.error("Error in delete progressLog", e);
                }
            }
        }

        public void run() {
            while(!Thread.currentThread().isInterrupted()) {
                for(int i = 0; i < this.progressLogs.length; i++) {
                    try {
                        this.offset[i] = dumpFromOffset(new Path(this.progressLogs[i]), this.offset[i]);
                    } catch (FileNotFoundException e) {
                        // ignore because of not created in worker.
                    } catch (IOException e) {
                        LOG.warn(String.format("Error in dump progress log %s: %s", getName(), e.getMessage()));
                    } catch (Throwable e) {
                        LOG.warn(String.format("Error in thread %s: %s", getName(), e.getMessage()));
                    }
                }

                try {
                    Thread.sleep(2000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
            LOG.debug("DEBUG: Exit from tail thread.");
        }

        private long dumpFromOffset(Path item, long offset) throws IOException {
            if(!HDFSUtils.getFS(item).exists(item)) {
                // if file is not there, just return initial offset and wait for it is created
                return 0L;
            }

            FSDataInputStream in;
            try {
                in = HDFSUtils.getFS(item).open(item);
            } catch (Exception e) {
                // in hadoop 0.20.2, we found InteruptedException here and cannot be caught by run, here is to ignore
                // such exception. It's ok we return old offset to read message twice.
                return offset;
            }
            ByteArrayOutputStream out = null;
            DataOutputStream dataOut = null;
            try {
                out = new ByteArrayOutputStream();
                dataOut = new DataOutputStream(out);
                in.seek(offset);
                // use conf so the system configured io block size is used
                IOUtils.copyBytes(in, out, HDFSUtils.getFS(item).getConf(), false);
                String msgs = new String(out.toByteArray(), Charset.forName("UTF-8")).trim();
                if(StringUtils.isNotEmpty(msgs)) {
                    for(String msg: Splitter.on('\n').split(msgs)) {
                        LOG.info(msg.trim());
                    }
                }
                offset = in.getPos();
            } catch (IOException e) {
                if(e.getMessage().indexOf("Cannot seek after EOF") < 0) {
                    throw e;
                } else {
                    // LOG.warn(e.getMessage());
                }
            } finally {
                IOUtils.closeStream(in);
                IOUtils.closeStream(dataOut);
            }
            return offset;
        }

        public void deleteProgressFiles(String trainLogFile) throws IOException {
            if(StringUtils.isNotBlank(trainLogFile)) {
                BufferedWriter writer = null;
                try {
                    writer = new BufferedWriter(new FileWriter(trainLogFile));
                    for(String progressFile: this.progressLogs) {
                        Reader reader = ShifuFileUtils.getReader(progressFile, SourceType.HDFS);
                        org.apache.commons.io.IOUtils.copy(reader, writer);
                        org.apache.commons.io.IOUtils.closeQuietly(reader);
                    }
                } catch (IOException e) {
                    LOG.error("Fail to copy train log - {}", trainLogFile);
                } finally {
                    org.apache.commons.io.IOUtils.closeQuietly(writer);
                }
            }

            for(String progressFile: this.progressLogs) {
                Path filePath = new Path(progressFile);
                HDFSUtils.getFS(filePath).delete(filePath, true);
            }
        }

    }

}
