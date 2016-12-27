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
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.lang.Thread.UncaughtExceptionHandler;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Scanner;

import ml.shifu.guagua.GuaguaConstants;
import ml.shifu.guagua.hadoop.util.HDPUtils;
import ml.shifu.guagua.mapreduce.GuaguaMapReduceClient;
import ml.shifu.guagua.mapreduce.GuaguaMapReduceConstants;
import ml.shifu.shifu.actor.AkkaSystemExecutor;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelBasicConf.RunMode;
import ml.shifu.shifu.container.obj.ModelTrainConf.MultipleClassification;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.AbstractTrainer;
import ml.shifu.shifu.core.TreeModel;
import ml.shifu.shifu.core.alg.LogisticRegressionTrainer;
import ml.shifu.shifu.core.alg.NNTrainer;
import ml.shifu.shifu.core.alg.SVMTrainer;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.dtrain.dt.DTMaster;
import ml.shifu.shifu.core.dtrain.dt.DTMasterParams;
import ml.shifu.shifu.core.dtrain.dt.DTOutput;
import ml.shifu.shifu.core.dtrain.dt.DTWorker;
import ml.shifu.shifu.core.dtrain.dt.DTWorkerParams;
import ml.shifu.shifu.core.dtrain.gs.GridSearch;
import ml.shifu.shifu.core.dtrain.lr.LogisticRegressionContants;
import ml.shifu.shifu.core.dtrain.lr.LogisticRegressionMaster;
import ml.shifu.shifu.core.dtrain.lr.LogisticRegressionOutput;
import ml.shifu.shifu.core.dtrain.lr.LogisticRegressionParams;
import ml.shifu.shifu.core.dtrain.lr.LogisticRegressionWorker;
import ml.shifu.shifu.core.dtrain.nn.NNConstants;
import ml.shifu.shifu.core.dtrain.nn.NNMaster;
import ml.shifu.shifu.core.dtrain.nn.NNOutput;
import ml.shifu.shifu.core.dtrain.nn.NNParams;
import ml.shifu.shifu.core.dtrain.nn.NNParquetWorker;
import ml.shifu.shifu.core.dtrain.nn.NNWorker;
import ml.shifu.shifu.core.validator.ModelInspector.ModelStep;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.guagua.GuaguaParquetMapReduceClient;
import ml.shifu.shifu.guagua.ShifuInputFormat;
import ml.shifu.shifu.pig.PigExecutor;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.Environment;
import ml.shifu.shifu.util.HDFSUtils;
import ml.shifu.shifu.util.JSONUtils;

import org.antlr.runtime.RecognitionException;
import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.collections.ListUtils;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.pig.LoadPushDown.RequiredField;
import org.apache.pig.LoadPushDown.RequiredFieldList;
import org.apache.pig.data.DataType;
import org.apache.pig.impl.PigContext;
import org.apache.pig.impl.util.JarManager;
import org.apache.pig.impl.util.ObjectSerializer;
import org.apache.zookeeper.ZooKeeper;
import org.encog.ml.MLInputOutput;
import org.encog.ml.data.MLDataSet;
import org.jboss.netty.bootstrap.ServerBootstrap;
import org.joda.time.ReadableInstant;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.xerial.snappy.Snappy;

import parquet.ParquetRuntimeException;
import parquet.column.ParquetProperties;
import parquet.column.values.bitpacking.Packer;
import parquet.encoding.Generator;
import parquet.format.PageType;
import parquet.hadoop.ParquetRecordReader;
import parquet.org.codehaus.jackson.Base64Variant;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.base.Splitter;

/**
 * Train processor, produce model based on the normalized dataset
 */
public class TrainModelProcessor extends BasicModelProcessor implements Processor {

    private final static Logger LOG = LoggerFactory.getLogger(TrainModelProcessor.class);

    private static final int VAR_SELECT_TRAINING_DECAY_EPOCHES_THRESHOLD = 200;

    public static final String SHIFU_DEFAULT_DTRAIN_PARALLEL = "true";

    private boolean isDryTrain, isDebug;
    private List<AbstractTrainer> trainers;

    private static final String LOGS = "./logs";

    /**
     * If for variable selection, only using bagging number 1 to train only one model.
     */
    private boolean isForVarSelect;

    public TrainModelProcessor() {
    }

    public TrainModelProcessor(Map<String, Object> otherConfigs) {
        super.otherConfigs = otherConfigs;
    }

    /**
     * Constructor
     * 
     * @param isDryTrain
     *            dryTrain flag, if it's true, the trainer would start training
     * @param isDebug
     *            debug flag, if it's true, shifu will create log file to record
     *            each training status
     */
    public TrainModelProcessor(boolean isDryTrain, boolean isDebug) {
        super();

        this.isDebug = isDebug;
        this.isDryTrain = isDryTrain;

        trainers = new ArrayList<AbstractTrainer>();
    }

    /**
     * run training process
     */
    @Override
    public int run() throws Exception {
        if(!this.isForVarSelect()) {
            LOG.info("Step Start: train");
        }
        long start = System.currentTimeMillis();
        try {
            setUp(ModelStep.TRAIN);

            if(isDebug) {
                File file = new File(LOGS);
                if(!file.exists() && !file.mkdir()) {
                    throw new RuntimeException("logs file is created failed.");
                }
            }

            RunMode runMode = super.modelConfig.getBasic().getRunMode();
            switch(runMode) {
                case DIST:
                case MAPRED:
                    validateDistributedTrain();
                    syncDataToHdfs(super.modelConfig.getDataSet().getSource());
                    checkAndCleanDataForTreeModels();
                    runDistributedTrain();
                    break;
                case LOCAL:
                default:
                    runAkkaTrain(isForVarSelect ? 1 : modelConfig.getBaggingNum());
                    break;
            }

            syncDataToHdfs(modelConfig.getDataSet().getSource());

            clearUp(ModelStep.TRAIN);
        } catch (Exception e) {
            LOG.error("Error:", e);
            return -1;
        }
        if(!this.isForVarSelect()) {
            LOG.info("Step Finished: train with {} ms", (System.currentTimeMillis() - start));
        }
        return 0;
    }

    /**
     * For RF/GBT model, no need do normalizing, but clean and filter data is needed. Before real training, we have to
     * clean and filter data.
     */
    private void checkAndCleanDataForTreeModels() throws IOException {
        String alg = super.getModelConfig().getTrain().getAlgorithm();
        // only for tree models
        if(!CommonUtils.isDesicionTreeAlgorithm(alg)) {
            return;
        }

        SourceType sourceType = modelConfig.getDataSet().getSource();
        String cleanedDataPath = super.pathFinder.getCleanedDataPath();
        String needReGen = Environment.getProperty("shifu.tree.regeninput", Boolean.FALSE.toString());
        if(ShifuFileUtils.isFileExists(cleanedDataPath, sourceType)
                && Boolean.FALSE.toString().equalsIgnoreCase(needReGen)) {
            LOG.warn("For RF/GBT, training input in {} exists, no need to regenerate it.", cleanedDataPath);
            LOG.warn("Need regen it, please set shifu.tree.regeninput in shifuconfig to true.");
        } else {
            LOG.info("Start to generate clean data for tree molde ... ");
            if(ShifuFileUtils.isFileExists(cleanedDataPath, sourceType)) {
                ShifuFileUtils.deleteFile(cleanedDataPath, sourceType);
            }
            String cleandedValidationDataPath = pathFinder.getCleanedValidationDataPath();
            if(ShifuFileUtils.isFileExists(cleandedValidationDataPath, sourceType)) {
                ShifuFileUtils.deleteFile(cleandedValidationDataPath, sourceType);
            }

            Map<String, String> paramsMap = new HashMap<String, String>();
            paramsMap.put("sampleRate", modelConfig.getNormalizeSampleRate().toString());
            paramsMap.put("sampleNegOnly", ((Boolean) modelConfig.isNormalizeSampleNegOnly()).toString());
            paramsMap.put("delimiter", CommonUtils.escapePigString(modelConfig.getDataSetDelimiter()));

            try {
                String normPigPath = pathFinder.getAbsolutePath("scripts/Normalize.pig");

                paramsMap.put(Constants.IS_COMPRESS, "true");
                paramsMap.put(Constants.IS_NORM_FOR_CLEAN, "true");
                paramsMap.put(Constants.PATH_NORMALIZED_DATA, pathFinder.getCleanedDataPath());
                PigExecutor.getExecutor().submitJob(modelConfig, normPigPath, paramsMap, sourceType, super.pathFinder);
                if(StringUtils.isNotBlank(modelConfig.getValidationDataSetRawPath())) {
                    paramsMap.put(Constants.IS_COMPRESS, "false");
                    paramsMap.put(Constants.PATH_RAW_DATA, modelConfig.getValidationDataSetRawPath());
                    paramsMap.put(Constants.PATH_NORMALIZED_DATA, pathFinder.getCleanedValidationDataPath());
                    PigExecutor.getExecutor().submitJob(modelConfig, normPigPath, paramsMap, sourceType,
                            super.pathFinder);
                }
            } catch (IOException e) {
                throw new ShifuException(ShifuErrorCode.ERROR_RUNNING_PIG_JOB, e);
            } catch (Throwable e) {
                throw new RuntimeException(e);
            }
            LOG.info("Generate clean data for tree molde successful.");
        }
    }

    /**
     * run training process with number of bags
     * 
     * @param numBags
     *            number of bags, it decide how much trainer will start training
     * @throws IOException
     */
    private void runAkkaTrain(int numBags) throws IOException {
        File models = new File("models");
        FileUtils.deleteDirectory(models);
        FileUtils.forceMkdir(models);

        trainers.clear();

        for(int i = 0; i < numBags; i++) {
            AbstractTrainer trainer;
            if(modelConfig.getAlgorithm().equalsIgnoreCase("NN")) {
                trainer = new NNTrainer(modelConfig, i, isDryTrain);
            } else if(modelConfig.getAlgorithm().equalsIgnoreCase("SVM")) {
                trainer = new SVMTrainer(this.modelConfig, i, isDryTrain);
            } else if(modelConfig.getAlgorithm().equalsIgnoreCase("LR")) {
                trainer = new LogisticRegressionTrainer(this.modelConfig, i, isDryTrain);
            } else {
                throw new ShifuException(ShifuErrorCode.ERROR_UNSUPPORT_ALG);
            }

            trainers.add(trainer);
        }

        List<Scanner> scanners = null;
        if(modelConfig.getAlgorithm().equalsIgnoreCase("DT")) {
            LOG.info("Raw Data: " + pathFinder.getNormalizedDataPath());
            try {
                scanners = ShifuFileUtils.getDataScanners(modelConfig.getDataSetRawPath(), modelConfig.getDataSet()
                        .getSource());
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
                scanners = ShifuFileUtils.getDataScanners(pathFinder.getNormalizedDataPath(), modelConfig.getDataSet()
                        .getSource());
            } catch (IOException e) {
                throw new ShifuException(ShifuErrorCode.ERROR_INPUT_NOT_FOUND, e, pathFinder.getNormalizedDataPath());
            }
            if(CollectionUtils.isNotEmpty(scanners)) {
                AkkaSystemExecutor.getExecutor().submitModelTrainJob(modelConfig, columnConfigList, scanners, trainers);
            }
        }

        // release
        closeScanners(scanners);
    }

    /**
     * get the trainer list
     * 
     * @return the trainer list
     */
    public List<AbstractTrainer> getTrainers() {
        return trainers;
    }

    /**
     * get the trainer
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

    private void validateDistributedTrain() throws IOException {
        String alg = super.getModelConfig().getTrain().getAlgorithm();
        if(!(NNConstants.NN_ALG_NAME.equalsIgnoreCase(alg) // NN algorithm
                || LogisticRegressionContants.LR_ALG_NAME.equalsIgnoreCase(alg) // LR algorithm
        || CommonUtils.isDesicionTreeAlgorithm(alg))) { // RF or GBT algortihm
            throw new IllegalArgumentException(
                    "Currently we only support NN, LR, RF(RandomForest) and GBDT(Gradient Boost Desicion Tree) distributed training.");
        }

        if((LogisticRegressionContants.LR_ALG_NAME.equalsIgnoreCase(alg) || CommonConstants.GBT_ALG_NAME
                .equalsIgnoreCase(alg))
                && modelConfig.isClassification()
                && modelConfig.getTrain().getMultiClassifyMethod() == MultipleClassification.NATIVE) {
            throw new IllegalArgumentException(
                    "Distributed LR, GBDT(Gradient Boost Desicion Tree) only support binary classification, native multiple classification is not supported.");
        }

        if(modelConfig.isClassification() && modelConfig.getTrain().isOneVsAll()
                && !CommonUtils.isDesicionTreeAlgorithm(alg) && !NNConstants.NN_ALG_NAME.equalsIgnoreCase(alg)) {
            throw new IllegalArgumentException("Only GBT and RF and NN support OneVsAll multiple classification.");
        }

        if(super.getModelConfig().getDataSet().getSource() != SourceType.HDFS) {
            throw new IllegalArgumentException("Currently we only support distributed training on HDFS source type.");
        }

        if(isDebug()) {
            LOG.warn("Currently we haven't debug logic. It's the same as you don't set it.");
        }

        // check if parquet format norm output is consistent with current isParquet setting.
        boolean isParquetMetaFileExist = ShifuFileUtils.getFileSystemBySourceType(
                super.getModelConfig().getDataSet().getSource()).exists(
                new Path(super.getPathFinder().getNormalizedDataPath(), "_common_metadata"));
        if(super.modelConfig.getNormalize().getIsParquet() && !isParquetMetaFileExist) {
            throw new IllegalArgumentException(
                    "Your normlized input in "
                            + super.getPathFinder().getNormalizedDataPath()
                            + " is not parquet format. Please keep isParquet and re-run norm again and then run training step or change isParquet to false.");
        } else if(!super.modelConfig.getNormalize().getIsParquet() && isParquetMetaFileExist) {
            throw new IllegalArgumentException(
                    "Your normlized input in "
                            + super.getPathFinder().getNormalizedDataPath()
                            + " is parquet format. Please keep isParquet and re-run norm again or change isParquet directly to true.");
        }

        GridSearch gridSearch = new GridSearch(modelConfig.getTrain().getParams());
        if(!NNConstants.NN_ALG_NAME.equalsIgnoreCase(alg) && !CommonUtils.isDesicionTreeAlgorithm(alg)
                && gridSearch.hasHyperParam()) {
            // if grid search but not NN, not RF, not GBT
            throw new IllegalArgumentException("Grid search only supports NN, GBT and RF algorithms");
        }

        if(gridSearch.hasHyperParam() && super.getModelConfig().getDataSet().getSource() != SourceType.HDFS
                && modelConfig.isDistributedRunMode()) {
            // if grid search but not mapred/dist run mode, not hdfs raw data set
            throw new IllegalArgumentException("Grid search only supports NN, GBT and RF algorithms");
        }
    }

    protected void runDistributedTrain() throws IOException, InterruptedException, ClassNotFoundException {
        LOG.info("Started {} d-training.", isDryTrain ? "dry" : "");

        Configuration conf = new Configuration();

        SourceType sourceType = super.getModelConfig().getDataSet().getSource();

        final List<String> args = new ArrayList<String>();

        GridSearch gs = new GridSearch(modelConfig.getTrain().getParams());

        prepareCommonParams(gs.hasHyperParam(), args, sourceType);

        String alg = super.getModelConfig().getTrain().getAlgorithm();

        // add tmp models folder to config
        FileSystem fileSystem = ShifuFileUtils.getFileSystemBySourceType(sourceType);
        Path tmpModelsPath = fileSystem.makeQualified(new Path(super.getPathFinder().getPathBySourceType(
                new Path(Constants.TMP, Constants.DEFAULT_MODELS_TMP_FOLDER), sourceType)));
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, CommonConstants.SHIFU_TMP_MODELS_FOLDER,
                tmpModelsPath.toString()));
        int baggingNum = isForVarSelect ? 1 : super.getModelConfig().getBaggingNum();
        if(modelConfig.isClassification() && modelConfig.getTrain().isOneVsAll()) {
            // one vs all multiple classification, we need multiple bagging jobs to do ONEVSALL
            baggingNum = modelConfig.getTags().size();
            if(baggingNum != super.getModelConfig().getBaggingNum()) {
                LOG.warn("'train:baggingNum' is set to {} because of ONEVSALL multiple classification.", baggingNum);
            }
        }

        long start = System.currentTimeMillis();
        LOG.info("Distributed trainning with baggingNum: {}", baggingNum);
        boolean isParallel = Boolean.valueOf(
                Environment.getProperty(Constants.SHIFU_DTRAIN_PARALLEL, SHIFU_DEFAULT_DTRAIN_PARALLEL)).booleanValue();
        GuaguaMapReduceClient guaguaClient;
        if(modelConfig.getNormalize().getIsParquet()) {
            guaguaClient = new GuaguaParquetMapReduceClient();

            // set required field list to make sure we only load selected columns.
            RequiredFieldList requiredFieldList = new RequiredFieldList();

            int[] inputOutputIndex = DTrainUtils.getInputOutputCandidateCounts(this.columnConfigList);
            int inputNodeCount = inputOutputIndex[0] == 0 ? inputOutputIndex[2] : inputOutputIndex[0];
            int candidateCount = inputOutputIndex[2];

            for(ColumnConfig columnConfig: super.columnConfigList) {
                if(columnConfig.isTarget()) {
                    requiredFieldList.add(new RequiredField(columnConfig.getColumnName(), columnConfig.getColumnNum(),
                            null, DataType.FLOAT));
                } else {
                    if(inputNodeCount == candidateCount) {
                        // no any variables are selected
                        if(!columnConfig.isMeta() && !columnConfig.isTarget()
                                && CommonUtils.isGoodCandidate(columnConfig)) {
                            requiredFieldList.add(new RequiredField(columnConfig.getColumnName(), columnConfig
                                    .getColumnNum(), null, DataType.FLOAT));
                        }
                    } else {
                        if(!columnConfig.isMeta() && !columnConfig.isTarget() && columnConfig.isFinalSelect()) {
                            requiredFieldList.add(new RequiredField(columnConfig.getColumnName(), columnConfig
                                    .getColumnNum(), null, DataType.FLOAT));
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
        } else {
            guaguaClient = new GuaguaMapReduceClient();
        }

        int parallelNum = Integer
                .parseInt(Environment.getProperty(CommonConstants.SHIFU_TRAIN_BAGGING_INPARALLEL, "5"));
        int parallelGroups = 1;
        if(gs.hasHyperParam()) {
            parallelGroups = (gs.getFlattenParams().size() % parallelNum == 0 ? gs.getFlattenParams().size()
                    / parallelNum : gs.getFlattenParams().size() / parallelNum + 1);
        } else {
            parallelGroups = baggingNum % parallelNum == 0 ? baggingNum / parallelNum : baggingNum / parallelNum + 1;
        }
        List<String> progressLogList = new ArrayList<String>(baggingNum);
        boolean isOneJobNotContinuous = false;
        for(int j = 0; j < parallelGroups; j++) {
            int currBags = baggingNum;
            if(gs.hasHyperParam()) {
                if(j == parallelGroups - 1) {
                    currBags = gs.getFlattenParams().size() % parallelNum == 0 ? parallelNum : gs.getFlattenParams()
                            .size() % parallelNum;
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
            for(int k = 0; k < currBags; k++) {
                int i = j * parallelNum + k;
                if(gs.hasHyperParam()) {
                    LOG.info("Start the {}th grid search job with params: {}", i, gs.getParams(i));
                }
                List<String> localArgs = new ArrayList<String>(args);
                // set name for each bagging job.
                localArgs.add("-n");
                localArgs.add(String.format("Shifu Master-Workers %s Training Iteration: %s id:%s", alg, super
                        .getModelConfig().getModelSetName(), i));
                LOG.info("Start trainer with id: {}", i);
                String modelName = getModelName(i);
                Path modelPath = fileSystem.makeQualified(new Path(super.getPathFinder().getModelsPath(sourceType),
                        modelName));
                // check if job is continunous training, this can be set multiple times and we only get last one
                boolean isContinous = checkContinuousTraining(fileSystem, localArgs, modelPath);
                if(isContinous && CommonConstants.RF_ALG_NAME.equalsIgnoreCase(modelConfig.getAlgorithm())) {
                    isContinous = false;
                    LOG.warn("RF does not support continuous training");
                }
                // of course gs not support continuous model training
                if(gs.hasHyperParam()) {
                    isContinous = false;
                }
                if(!isContinous && !isOneJobNotContinuous) {
                    isOneJobNotContinuous = true;
                    // delete all old models if not continous
                    String srcModelPath = super.getPathFinder().getModelsPath(sourceType);
                    String mvModelPath = srcModelPath + "_" + System.currentTimeMillis();
                    LOG.info("Old model path has been moved to {}", mvModelPath);
                    fileSystem.rename(new Path(srcModelPath), new Path(mvModelPath));
                    fileSystem.mkdirs(new Path(srcModelPath));
                    FileSystem.getLocal(conf).delete(new Path(super.getPathFinder().getModelsPath(SourceType.LOCAL)),
                            true);
                }
                localArgs.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, CommonConstants.GUAGUA_OUTPUT,
                        modelPath.toString()));
                if(gs.hasHyperParam()) {
                    Path valErrPath = fileSystem.makeQualified(new Path(super.getPathFinder().getValErrorPath(
                            sourceType), "val_error_" + i));
                    localArgs.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT,
                            CommonConstants.GS_VALIDATION_ERROR, valErrPath.toString()));
                }
                localArgs.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, CommonConstants.SHIFU_TRAINER_ID,
                        String.valueOf(i)));
                final String progressLogFile = getProgressLogFile(i);
                progressLogList.add(progressLogFile);
                localArgs.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT,
                        CommonConstants.SHIFU_DTRAIN_PROGRESS_FILE, progressLogFile));
                String hdpVersion = HDPUtils.getHdpVersionForHDP224();
                if(StringUtils.isNotBlank(hdpVersion)) {
                    localArgs.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, "hdp.version", hdpVersion));
                    HDPUtils.addFileToClassPath(HDPUtils.findContainingFile("hdfs-site.xml"), conf);
                    HDPUtils.addFileToClassPath(HDPUtils.findContainingFile("core-site.xml"), conf);
                    HDPUtils.addFileToClassPath(HDPUtils.findContainingFile("mapred-site.xml"), conf);
                    HDPUtils.addFileToClassPath(HDPUtils.findContainingFile("yarn-site.xml"), conf);
                }

                if(isParallel) {
                    guaguaClient.addJob(localArgs.toArray(new String[0]));
                } else {
                    TailThread tailThread = startTailThread(new String[] { progressLogFile });
                    guaguaClient.createJob(localArgs.toArray(new String[0])).waitForCompletion(true);
                    stopTailThread(tailThread);
                }
            }

            if(isParallel) {
                TailThread tailThread = startTailThread(progressLogList.toArray(new String[0]));
                guaguaClient.run();
                stopTailThread(tailThread);
            }
        }

        // copy all models to local after all jobs are finished
        if(!gs.hasHyperParam()) {
            // copy model files at last.
            for(int i = 0; i < baggingNum; i++) {
                String modelName = getModelName(i);
                Path modelPath = fileSystem.makeQualified(new Path(super.getPathFinder().getModelsPath(sourceType),
                        modelName));
                if(ShifuFileUtils.getFileSystemBySourceType(sourceType).exists(modelPath)) {
                    copyModelToLocal(modelName, modelPath, sourceType);
                } else {
                    LOG.warn("Model {} isn't there, maybe job is failed, for bagging it can be ignored.",
                            modelPath.toString());
                }
            }

            // copy temp model files, for RF/GBT, not to copy tmp models because of larger space needed, for others by
            // default copy tmp models to local
            boolean copyTmpModelsToLocal = Boolean.TRUE.toString().equalsIgnoreCase(
                    Environment.getProperty(Constants.SHIFU_TMPMODEL_COPYTOLOCAL, "true"));
            if(CommonUtils.isDesicionTreeAlgorithm(modelConfig.getAlgorithm())) {
                copyTmpModelsToLocal = Boolean.TRUE.toString().equalsIgnoreCase(
                        Environment.getProperty(Constants.SHIFU_TMPMODEL_COPYTOLOCAL, "false"));
            }

            if(copyTmpModelsToLocal) {
                copyTmpModelsToLocal(tmpModelsPath, sourceType);
            } else {
                LOG.info("Tmp models are not coied into local, please find them in hdfs path: {}", tmpModelsPath);
            }
            LOG.info("Distributed training finished in {}ms.", System.currentTimeMillis() - start);
        }

        if(gs.hasHyperParam()) {
            // select the best parameter composite in grid search
            LOG.info("Original grid search params: {}", modelConfig.getParams());
            Map<String, Object> params = findBestParams(sourceType, fileSystem, gs);
            // TODO, copy top 5 models for evaluation? (no need further train)
            for(Entry<String, Object> entry: params.entrySet()) {
                modelConfig.getParams().put(entry.getKey(), entry.getValue());
            }
            super.pathFinder.getModelConfigPath(SourceType.LOCAL);
            // update ModelConfig.json
            JSONUtils.writeValue(new File(super.pathFinder.getModelConfigPath(SourceType.LOCAL)), modelConfig);
            LOG.info("Grid search on distributed training finished in {}ms.", System.currentTimeMillis() - start);
        }
    }

    private Map<String, Object> findBestParams(SourceType sourceType, FileSystem fileSystem, GridSearch gs)
            throws IOException {
        // read validationerror and find the best one update ModelConfig.
        double minValErr = Double.MAX_VALUE;
        int minIndex = -1;
        for(int i = 0; i < gs.getFlattenParams().size(); i++) {
            Path valErrPath = fileSystem.makeQualified(new Path(super.getPathFinder().getValErrorPath(sourceType),
                    "val_error_" + i));
            if(ShifuFileUtils.isFileExists(valErrPath.toString(), sourceType)) {
                double valErr;
                BufferedReader reader = null;
                try {
                    reader = ShifuFileUtils.getReader(valErrPath.toString(), sourceType);
                    String valErrStr = reader.readLine().toString();
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

    static String serializeRequiredFieldList(RequiredFieldList requiredFieldList) {
        try {
            return ObjectSerializer.serialize(requiredFieldList);
        } catch (IOException e) {
            throw new RuntimeException("Failed to searlize required fields.", e);
        }
    }

    private boolean checkContinuousTraining(FileSystem fileSystem, List<String> localArgs, Path modelPath)
            throws IOException {
        boolean finalContinuous = false;
        if(Boolean.TRUE.toString().equals(this.modelConfig.getTrain().getIsContinuous().toString())) {
            // if varselect d-training or no such existing models, directly to disable continuous training.
            if(this.isForVarSelect) {
                finalContinuous = false;
                LOG.warn("For varSelect step, continous model training is always disabled.");
            } else if(!fileSystem.exists(modelPath)) {
                finalContinuous = false;
                LOG.info("No existing model, model training will start from scratch.");
            } else if(NNConstants.NN_ALG_NAME.equalsIgnoreCase(modelConfig.getAlgorithm())
                    && !inputOutputModelCheckSuccess(fileSystem, modelPath)) {
                // TODO hidden layer size and activation functions should also be validated
                finalContinuous = false;
                LOG.warn("Model input and output settings are not consistent with input and output columns settings, "
                        + "model training will start from scratch.");
            } else if(CommonConstants.GBT_ALG_NAME.equalsIgnoreCase(modelConfig.getAlgorithm())) {
                TreeModel model = (TreeModel) CommonUtils.loadModel(this.modelConfig, this.columnConfigList, modelPath,
                        fileSystem);
                if(!model.getAlgorithm().equalsIgnoreCase(modelConfig.getAlgorithm())) {
                    finalContinuous = false;
                    LOG.warn("Only GBT supports continuous training, while not GBT, will start from scratch");
                } else if(!model.getLossStr().equalsIgnoreCase(
                        this.modelConfig.getTrain().getParams().get("Loss").toString())) {
                    finalContinuous = false;
                    LOG.warn("Loss is changed, continuous training is disabled, will start from scratch");
                } else {
                    finalContinuous = true;
                }
            } else if(CommonConstants.RF_ALG_NAME.equalsIgnoreCase(modelConfig.getAlgorithm())) {
                finalContinuous = false;
                LOG.warn("RF doesn't support continuous training");
            } else {
                finalContinuous = true;
            }
        } else {
            finalContinuous = false;
        }
        localArgs.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, CommonConstants.CONTINUOUS_TRAINING,
                finalContinuous));
        return finalContinuous;
    }

    private boolean inputOutputModelCheckSuccess(FileSystem fileSystem, Path modelPath) throws IOException {
        MLInputOutput model = (MLInputOutput) CommonUtils.loadModel(this.modelConfig, this.columnConfigList, modelPath,
                fileSystem);
        int[] outputCandidateCounts = DTrainUtils.getInputOutputCandidateCounts(getColumnConfigList());
        return model.getInputCount() == outputCandidateCounts[0] && model.getOutputCount() == outputCandidateCounts[1];
    }

    private String getProgressLogFile(int i) {
        return String.format("tmp/%s_%s.log", System.currentTimeMillis(), i);
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
        thread.deleteProgressFiles();
    }

    private TailThread startTailThread(final String[] progressLog) {
        TailThread thread = new TailThread(progressLog);
        thread.setName("Tail Progress Thread");
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
        if(!this.isDryTrain()) {
            if(ShifuFileUtils.getFileSystemBySourceType(sourceType).exists(tmpModelsDir)) {
                Path localTmpModelsFolder = new Path(Constants.TMP);
                HDFSUtils.getLocalFS().delete(localTmpModelsFolder, true);
                HDFSUtils.getLocalFS().mkdirs(localTmpModelsFolder);
                ShifuFileUtils.getFileSystemBySourceType(sourceType)
                        .copyToLocalFile(tmpModelsDir, localTmpModelsFolder);
            }
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

    private void prepareCommonParams(boolean isGsMode, final List<String> args, final SourceType sourceType) {
        String alg = super.getModelConfig().getTrain().getAlgorithm();

        args.add("-libjars");
        addRuntimeJars(args);

        args.add("-i");
        if(CommonUtils.isDesicionTreeAlgorithm(alg)) {
            args.add(ShifuFileUtils.getFileSystemBySourceType(sourceType)
                    .makeQualified(new Path(super.getPathFinder().getCleanedDataPath())).toString());
        } else {
            args.add(ShifuFileUtils.getFileSystemBySourceType(sourceType)
                    .makeQualified(new Path(super.getPathFinder().getNormalizedDataPath())).toString());
        }

        if(StringUtils.isNotBlank(modelConfig.getValidationDataSetRawPath())) {
            args.add("-inputformat");
            args.add(ShifuInputFormat.class.getName());
        }

        String zkServers = Environment.getProperty(Environment.ZOO_KEEPER_SERVERS);
        if(StringUtils.isEmpty(zkServers)) {
            LOG.warn("No specified zookeeper settings from zookeeperServers in shifuConfig file, Guagua will set embeded zookeeper server in client process or master node. For fail-over zookeeper applications, specified zookeeper servers are strongly recommended.");
        } else {
            args.add("-z");
            args.add(zkServers);
        }

        if(LogisticRegressionContants.LR_ALG_NAME.equalsIgnoreCase(alg)) {
            this.prepareLRParams(args, sourceType);
        } else if(NNConstants.NN_ALG_NAME.equalsIgnoreCase(alg)) {
            this.prepareNNParams(args, sourceType);
        } else if(CommonUtils.isDesicionTreeAlgorithm(alg)) {
            this.prepareDTParams(args, sourceType);
        }

        args.add("-c");
        int numTrainEpoches = super.getModelConfig().getTrain().getNumTrainEpochs();
        // only for NN varselect, use half of epochs for sensitivity analysis
        // if for gs mode, half of iterations are used
        if(NNConstants.NN_ALG_NAME.equalsIgnoreCase(alg) && (this.isForVarSelect() || isGsMode)
                && numTrainEpoches >= VAR_SELECT_TRAINING_DECAY_EPOCHES_THRESHOLD) {
            numTrainEpoches = numTrainEpoches / 2;
        }
        // the reason to add 1 is that the first iteration in implementation is used for training preparation.
        numTrainEpoches = numTrainEpoches + 1;

        args.add(String.valueOf(numTrainEpoches));

        if(CommonUtils.isDesicionTreeAlgorithm(alg)) {
            // for tree models, using cleaned validation data path
            args.add(String.format(
                    CommonConstants.MAPREDUCE_PARAM_FORMAT,
                    CommonConstants.CROSS_VALIDATION_DIR,
                    ShifuFileUtils.getFileSystemBySourceType(sourceType)
                            .makeQualified(new Path(super.getPathFinder().getCleanedValidationDataPath(sourceType)))
                            .toString()));
        } else {
            args.add(String.format(
                    CommonConstants.MAPREDUCE_PARAM_FORMAT,
                    CommonConstants.CROSS_VALIDATION_DIR,
                    ShifuFileUtils.getFileSystemBySourceType(sourceType)
                            .makeQualified(new Path(super.getPathFinder().getNormalizedValidationDataPath(sourceType)))
                            .toString()));
        }
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, NNConstants.MAPRED_JOB_QUEUE_NAME,
                Environment.getProperty(Environment.HADOOP_JOB_QUEUE, Constants.DEFAULT_JOB_QUEUE)));
        args.add(String.format(
                CommonConstants.MAPREDUCE_PARAM_FORMAT,
                CommonConstants.SHIFU_MODEL_CONFIG,
                ShifuFileUtils.getFileSystemBySourceType(sourceType).makeQualified(
                        new Path(super.getPathFinder().getModelConfigPath(sourceType)))));
        args.add(String.format(
                CommonConstants.MAPREDUCE_PARAM_FORMAT,
                CommonConstants.SHIFU_COLUMN_CONFIG,
                ShifuFileUtils.getFileSystemBySourceType(sourceType).makeQualified(
                        new Path(super.getPathFinder().getColumnConfigPath(sourceType)))));
        args.add(String
                .format(CommonConstants.MAPREDUCE_PARAM_FORMAT, CommonConstants.MODELSET_SOURCE_TYPE, sourceType));
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, CommonConstants.SHIFU_DRY_DTRAIN, isDryTrain()));
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, NNConstants.NN_POISON_SAMPLER,
                Environment.getProperty(NNConstants.NN_POISON_SAMPLER, "true")));
        // hard code set computation threshold for 50s. Can be changed in shifuconfig file
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT,
                GuaguaConstants.GUAGUA_COMPUTATION_TIME_THRESHOLD, 60 * 1000L));
        setHeapSizeAndSplitSize(args);

        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, "mapreduce.map.cpu.vcores", modelConfig
                .getTrain().getWorkerThreadCount() == null ? 1 : modelConfig.getTrain().getWorkerThreadCount()));

        // one can set guagua conf in shifuconfig
        for(Map.Entry<Object, Object> entry: Environment.getProperties().entrySet()) {
            if(CommonUtils.isHadoopConfigurationInjected(entry.getKey().toString())) {
                args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, entry.getKey().toString(), entry
                        .getValue().toString()));
            }
        }
    }

    private void setHeapSizeAndSplitSize(final List<String> args) {
        // can be override by shifuconfig, ok for hard code
        if(this.isDebug()) {
            args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT,
                    GuaguaMapReduceConstants.MAPRED_CHILD_JAVA_OPTS,
                    "-Xms2048m -Xmx2048m -verbose:gc -XX:+PrintGCDetails -XX:+PrintGCTimeStamps"));
        } else {
            args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT,
                    GuaguaMapReduceConstants.MAPRED_CHILD_JAVA_OPTS,
                    "-Xms2048m -Xmx2048m -verbose:gc -XX:+PrintGCDetails -XX:+PrintGCTimeStamps"));
            args.add(String
                    .format(CommonConstants.MAPREDUCE_PARAM_FORMAT,
                            "mapreduce.map.java.opts",
                            "-Xms2048m -Xmx2048m -server -XX:+UseParNewGC -XX:+UseConcMarkSweepGC "
                                    + "-XX:CMSInitiatingOccupancyFraction=70 -verbose:gc -XX:+PrintGCDetails -XX:+PrintGCTimeStamps"));
        }
        if(super.modelConfig.getNormalize().getIsParquet()) {
            args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, GuaguaConstants.GUAGUA_SPLIT_COMBINABLE,
                    Environment.getProperty(GuaguaConstants.GUAGUA_SPLIT_COMBINABLE, "false")));
        } else {
            args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, GuaguaConstants.GUAGUA_SPLIT_COMBINABLE,
                    Environment.getProperty(GuaguaConstants.GUAGUA_SPLIT_COMBINABLE, "true")));
            // set to 512M to save mappers, sometimes maybe OOM, users should tune guagua.split.maxCombinedSplitSize in
            // shifuconfig
            args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT,
                    GuaguaConstants.GUAGUA_SPLIT_MAX_COMBINED_SPLIT_SIZE,
                    Environment.getProperty(GuaguaConstants.GUAGUA_SPLIT_MAX_COMBINED_SPLIT_SIZE, "268435456")));
        }
        // special tuning parameters for shifu, 0.97 means each iteation master wait for 97% workers and then can go to
        // next iteration.
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, GuaguaConstants.GUAGUA_MIN_WORKERS_RATIO, 0.97));
        // 2 seconds if waiting over 10, consider 99% workers; these two can be overrided in shifuconfig
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, GuaguaConstants.GUAGUA_MIN_WORKERS_TIMEOUT,
                2 * 1000L));
    }

    private void copyModelToLocal(String modelName, Path modelPath, SourceType sourceType) throws IOException {
        if(!this.isDryTrain()) {
            ShifuFileUtils.getFileSystemBySourceType(sourceType).copyToLocalFile(modelPath,
                    new Path(Constants.MODELS, modelName));
        }
    }

    // GuaguaOptionsParser doesn't to support *.jar currently.
    private void addRuntimeJars(final List<String> args) {
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

        String hdpVersion = HDPUtils.getHdpVersionForHDP224();
        if(StringUtils.isNotBlank(hdpVersion)) {
            jars.add(HDPUtils.findContainingFile("hdfs-site.xml"));
            jars.add(HDPUtils.findContainingFile("core-site.xml"));
            jars.add(HDPUtils.findContainingFile("mapred-site.xml"));
            jars.add(HDPUtils.findContainingFile("yarn-site.xml"));
        }

        args.add(StringUtils.join(jars, NNConstants.LIB_JAR_SEPARATOR));
    }

    /**
     * Get model name
     * 
     * @param i
     *            index for model name
     */
    public String getModelName(int i) {
        String alg = super.getModelConfig().getTrain().getAlgorithm();
        return String.format("model%s.%s", i, alg.toLowerCase());
    }

    // d-train part ends here

    public boolean isDryTrain() {
        return isDryTrain;
    }

    public void setDryTrain(boolean isDryTrain) {
        this.isDryTrain = isDryTrain;
    }

    public boolean isDebug() {
        return isDebug;
    }

    public void setDebug(boolean isDebug) {
        this.isDebug = isDebug;
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
     * A thread used to tail progress log from hdfs log file.
     */
    private static class TailThread extends Thread {
        private long offset[];
        private String[] progressLogs;

        public TailThread(String[] progressLogs) {
            this.progressLogs = progressLogs;
            this.offset = new long[this.progressLogs.length];
            for(String progressLog: progressLogs) {
                try {
                    // delete it firstly, it will be updated from master
                    HDFSUtils.getFS().delete(new Path(progressLog), true);
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
            FSDataInputStream in;
            try {
                in = HDFSUtils.getFS().open(item);
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
                IOUtils.copyBytes(in, out, HDFSUtils.getFS().getConf(), false);
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

        public void deleteProgressFiles() throws IOException {
            for(String progressFile: this.progressLogs) {
                HDFSUtils.getFS().delete(new Path(progressFile), true);
            }
        }

    }

}
