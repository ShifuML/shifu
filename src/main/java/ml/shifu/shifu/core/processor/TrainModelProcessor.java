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
package ml.shifu.shifu.core.processor;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.lang.Thread.UncaughtExceptionHandler;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import ml.shifu.shifu.actor.AkkaSystemExecutor;
import ml.shifu.shifu.container.obj.ModelBasicConf.RunMode;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.AbstractTrainer;
import ml.shifu.shifu.core.alg.LogisticRegressionTrainer;
import ml.shifu.shifu.core.alg.NNTrainer;
import ml.shifu.shifu.core.alg.SVMTrainer;
import ml.shifu.shifu.core.dtrain.NNConstants;
import ml.shifu.shifu.core.dtrain.NNMaster;
import ml.shifu.shifu.core.dtrain.NNOutput;
import ml.shifu.shifu.core.dtrain.NNParams;
import ml.shifu.shifu.core.dtrain.NNWorker;
import ml.shifu.shifu.core.validator.ModelInspector.ModelStep;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.Environment;
import ml.shifu.shifu.util.HDFSUtils;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.collections.ListUtils;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.pig.impl.util.JarManager;
import org.apache.zookeeper.ZooKeeper;
import org.encog.ml.data.MLDataSet;
import ml.shifu.guagua.GuaguaConstants;
import ml.shifu.guagua.mapreduce.GuaguaMapReduceClient;
import ml.shifu.guagua.mapreduce.GuaguaMapReduceConstants;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.base.Splitter;

/**
 * Train processor, produce model based on the normalized dataset
 *
 * 
 */
public class TrainModelProcessor extends BasicModelProcessor implements Processor {

    private final static Logger LOG = LoggerFactory.getLogger(TrainModelProcessor.class);

    public static final String SHIFU_DEFAULT_DTRAIN_PARALLEL = "true";

    public static final String SHIFU_DTRAIN_PARALLEL = "shifu.dtrain.parallel";

    private boolean isDryTrain, isDebug;
    private List<AbstractTrainer> trainers;

    private static final String LOGS = "./logs";

    public TrainModelProcessor() {
    }

    /**
     * Constructor
     * 
     * @param isDryTrain
     *            dryTrain flag, if it's true, the trainer would start training
     * @param isDebug
     *            debug flag, if it's true, shifu will create log file to record each training status
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
        setUp(ModelStep.TRAIN);

        if(isDebug) {
            File file = new File(LOGS);
            if(!file.mkdir()) {
                throw new RuntimeException("logs file is created failed.");
            }
        }

        RunMode runMode = super.modelConfig.getBasic().getRunMode();
        switch(runMode) {
            case mapred:
                validatePigTrain();
                syncDataToHdfs(super.modelConfig.getDataSet().getSource());
                runDistributedTrain();
                break;
            case local:
            default:
                runAkkaTrain(modelConfig.getBaggingNum());
                break;
        }

        clearUp(ModelStep.TRAIN);

        LOG.info("Step Finished: train");
        return 0;
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
        models.delete();
        models.mkdir();

        trainers.clear();

        for(int i = 0; i < numBags; i++) {
            AbstractTrainer trainer;
            if(modelConfig.getAlgorithm().equalsIgnoreCase("NN")) {
                trainer = new NNTrainer(modelConfig, i + 1, isDryTrain);
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

    // d-train part starts here
    private void validatePigTrain() {
        if(!NNConstants.NN_ALG_NAME.equalsIgnoreCase(super.getModelConfig().getTrain().getAlgorithm())) {
            throw new IllegalArgumentException("Currently we only support NN distributed training.");
        }

        if(super.getModelConfig().getDataSet().getSource() != SourceType.HDFS) {
            throw new IllegalArgumentException("Currently we only support distributed training on HDFS source type.");
        }

        if(isDebug()) {
            LOG.warn("Currently we haven't debug logic. It's the same as you don't set it.");
        }

    }

    protected void runDistributedTrain() throws IOException, InterruptedException, ClassNotFoundException {
        LOG.info("Started {} d-training.", isDryTrain ? "dry" : "");

        SourceType sourceType = super.getModelConfig().getDataSet().getSource();

        final List<String> args = new ArrayList<String>();

        prepareCommonParams(args, sourceType);

        // add tmp models folder to config
        Path tmpModelsPath = ShifuFileUtils.getFileSystemBySourceType(sourceType).makeQualified(
                new Path(super.getPathFinder().getPathBySourceType(
                        new Path(Constants.TMP, Constants.DEFAULT_MODELS_TMP_FOLDER), sourceType)));
        args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT, NNConstants.NN_TMP_MODELS_FOLDER,
                tmpModelsPath.toString()));
        int baggingNum = super.getModelConfig().getBaggingNum();

        long start = System.currentTimeMillis();
        LOG.info("Distributed trainning with baggingNum: {}", baggingNum);
        boolean isParallel = Boolean.valueOf(
                Environment.getProperty(SHIFU_DTRAIN_PARALLEL, SHIFU_DEFAULT_DTRAIN_PARALLEL)).booleanValue();
        GuaguaMapReduceClient guaguaClient = new GuaguaMapReduceClient();
        List<String> progressLogList = new ArrayList<String>(baggingNum);
        for(int i = 0; i < baggingNum; i++) {
            // set name for each bagging job.
            args.add("-n");
            args.add(String.format("Shifu Master-Workers NN Iteration: %s id:%s", super.getModelConfig()
                    .getModelSetName(), i + 1));
            LOG.info("Start trainer with id: {}", (i + 1));
            String modelName = getModelName(i + 1);
            Path modelPath = ShifuFileUtils.getFileSystemBySourceType(sourceType).makeQualified(
                    new Path(super.getPathFinder().getModelsPath(sourceType), modelName));
            args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT, NNConstants.GUAGUA_NN_OUTPUT,
                    modelPath.toString()));
            args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT, NNConstants.NN_TRAINER_ID, String.valueOf(i + 1)));
            final String progressLogFile = getProgressLogFile(i + 1);
            progressLogList.add(progressLogFile);
            args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT, NNConstants.NN_PROGRESS_FILE, progressLogFile));
            if(isParallel) {
                guaguaClient.addJob(args.toArray(new String[0]));
            } else {
                TailThread tailThread = startTailThread(new String[] { progressLogFile });
                guaguaClient.creatJob(args.toArray(new String[0])).waitForCompletion(true);
                stopTailThread(tailThread);
            }
        }

        if(isParallel) {
            TailThread tailThread = startTailThread(progressLogList.toArray(new String[0]));
            guaguaClient.run();
            stopTailThread(tailThread);
        }

        // copy model files at last.
        for(int i = 0; i < baggingNum; i++) {
            String modelName = getModelName(i + 1);
            Path modelPath = ShifuFileUtils.getFileSystemBySourceType(sourceType).makeQualified(
                    new Path(super.getPathFinder().getModelsPath(sourceType), modelName));
            copyModelToLocal(modelName, modelPath, sourceType);
        }

        // copy temp model files
        copyTmpModelsToLocal(tmpModelsPath, sourceType);
        LOG.info("Distributed trainning finished in {}ms.", System.currentTimeMillis() - start);
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

    private void prepareCommonParams(final List<String> args, final SourceType sourceType) {
        args.add("-libjars");
        addRuntimeJars(args);

        args.add("-i");
        args.add(ShifuFileUtils.getFileSystemBySourceType(sourceType)
                .makeQualified(new Path(super.getPathFinder().getNormalizedDataPath())).toString());

        args.add("-z");
        String zkServers = Environment.getProperty(Environment.ZOO_KEEPER_SERVERS);
        if(StringUtils.isEmpty(zkServers)) {
            throw new IllegalArgumentException(
                    "Zookeeper is used for distributed training coordination, please set 'zookeeperServers' firstly in '$SHIFU_HOME/conf/shifuconfig' file. The value is like 'server1:port1,server2:port2'.");
        }

        args.add(zkServers);

        args.add("-w");
        args.add(NNWorker.class.getName());

        args.add("-m");
        args.add(NNMaster.class.getName());

        args.add("-c");
        // the reason to add 1 is that the first iteration in D-NN implementation is used for training preparation.
        int numTrainEpochs = super.getModelConfig().getTrain().getNumTrainEpochs() + 1;

        args.add(String.valueOf(numTrainEpochs));

        args.add("-mr");
        args.add(NNParams.class.getName());

        args.add("-wr");
        args.add(NNParams.class.getName());

        args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT, NNConstants.MAPRED_JOB_QUEUE_NAME,
                Environment.getProperty(Environment.HADOOP_JOB_QUEUE, Constants.DEFAULT_JOB_QUEUE)));
        args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT, GuaguaConstants.GUAGUA_MASTER_INTERCEPTERS,
                NNOutput.class.getName()));
        args.add(String.format(
                NNConstants.MAPREDUCE_PARAM_FORMAT,
                NNConstants.SHIFU_NN_MODEL_CONFIG,
                ShifuFileUtils.getFileSystemBySourceType(sourceType).makeQualified(
                        new Path(super.getPathFinder().getModelConfigPath(sourceType)))));
        args.add(String.format(
                NNConstants.MAPREDUCE_PARAM_FORMAT,
                NNConstants.SHIFU_NN_COLUMN_CONFIG,
                ShifuFileUtils.getFileSystemBySourceType(sourceType).makeQualified(
                        new Path(super.getPathFinder().getColumnConfigPath(sourceType)))));
        args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT, NNConstants.NN_MODELSET_SOURCE_TYPE, sourceType));
        args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT, NNConstants.NN_DRY_TRAIN, isDryTrain()));
        // hard code set computation threshold for 40s. TODO, set it in shifuconfig.
        args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT, GuaguaConstants.GUAGUA_COMPUTATION_TIME_THRESHOLD,
                40 * 1000l));
        setHeapSizeAndSplitSize(args);

        // one can set guagua conf in shifuconfig
        for(Map.Entry<Object, Object> entry: Environment.getProperties().entrySet()) {
            if(entry.getKey().toString().startsWith("nn") || entry.getKey().toString().startsWith("guagua")
                    || entry.getKey().toString().startsWith("mapred")) {
                args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT, entry.getKey().toString(), entry.getValue()
                        .toString()));
            }
        }

    }

    private void setHeapSizeAndSplitSize(final List<String> args) {
        // TODO tmp setting 1G heap for each worker, need to be set in ModelConfig, each split is set to 256M for heap
        // with 1G, should be set in ModelConfig also. Replace string as constants.
        if(this.isDebug()) {
            args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT, GuaguaMapReduceConstants.MAPRED_CHILD_JAVA_OPTS,
                    "-Xmn128m -Xms1G -Xmx1G -verbose:gc -XX:+PrintGCDetails -XX:+PrintGCTimeStamps"));
        } else {
            args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT, GuaguaMapReduceConstants.MAPRED_CHILD_JAVA_OPTS,
                    "-Xmn128m -Xms1G -Xmx1G"));
        }
        args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT, GuaguaConstants.GUAGUA_SPLIT_COMBINABLE,
                SHIFU_DEFAULT_DTRAIN_PARALLEL));
        args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT,
                GuaguaConstants.GUAGUA_SPLIT_MAX_COMBINED_SPLIT_SIZE, "268435456"));
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

        args.add(StringUtils.join(jars, NNConstants.LIB_JAR_SEPARATOR));
    }

    /**
     * Get NN model name
     * 
     * @param i
     *            index for model name
     */
    public static String getModelName(int i) {
        return String.format("model%s.nn", i);
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
     * A thread used to tail progress log from hdfs log file.
     * 
     * 
     */
    private static class TailThread extends Thread {
        private volatile long offset[];
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
            FSDataInputStream in = HDFSUtils.getFS().open(item);
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
                    LOG.warn(e.getMessage());
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
