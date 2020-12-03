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
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;
import java.util.concurrent.CountDownLatch;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.fs.Path;
import org.apache.pig.tools.pigstats.JobStats;
import org.apache.pig.tools.pigstats.PigStats;
import org.encog.ml.BasicML;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ml.shifu.shifu.actor.AkkaSystemExecutor;
import ml.shifu.shifu.column.NSColumn;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.container.obj.ModelBasicConf;
import ml.shifu.shifu.container.obj.PerformanceResult;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.ConfusionMatrix;
import ml.shifu.shifu.core.PerformanceEvaluator;
import ml.shifu.shifu.core.Scorer;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.dt.IndependentTreeModel;
import ml.shifu.shifu.core.eval.GainChart;
import ml.shifu.shifu.core.model.ModelSpec;
import ml.shifu.shifu.core.validator.ModelInspector.ModelStep;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.pig.PigExecutor;
import ml.shifu.shifu.util.Base64Utils;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.Environment;
import ml.shifu.shifu.util.HdfsPartFile;
import ml.shifu.shifu.util.ModelSpecLoaderUtils;

/**
 * EvalModelProcessor class
 */
public class EvalModelProcessor extends BasicModelProcessor implements Processor {

    /**
     * log object
     */
    private final static Logger LOG = LoggerFactory.getLogger(EvalModelProcessor.class);

    /**
     * Step for evaluation
     */
    public enum EvalStep {
        LIST, NEW, DELETE, RUN, PERF, SCORE, AUDIT, CONFMAT, NORM, GAINCHART;
    }

    public static final String NOSORT = "NOSORT";
    public static final String EXPECT_AUDIT_CNT = "EXPECT_AUDIT_CNT";
    public static final String REF_MODEL = "REF_MODEL";

    private String evalName = null;

    private EvalStep evalStep;

    private long evalRecords = 0l;

    private static final Random RANDOM = new Random();

    /**
     * Constructor
     * 
     * @param step
     *            the evaluation step
     */
    public EvalModelProcessor(EvalStep step) {
        this.evalStep = step;
    }

    public EvalModelProcessor(EvalStep step, Map<String, Object> otherConfigs) {
        this.evalStep = step;
        super.otherConfigs = otherConfigs;
    }

    /**
     * Constructor
     * 
     * @param step
     *            the evaluation step
     * @param name
     *            the evaluation name
     */
    public EvalModelProcessor(EvalStep step, String name) {
        this.evalName = name;
        this.evalStep = step;
    }

    /**
     * Constructor
     *
     * @param step
     *            the evaluation step
     * @param name
     *            the evaluation name
     * @param params
     *            the command params
     */
    public EvalModelProcessor(EvalStep step, String name, Map<String, Object> params) {
        this(step, name);
        this.params = params;
    }

    /**
     * Runner for evaluation
     */
    @Override
    public int run() throws Exception {
        LOG.info("Step Start: eval");
        long start = System.currentTimeMillis();
        try {
            if(needsToCopyRefModels(evalStep)) {
                if(!copyRefModels()) {
                    LOG.error("Fail to copy refer models.");
                    return -1;
                }
            }

            setUp(ModelStep.EVAL);
            syncDataToHdfs(modelConfig.getDataSet().getSource());

            switch(evalStep) {
                case LIST:
                    listEvalSet();
                    break;
                case NEW:
                    createNewEval(evalName);
                    break;
                case DELETE:
                    deleteEvalSet(evalName);
                    break;
                case RUN:
                    runEval(getEvalConfigListFromInput());
                    break;
                case NORM:
                    runNormalize(getEvalConfigListFromInput());
                    break;
                case PERF:
                    // FIXME, here should be failed because of this.evalRecords is 0. how to fix it
                    runPerformance(getEvalConfigListFromInput());
                    break;
                case SCORE:
                    runScore(getEvalConfigListFromInput());
                    break;
                case AUDIT:
                    runGenAudit(getEvalConfigListFromInput());
                    break;
                case CONFMAT:
                    // FIXME, here should be failed
                    runConfusionMatrix(getEvalConfigListFromInput());
                    break;
                default:
                    break;
            }

            syncDataToHdfs(modelConfig.getDataSet().getSource());

            clearUp(ModelStep.EVAL);
        } catch (ShifuException e) {
            LOG.error("Error:" + e.getError().toString() + "; msg:" + e.getMessage(), e);
            return -1;
        } catch (Exception e) {
            LOG.error("Error:" + e.getMessage(), e);
            return -1;
        }
        LOG.info("Step Finished: eval with {} ms", (System.currentTimeMillis() - start));
        return 0;
    }

    private void deleteEvalSet(String evalSetName) {
        EvalConfig evalConfig = modelConfig.getEvalConfigByName(evalSetName);
        if(evalConfig == null) {
            LOG.error("{} eval set doesn't exist.", evalSetName);
        } else {
            modelConfig.getEvals().remove(evalConfig);
            try {
                saveModelConfig();
            } catch (IOException e) {
                throw new ShifuException(ShifuErrorCode.ERROR_WRITE_MODELCONFIG, e);
            }
            LOG.info("Done. Delete eval set - " + evalSetName);
        }
    }

    private void listEvalSet() {
        List<EvalConfig> evals = modelConfig.getEvals();
        if(CollectionUtils.isNotEmpty(evals)) {
            LOG.info("There are {} eval sets.", evals.size());
            for(EvalConfig evalConfig: evals) {
                LOG.info("\t - {}", evalConfig.getName());
            }
        }
    }

    private List<EvalConfig> getEvalConfigListFromInput() {
        List<EvalConfig> evalSetList = new ArrayList<EvalConfig>();

        if(StringUtils.isNotBlank(evalName)) {
            String[] evalList = evalName.split(",");
            for(String eval: evalList) {
                EvalConfig evalConfig = modelConfig.getEvalConfigByName(eval);
                if(evalConfig == null) {
                    LOG.error("The evalset - " + eval + " doesn't exist!");
                } else {
                    evalSetList.add(evalConfig);
                }
            }
        } else {
            evalSetList = modelConfig.getEvals();
            if(CollectionUtils.isEmpty(evalSetList)) {
                throw new ShifuException(ShifuErrorCode.ERROR_MODEL_EVALSET_DOESNT_EXIST);
            }
        }

        return evalSetList;
    }

    /**
     * run score only
     * 
     * @param evalSetList
     *            eval config list
     * @throws IOException
     *             any io exception
     */
    private void runScore(List<EvalConfig> evalSetList) throws IOException {
        // do the validation before scoring the data set
        for(EvalConfig evalConfig: evalSetList) {
            validateEvalColumnConfig(evalConfig);
        }

        // do it only once
        syncDataToHdfs(evalSetList);

        if(Environment.getBoolean(Constants.SHIFU_EVAL_PARALLEL, true) && modelConfig.isMapReduceRunMode()
                && evalSetList.size() > 1) {
            // run in parallel
            int parallelNum = Environment.getInt(Constants.SHIFU_EVAL_PARALLEL_NUM, 5);
            if(parallelNum <= 0 || parallelNum > 100) {
                throw new IllegalArgumentException(Constants.SHIFU_EVAL_PARALLEL_NUM
                        + " in shifuconfig should be in (0, 100], by default it is 5.");
            }

            int evalSize = evalSetList.size();
            int mod = evalSize % parallelNum;
            int batch = evalSize / parallelNum;
            batch = (mod == 0 ? batch : (batch + 1));

            for(int i = 0; i < batch; i++) {
                int batchSize = (mod != 0 && i == (batch - 1)) ? mod : parallelNum;
                // lunch current batch size
                LOG.info("Starting to run eval score in {}/{} round", (i + 1), batch);
                final CountDownLatch cdl = new CountDownLatch(batchSize);
                for(int j = 0; j < batchSize; j++) {
                    int currentIndex = i * parallelNum + j;
                    final EvalConfig config = evalSetList.get(currentIndex);
                    // save tmp models
                    Thread evalRunThread = new Thread(new Runnable() {
                        @Override
                        public void run() {
                            try {
                                runScore(config);
                            } catch (IOException e) {
                                LOG.error("Exception in eval score:", e);
                            } catch (Exception e) {
                                LOG.error("Exception in eval score:", e);
                            }
                            cdl.countDown();
                        }
                    }, config.getName());
                    // print eval name to log4j console to make each one is easy to be get from logs
                    evalRunThread.start();

                    // each one sleep 4s to avoid conflict in initialization
                    try {
                        Thread.sleep(4000);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                }

                LOG.info("Starting to wait eval score in {}/{} round", (i + 1), batch);
                // await all threads done
                try {
                    cdl.await();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                LOG.info("Finish eval score in {}/{} round", (i + 1), batch);
            }
            LOG.info("Finish all eval score parallel running with eval size {}.", evalSize);
        } else {
            // for old sequential runs
            for(final EvalConfig config: evalSetList) {
                runScore(config);
            }
        }
    }

    /**
     * Run score only
     * 
     * @param config
     *            the eval config instance
     * @throws IOException
     *             any io exception
     */
    private void runScore(EvalConfig config) throws IOException {
        // create evalset home directory firstly in local file system
        // PathFinder pathFinder = new PathFinder(modelConfig);
        // String evalSetPath = pathFinder.getEvalSetPath(config, SourceType.LOCAL);
        // FileUtils.forceMkdir(new File(evalSetPath));
        // syncDataToHdfs(config.getDataSet().getSource());

        switch(modelConfig.getBasic().getRunMode()) {
            case DIST:
            case MAPRED:
                runDistScore(config, -1); // FIXME, only support non multi-task training
                break;
            case LOCAL:
                runAkkaScore(config);
                break;
            default:
                break;
        }
    }

    /**
     * Run normalization against the evaluation data sets based on existing ColumnConfig.json which is from training
     * data set.
     * 
     * @param evalConfigList
     *            the eval config list
     * @throws IOException
     *             any io exception
     */
    private void runNormalize(List<EvalConfig> evalConfigList) throws IOException {
        for(EvalConfig evalConfig: evalConfigList) {
            runNormalize(evalConfig);
        }
    }

    /**
     * Run normalization against the evaluation data set based on existing ColumnConfig.json which is from training
     * data set.
     * 
     * @param evalConfig
     *            the eval config instance
     * @throws IOException
     *             when any IO exception
     * @throws IllegalArgumentException
     *             if LOCAL run mode
     */
    private void runNormalize(EvalConfig evalConfig) throws IOException {
        String evalSetPath = super.pathFinder.getEvalSetPath(evalConfig, SourceType.LOCAL);
        FileUtils.forceMkdir(new File(evalSetPath));
        syncDataToHdfs(evalConfig.getDataSet().getSource());

        switch(modelConfig.getBasic().getRunMode()) {
            case DIST:
            case MAPRED:
                runPigNormalize(evalConfig);
                break;
            case LOCAL:
            default:
                throw new IllegalArgumentException("Eval norm doesn't support LOCAL run mode.");
        }
    }

    /**
     * run pig mode scoring
     * 
     * @param evalConfig
     *            the name for evaluation
     * @throws IOException
     *             any io exception
     */
    @SuppressWarnings("deprecation")
    private ScoreStatus runDistScore(EvalConfig evalConfig, int index) throws IOException {
        // clean up output directories
        SourceType sourceType = evalConfig.getDataSet().getSource();

        ShifuFileUtils.deleteFile(pathFinder.getEvalNormalizedPath(evalConfig), sourceType);
        ShifuFileUtils.deleteFile(pathFinder.getEvalScorePath(evalConfig), sourceType);
        ShifuFileUtils.deleteFile(pathFinder.getEvalPerformancePath(evalConfig), sourceType);

        // prepare special parameters and execute pig
        Map<String, String> paramsMap = new HashMap<String, String>();

        paramsMap.put(Constants.SOURCE_TYPE, sourceType.toString());
        paramsMap.put("pathEvalRawData", evalConfig.getDataSet().getDataPath());
        paramsMap.put("pathEvalNormalized", pathFinder.getEvalNormalizedPath(evalConfig));
        paramsMap.put("pathEvalScore", pathFinder.getEvalScorePath(evalConfig));
        paramsMap.put("pathEvalPerformance", pathFinder.getEvalPerformancePath(evalConfig));
        paramsMap.put("eval_set_name", evalConfig.getName());
        paramsMap.put("delimiter", CommonUtils.escapePigString(evalConfig.getDataSet().getDataDelimiter()));
        if(index == -1 && !modelConfig.isMultiTask()) {
            paramsMap.put("columnIndex", evalConfig.getPerformanceScoreSelector().trim());
        } else {
            paramsMap.put("columnIndex", "model" + index); // hard code here, TODO, need extract
        }
        paramsMap.put("scale",
                Environment.getProperty(Constants.SHIFU_SCORE_SCALE, Integer.toString(Scorer.DEFAULT_SCORE_SCALE)));
        paramsMap.put(CommonConstants.MTL_INDEX, index + "");

        String expressionsAsString = super.modelConfig.getSegmentFilterExpressionsAsString();
        Environment.getProperties().put("shifu.segment.expressions", expressionsAsString);

        String pigScript = "scripts/Eval.pig";
        Map<String, String> confMap = new HashMap<String, String>();

        // max min score folder
        Path path = new Path(
                "tmp" + File.separator + "maxmin_score_" + System.currentTimeMillis() + "_" + RANDOM.nextLong());
        String maxMinScoreFolder = ShifuFileUtils.getFileSystemBySourceType(sourceType, path).makeQualified(path)
                .toString();
        confMap.put(Constants.SHIFU_EVAL_MAXMIN_SCORE_OUTPUT, maxMinScoreFolder);
        if(modelConfig.isClassification()
                || (isNoSort() && (EvalStep.SCORE.equals(this.evalStep) || EvalStep.AUDIT.equals(this.evalStep)))) {
            pigScript = "scripts/EvalScore.pig";
        }
        try {
            PigExecutor.getExecutor().submitJob(modelConfig, pathFinder.getScriptPath(pigScript), paramsMap,
                    evalConfig.getDataSet().getSource(), confMap, super.pathFinder);
        } catch (IOException e) {
            throw new ShifuException(ShifuErrorCode.ERROR_RUNNING_PIG_JOB, e);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }

        Iterator<JobStats> iter = PigStats.get().getJobGraph().iterator();

        while(iter.hasNext()) {
            JobStats jobStats = iter.next();
            long evalRecords = jobStats.getHadoopCounters().getGroup(Constants.SHIFU_GROUP_COUNTER)
                    .getCounter(Constants.COUNTER_RECORDS);
            LOG.info("Total valid eval records is : {}", evalRecords);
            // If no basic record counter, check next one
            if(evalRecords == 0L) {
                continue;
            }
            this.evalRecords = evalRecords;
            // mtlIndex here set to -1 since each eval pig job, output COUNTER are the same name.
            return getScoreStatus(sourceType, maxMinScoreFolder, jobStats, evalRecords, -1);
        }
        return null;
    }

    @SuppressWarnings("deprecation")
    private ScoreStatus getScoreStatus(SourceType sourceType, String maxMinScoreFolder, JobStats jobStats,
            long evalRecords, int postfix) throws IOException {
        long pigPosTags = jobStats.getHadoopCounters().getGroup(Constants.SHIFU_GROUP_COUNTER)
                .getCounter(postfix == -1 ? Constants.COUNTER_POSTAGS : Constants.COUNTER_POSTAGS + "_" + postfix);
        long pigNegTags = jobStats.getHadoopCounters().getGroup(Constants.SHIFU_GROUP_COUNTER)
                .getCounter(postfix == -1 ? Constants.COUNTER_NEGTAGS : Constants.COUNTER_NEGTAGS + "_" + postfix);

        double pigPosWeightTags = jobStats.getHadoopCounters().getGroup(Constants.SHIFU_GROUP_COUNTER)
                .getCounter(postfix == -1 ? Constants.COUNTER_WPOSTAGS : Constants.COUNTER_WPOSTAGS + "_" + postfix)
                / (Constants.EVAL_COUNTER_WEIGHT_SCALE * 1.0d);
        double pigNegWeightTags = jobStats.getHadoopCounters().getGroup(Constants.SHIFU_GROUP_COUNTER)
                .getCounter(postfix == -1 ? Constants.COUNTER_WNEGTAGS : Constants.COUNTER_WNEGTAGS + "_" + postfix)
                / (Constants.EVAL_COUNTER_WEIGHT_SCALE * 1.0d);

        LOG.info("Total positive record count is : {}", pigPosTags);
        LOG.info("Total negative record count is : {}", pigNegTags);
        LOG.info("Total weighted positive record count is : {}", pigPosWeightTags);
        LOG.info("Total weighted negative record count is : {}", pigNegWeightTags);

        long totalRunTime = jobStats.getHadoopCounters().getGroup(Constants.SHIFU_GROUP_COUNTER)
                .getCounter(Constants.TOTAL_MODEL_RUNTIME);

        LOG.info("Avg SLA for eval model scoring is {} micro seconds", totalRunTime / evalRecords);

        double maxScore = Integer.MIN_VALUE;
        double minScore = Integer.MAX_VALUE;
        if(modelConfig.isRegression()) {
            double[] maxMinScores = locateMaxMinScoreFromFile(sourceType, maxMinScoreFolder);
            maxScore = maxMinScores[0];
            minScore = maxMinScores[1];
            LOG.info("Raw max score is {}, raw min score is {}", maxScore, minScore);
            if(postfix == -1
                    || (this.mtlColumnConfigLists != null && postfix == this.mtlColumnConfigLists.size() - 1)) {
                ShifuFileUtils.deleteFile(maxMinScoreFolder, sourceType);
            }
        }
        // only one pig job with such counters, return
        return new ScoreStatus(pigPosTags, pigNegTags, pigPosWeightTags, pigNegWeightTags, maxScore, minScore,
                evalRecords);
    }

    private double[] locateMaxMinScoreFromFile(SourceType sourceType, String maxMinScoreFolder) throws IOException {
        List<Scanner> scanners = null;
        double maxScore = Double.MIN_VALUE;
        double minScore = Double.MAX_VALUE;
        try {
            // here only works for 1 reducer
            scanners = ShifuFileUtils.getDataScanners(maxMinScoreFolder, sourceType);
            for(Scanner scanner: scanners) {
                if(scanner.hasNext()) {
                    String line = scanner.nextLine().trim();
                    String[] splits = line.split(",");
                    if(splits.length >= 2) {
                        double localMaxScore = Double.parseDouble(splits[0]);
                        if(maxScore < localMaxScore) {
                            maxScore = localMaxScore;
                        }

                        Double localMinScore = Double.parseDouble(splits[1]);
                        if(minScore > localMinScore) {
                            minScore = localMinScore;
                        }
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
        return new double[] { maxScore, minScore };
    }

    /**
     * Normalize evaluation dataset based on pig distributed solution.
     * 
     * @param evalConfig
     *            eval config instance
     * @throws IOException
     *             any io exception
     */
    private void runPigNormalize(EvalConfig evalConfig) throws IOException {
        SourceType sourceType = evalConfig.getDataSet().getSource();

        // clean up output directories
        ShifuFileUtils.deleteFile(pathFinder.getEvalNormalizedPath(evalConfig), sourceType);

        // prepare special parameters and execute pig
        Map<String, String> paramsMap = new HashMap<String, String>();

        paramsMap.put(Constants.SOURCE_TYPE, sourceType.toString());
        paramsMap.put("pathEvalRawData", evalConfig.getDataSet().getDataPath());
        paramsMap.put("pathEvalNormalized", pathFinder.getEvalNormalizedPath(evalConfig));
        paramsMap.put("eval_set_name", evalConfig.getName());
        paramsMap.put("delimiter", evalConfig.getDataSet().getDataDelimiter());
        paramsMap.put("scale",
                Environment.getProperty(Constants.SHIFU_SCORE_SCALE, Integer.toString(Scorer.DEFAULT_SCORE_SCALE)));
        paramsMap.put(Constants.STRICT_MODE, Boolean.toString(isStrict()));

        String expressionsAsString = super.modelConfig.getSegmentFilterExpressionsAsString();
        Environment.getProperties().put("shifu.segment.expressions", expressionsAsString);

        String pigScript = "scripts/EvalNorm.pig";

        try {
            PigExecutor.getExecutor().submitJob(modelConfig, pathFinder.getScriptPath(pigScript), paramsMap,
                    evalConfig.getDataSet().getSource());
        } catch (IOException e) {
            throw new ShifuException(ShifuErrorCode.ERROR_RUNNING_PIG_JOB, e);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * run akka mode scoring
     * 
     * @param config
     *            the name for evaluation
     * @throws IOException
     *             any io exception
     */
    private void runAkkaScore(EvalConfig config) throws IOException {
        SourceType sourceType = config.getDataSet().getSource();
        List<Scanner> scanners = ShifuFileUtils
                .getDataScanners(ShifuFileUtils.expandPath(config.getDataSet().getDataPath(), sourceType), sourceType);

        AkkaSystemExecutor.getExecutor().submitModelEvalJob(modelConfig,
                ShifuFileUtils.searchColumnConfig(config, this.columnConfigList), config, scanners);

        // FIXME A bug here in local mode, compute eval records please
        // this.evalRecords = ...;
        closeScanners(scanners);
    }

    /**
     * Create a evaluation with <code>name</code>
     * 
     * @param name
     *            - the evaluation set name
     * @throws IOException
     *             any io exception
     */
    private void createNewEval(String name) throws IOException {
        EvalConfig evalConfig = modelConfig.getEvalConfigByName(name);
        if(evalConfig != null) {
            throw new ShifuException(ShifuErrorCode.ERROR_MODEL_EVALSET_ALREADY_EXIST,
                    "EvalSet - " + name + " already exists in ModelConfig. Please use another evalset name");
        }

        evalConfig = new EvalConfig();
        evalConfig.setName(name);
        evalConfig.setDataSet(modelConfig.getDataSet().cloneRawSourceData());
        // create empty <EvalSetName>Score.meta.column.names
        ShifuFileUtils.createFileIfNotExists(
                new Path(evalConfig.getName() + Constants.DEFAULT_CHAMPIONSCORE_META_COLUMN_FILE).toString(),
                SourceType.LOCAL);

        // create empty <EvalSetName>.meta.column.names
        String namesFilePath = Constants.COLUMN_META_FOLDER_NAME + File.separator + evalConfig.getName() + "."
                + Constants.DEFAULT_META_COLUMN_FILE;
        ShifuFileUtils.createFileIfNotExists(new Path(namesFilePath).toString(), SourceType.LOCAL);
        evalConfig.getDataSet().setMetaColumnNameFile(namesFilePath);

        modelConfig.getEvals().add(evalConfig);

        try {
            saveModelConfig();
        } catch (IOException e) {
            throw new ShifuException(ShifuErrorCode.ERROR_WRITE_MODELCONFIG, e);
        }
        LOG.info("Create Eval - " + name);
    }

    /**
     * Running evaluation including scoring and performance evaluation two steps.
     * 
     * <p>
     * This function will switch to pig or akka evaluation depends on the modelConfig running mode
     * 
     * @throws IOException
     *             any exception in running pig evaluation or akka evaluation
     */
    private void runEval(List<EvalConfig> evalSetList) throws IOException {
        // do it only once
        syncDataToHdfs(evalSetList);

        // validation for score column
        for(EvalConfig evalConfig: evalSetList) {
            List<String> scoreMetaColumns = evalConfig.getScoreMetaColumns(modelConfig);
            if(scoreMetaColumns.size() > 5) {
                LOG.error(
                        "Starting from 0.10.x, 'scoreMetaColumns' is used for benchmark score columns and limited to at most 5.");
                LOG.error(
                        "If meta columns are set in file of 'scoreMetaColumns', please move meta column config to 'eval#dataSet#metaColumnNameFile' part.");
                LOG.error(
                        "If 'eval#dataSet#metaColumnNameFile' is duplicated with training 'metaColumnNameFile', you can rename it to another file with different name.");
                return;
            }
        }

        if(Environment.getBoolean(Constants.SHIFU_EVAL_PARALLEL, true) && modelConfig.isMapReduceRunMode()
                && evalSetList.size() > 1) {
            // run in parallel
            int parallelNum = Environment.getInt(Constants.SHIFU_EVAL_PARALLEL_NUM, 5);
            if(parallelNum <= 0 || parallelNum > 100) {
                throw new IllegalArgumentException(Constants.SHIFU_EVAL_PARALLEL_NUM
                        + " in shifuconfig should be in (0, 100], by default it is 5.");
            }

            int evalSize = evalSetList.size();
            int mod = evalSize % parallelNum;
            int batch = evalSize / parallelNum;
            batch = (mod == 0 ? batch : (batch + 1));

            for(int i = 0; i < batch; i++) {
                int batchSize = (mod != 0 && i == (batch - 1)) ? mod : parallelNum;
                // lunch current batch size
                LOG.info("Starting to run eval score in {}/{} round", (i + 1), batch);
                final CountDownLatch cdl = new CountDownLatch(batchSize);
                for(int j = 0; j < batchSize; j++) {
                    int currentIndex = i * parallelNum + j;
                    final EvalConfig config = evalSetList.get(currentIndex);
                    // save tmp models
                    Thread evalRunThread = new Thread(new Runnable() {
                        @Override
                        public void run() {
                            try {
                                runEval(config);
                            } catch (IOException e) {
                                LOG.error("Exception in eval:", e);
                            } catch (Exception e) {
                                LOG.error("Exception in eval:", e);
                            }
                            cdl.countDown();
                        }
                    }, config.getName());
                    // print eval name to log4j console to make each one is easy to be get from logs
                    evalRunThread.start();

                    // each one sleep 3s to avoid conflict in initialization
                    try {
                        Thread.sleep(3000);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                }

                LOG.info("Starting to wait eval in {}/{} round", (i + 1), batch);
                // await all threads done
                try {
                    cdl.await();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                LOG.info("Finish eval in {}/{} round", (i + 1), batch);
            }
            LOG.info("Finish all eval parallel running with eval size {}.", evalSize);
        } else {
            // for old sequential runs
            for(EvalConfig evalConfig: evalSetList) {
                runEval(evalConfig);
            }
        }
    }

    @SuppressWarnings("deprecation")
    private void validateEvalColumnConfig(EvalConfig evalConfig) throws IOException {
        if(this.columnConfigList == null) {
            return;
        }

        String[] evalColumnNames = null;

        if(StringUtils.isNotBlank(evalConfig.getDataSet().getHeaderPath())) {
            String delimiter = StringUtils.isBlank(evalConfig.getDataSet().getHeaderDelimiter()) // get header delimiter
                    ? evalConfig.getDataSet().getDataDelimiter()
                    : evalConfig.getDataSet().getHeaderDelimiter();
            evalColumnNames = CommonUtils.getHeaders(evalConfig.getDataSet().getHeaderPath(), delimiter,
                    evalConfig.getDataSet().getSource());
        } else {
            String delimiter = StringUtils.isBlank(evalConfig.getDataSet().getHeaderDelimiter()) // get header delimiter
                    ? evalConfig.getDataSet().getDataDelimiter()
                    : evalConfig.getDataSet().getHeaderDelimiter();
            String[] fields = CommonUtils.takeFirstLine(evalConfig.getDataSet().getDataPath(), delimiter,
                    evalConfig.getDataSet().getSource());
            // first line of data meaning second line in data files excluding first header line
            String[] dataInFirstLine = CommonUtils.takeFirstTwoLines(evalConfig.getDataSet().getDataPath(), delimiter,
                    evalConfig.getDataSet().getSource())[1];
            if(dataInFirstLine != null && fields.length != dataInFirstLine.length) {
                throw new IllegalArgumentException(
                        "Eval header length and eval data length are not consistent, please check you header setting and data set setting in eval.");
            }

            // replace empty and / to _ to avoid pig column schema parsing issue, all columns with empty
            // char or / in its name in shifu will be replaced;
            for(int i = 0; i < fields.length; i++) {
                fields[i] = CommonUtils.normColumnName(fields[i]);
            }
            evalColumnNames = fields;
            // for(int i = 0; i < fields.length; i++) {
            // evalColumnNames[i] = CommonUtils.getRelativePigHeaderColumnName(fields[i]);
            // }
            LOG.warn("No header path is provided, we will try to read first line and detect schema.");
            LOG.warn("Schema in ColumnConfig.json are named as first line of data set path.");
        }

        Set<NSColumn> names = new HashSet<NSColumn>();
        for(String evalColumnName: evalColumnNames) {
            names.add(new NSColumn(evalColumnName));
        }

        String filterExpressions = super.modelConfig.getSegmentFilterExpressionsAsString();
        if(StringUtils.isNotBlank(filterExpressions)) {
            int segFilterSize = CommonUtils.split(filterExpressions,
                    Constants.SHIFU_STATS_FILTER_EXPRESSIONS_DELIMETER).length;
            for(int i = 0; i < segFilterSize; i++) {
                for(int j = 0; j < evalColumnNames.length; j++) {
                    names.add(new NSColumn(evalColumnNames[j] + "_" + (i + 1)));
                }
            }
        }
        if(Constants.GENERIC.equalsIgnoreCase(modelConfig.getAlgorithm())
                || Constants.TENSORFLOW.equalsIgnoreCase(modelConfig.getAlgorithm())
                || CommonUtils.isWDLModel(modelConfig.getAlgorithm())) {
            // TODO correct this logic
            return;
        }

        String evalTargetName = modelConfig.getTargetColumnName(evalConfig, null);
        NSColumn targetColumn = new NSColumn(evalTargetName);
        if(StringUtils.isNotBlank(evalTargetName) && !names.contains(targetColumn)
                && !names.contains(new NSColumn(targetColumn.getSimpleName()))) {
            throw new IllegalArgumentException("Target column " + evalTargetName + " does not exist in - "
                    + evalConfig.getDataSet().getHeaderPath());
        }

        NSColumn weightColumn = new NSColumn(evalConfig.getDataSet().getWeightColumnName());
        if(StringUtils.isNotBlank(evalConfig.getDataSet().getWeightColumnName()) && !names.contains(weightColumn)
                && !names.contains(new NSColumn(weightColumn.getSimpleName()))) {
            throw new IllegalArgumentException("Weight column " + evalConfig.getDataSet().getWeightColumnName()
                    + " does not exist in - " + evalConfig.getDataSet().getHeaderPath());
        }

        List<BasicML> models = null;
        try {
            models = ModelSpecLoaderUtils.loadBasicModels(modelConfig, evalConfig, SourceType.LOCAL,
                    evalConfig.getGbtConvertToProb(), evalConfig.getGbtScoreConvertStrategy());
        } catch (IOException e) {
            // if models folder not created or other exception, just ignore exception and skip models validations,
            // warning is ok.
            LOG.warn("Error occurred when loading models.", e);
        }
        if(CollectionUtils.isNotEmpty(models)) {
            validateFinalColumns(evalConfig, this.modelConfig.getModelSetName(), false, this.columnConfigList, names);
        }

        // no need check exception for submodels as loadSubModels has handled IOException well, TODO, inconsistent
        // exception processing with loadBasicModels
        List<ModelSpec> subModels = ModelSpecLoaderUtils.loadSubModels(modelConfig, this.columnConfigList, evalConfig,
                SourceType.LOCAL, evalConfig.getGbtConvertToProb(), evalConfig.getGbtScoreConvertStrategy());
        if(CollectionUtils.isNotEmpty(subModels)) {
            for(ModelSpec modelSpec: subModels) {
                validateFinalColumns(evalConfig, modelSpec.getModelName(), true, modelSpec.getColumnConfigList(),
                        names);
            }
        }
    }

    private void validateFinalColumns(EvalConfig evalConfig, String modelName, boolean isSubModel,
            List<ColumnConfig> columnConfigs, Set<NSColumn> names) {
        for(ColumnConfig config: columnConfigs) {
            NSColumn nsColumn = new NSColumn(config.getColumnName());
            if(config.isFinalSelect() && !names.contains(nsColumn)
                    && !names.contains(new NSColumn(nsColumn.getSimpleName()))) {
                throw new IllegalArgumentException(
                        "Final selected column " + config.getColumnName() + " in " + (isSubModel ? "sub[" : "current[")
                                + modelName + "]" + " does not exist in - " + evalConfig.getDataSet().getHeaderPath());
            }
        }
    }

    /**
     * Run evaluation per EvalConfig.
     * 
     * @param evalConfig
     *            the evaluation config instance.
     * @throws IOException
     *             when any IO exception
     */
    private void runEval(EvalConfig evalConfig) throws IOException {
        // create evalset home directory firstly in local file system
        synchronized(this) {
            validateEvalColumnConfig(evalConfig);
            // String evalSetPath = pathFinder.getEvalSetPath(evalConfig, SourceType.LOCAL);
            // FileUtils.forceMkdir(new File(evalSetPath));
            // syncDataToHdfs(evalConfig.getDataSet().getSource());
        }

        // each one sleep 8s to avoid conflict in initialization
        try {
            Thread.sleep(8000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        switch(modelConfig.getBasic().getRunMode()) {
            case DIST:
            case MAPRED:
                runDistEval(evalConfig);
                break;
            case LOCAL:
                runAkkaEval(evalConfig);
                break;
            default:
                break;
        }
    }

    @SuppressWarnings("deprecation")
    private boolean isGBTNotConvertToProb(EvalConfig evalConfig) {
        if(CommonConstants.GBT_ALG_NAME.equalsIgnoreCase(modelConfig.getTrain().getAlgorithm())) {
            if(IndependentTreeModel.isValidGbtScoreConvertStrategy(evalConfig.getGbtScoreConvertStrategy())) {
                if(Constants.GBT_SCORE_RAW_CONVETER.equalsIgnoreCase(evalConfig.getGbtScoreConvertStrategy())) {
                    return true;
                }
            } else {
                // if score strategy not set, check deprecated parameter getGbtConvertToProb
                if(!evalConfig.getGbtConvertToProb()) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Run distributed version of evaluation and performance review.
     * 
     * @param evalConfig
     *            the evaluation instance
     * @throws IOException
     *             when any exception in delete the old tmp files
     */
    private void runDistEval(EvalConfig evalConfig) throws IOException {
        if(modelConfig.isMultiTask()) {
            for(int i = 0; i < this.modelConfig.getMultiTaskTargetColumnNames().size(); i++) {
                runDistEval(evalConfig, i);
            }
        } else {
            runDistEval(evalConfig, -1);
        }
    }

    /**
     * Run distributed version of evaluation and performance review.
     * 
     * @param evalConfig
     *            the evaluation instance
     * @param mtlIndex
     *            multi-task index, if -1 means not multi-task evaluation
     * @throws IOException
     *             when any exception in delete the old tmp files
     */
    private void runDistEval(EvalConfig evalConfig, int mtlIndex) throws IOException {
        ScoreStatus ss = runDistScore(evalConfig, mtlIndex);

        List<String> scoreMetaColumns = evalConfig.getScoreMetaColumns(modelConfig);
        addReferModelScoreColumns(scoreMetaColumns);
        if(scoreMetaColumns == null || scoreMetaColumns.isEmpty() || !modelConfig.isRegression()) {
            // if no any champion score column set, go to previous evaluation with only challendge model
            runConfusionMatrix(evalConfig, ss, isGBTNotConvertToProb(evalConfig), false, -1);
            return;
        }

        // 1. Get challenge model performance
        List<PerformanceResult> prList = new ArrayList<PerformanceResult>();
        List<String> names = new ArrayList<String>();

        PerformanceResult challengeModelPerformance = runConfusionMatrix(evalConfig, ss,
                pathFinder.getEvalScorePath(evalConfig), pathFinder.getEvalPerformancePath(evalConfig), false, false,
                isGBTNotConvertToProb(evalConfig), modelConfig.isMultiTask(), mtlIndex);
        prList.add(challengeModelPerformance);
        if(mtlIndex == -1 && !modelConfig.isMultiTask()) {
            names.add(modelConfig.getBasic().getName() + "-" + evalConfig.getName());
        } else {
            names.add(modelConfig.getBasic().getName() + "-" + evalConfig.getName() + "-"
                    + modelConfig.getMultiTaskTargetColumnNames().get(mtlIndex));
        }

        // 2. Get all champion model performance
        for(String metaScoreColumn: scoreMetaColumns) {
            if(StringUtils.isBlank(metaScoreColumn)) {
                continue;
            }
            names.add(metaScoreColumn);

            LOG.info("Model score sort for {} in eval {} is started.", metaScoreColumn, evalConfig.getName());
            ScoreStatus newScoreStatus = runDistMetaScore(evalConfig, metaScoreColumn, mtlIndex);

            PerformanceResult championModelPerformance = runConfusionMatrix(evalConfig, newScoreStatus,
                    pathFinder.getEvalMetaScorePath(evalConfig, metaScoreColumn),
                    pathFinder.getEvalMetaPerformancePath(evalConfig, metaScoreColumn), false, false, 0, 1, 2,
                    modelConfig.isMultiTask(), mtlIndex);
            prList.add(championModelPerformance);
        }

        synchronized(this) {
            GainChart gc = new GainChart();
            boolean hasWeight = StringUtils.isNotBlank(evalConfig.getDataSet().getWeightColumnName());

            // 3. Compute gain chart and other eval performance files only in local.
            String htmlGainChart = pathFinder.getEvalFilePath(evalConfig.getName(),
                    evalConfig.getName() + "_gainchart.html", SourceType.LOCAL);
            LOG.info("Gain chart is generated in {}.", htmlGainChart);
            gc.generateHtml(evalConfig, modelConfig, htmlGainChart, prList, names);

            String hrmlPrRoc = pathFinder.getEvalFilePath(evalConfig.getName(), evalConfig.getName() + "_prroc.html",
                    SourceType.LOCAL);
            LOG.info("PR & ROC chart is generated in {}.", hrmlPrRoc);
            gc.generateHtml4PrAndRoc(evalConfig, modelConfig, hrmlPrRoc, prList, names);

            for(int i = 0; i < names.size(); i++) {
                String name = names.get(i);
                PerformanceResult pr = prList.get(i);
                String unitGainChartCsv = pathFinder.getEvalFilePath(evalConfig.getName(),
                        name + "_unit_wise_gainchart.csv", SourceType.LOCAL);
                LOG.info("Unit-wise gain chart data is generated in {} for eval {} and name {}.", unitGainChartCsv,
                        evalConfig.getName(), name);
                gc.generateCsv(evalConfig, modelConfig, unitGainChartCsv, pr.gains);
                if(hasWeight) {
                    String weightedGainChartCsv = pathFinder.getEvalFilePath(evalConfig.getName(),
                            name + "_weighted_gainchart.csv", SourceType.LOCAL);
                    LOG.info("Weighted gain chart data is generated in {} for eval {} and name {}.",
                            weightedGainChartCsv, evalConfig.getName(), name);
                    gc.generateCsv(evalConfig, modelConfig, weightedGainChartCsv, pr.weightedGains);
                }

                String prCsvFile = pathFinder.getEvalFilePath(evalConfig.getName(), name + "_unit_wise_pr.csv",
                        SourceType.LOCAL);
                LOG.info("Unit-wise pr data is generated in {} for eval {} and name {}.", prCsvFile,
                        evalConfig.getName(), name);
                gc.generateCsv(evalConfig, modelConfig, prCsvFile, pr.pr);

                if(hasWeight) {
                    String weightedPrCsvFile = pathFinder.getEvalFilePath(evalConfig.getName(),
                            name + "_weighted_pr.csv", SourceType.LOCAL);
                    LOG.info("Weighted pr data is generated in {} for eval {} and name {}.", weightedPrCsvFile,
                            evalConfig.getName(), name);
                    gc.generateCsv(evalConfig, modelConfig, weightedPrCsvFile, pr.weightedPr);
                }

                String rocCsvFile = pathFinder.getEvalFilePath(evalConfig.getName(), name + "_unit_wise_roc.csv",
                        SourceType.LOCAL);
                LOG.info("Unit-wise roc data is generated in {} for eval {} and name {}.", rocCsvFile,
                        evalConfig.getName(), name);
                gc.generateCsv(evalConfig, modelConfig, rocCsvFile, pr.roc);

                if(hasWeight) {
                    String weightedRocCsvFile = pathFinder.getEvalFilePath(evalConfig.getName(),
                            name + "_weighted_roc.csv", SourceType.LOCAL);
                    LOG.info("Weighted roc data is generated in {} for eval {} and name {}.", weightedRocCsvFile,
                            evalConfig.getName(), name);
                    gc.generateCsv(evalConfig, modelConfig, weightedRocCsvFile, pr.weightedRoc);
                }

                String modelScoreGainChartCsv = pathFinder.getEvalFilePath(evalConfig.getName(),
                        name + "_modelscore_gainchart.csv", SourceType.LOCAL);
                LOG.info("Model score gain chart data is generated in {} for eval {} and name {}.",
                        modelScoreGainChartCsv, evalConfig.getName(), name);
                gc.generateCsv(evalConfig, modelConfig, modelScoreGainChartCsv, pr.modelScoreList);
            }
            LOG.info("Performance Evaluation is done for {}.", evalConfig.getName());
        }
    }

    @SuppressWarnings("deprecation")
    private ScoreStatus runDistMetaScore(EvalConfig evalConfig, String metaScore, int mtlIndex) throws IOException {
        // TODO mtl index support
        SourceType sourceType = evalConfig.getDataSet().getSource();

        // clean up output directories
        ShifuFileUtils.deleteFile(pathFinder.getEvalMetaScorePath(evalConfig, metaScore), sourceType);

        // prepare special parameters and execute pig
        Map<String, String> paramsMap = new HashMap<String, String>();

        paramsMap.put(Constants.SOURCE_TYPE, sourceType.toString());
        paramsMap.put("pathEvalScoreData", pathFinder.getEvalScorePath(evalConfig));
        paramsMap.put("pathSortScoreData", pathFinder.getEvalMetaScorePath(evalConfig, metaScore));
        paramsMap.put("eval_set_name", evalConfig.getName());
        paramsMap.put("delimiter",
                Environment.getProperty(Constants.SHIFU_OUTPUT_DATA_DELIMITER, Constants.DEFAULT_DELIMITER));
        paramsMap.put("column_name", metaScore);

        String pigScript = "scripts/EvalScoreMetaSort.pig";
        Map<String, String> confMap = new HashMap<String, String>();
        // max min score folder
        Path path = new Path(
                "tmp" + File.separator + "maxmin_score_" + System.currentTimeMillis() + "_" + RANDOM.nextLong());
        String maxMinScoreFolder = ShifuFileUtils.getFileSystemBySourceType(sourceType, path).makeQualified(path)
                .toString();
        confMap.put(Constants.SHIFU_EVAL_MAXMIN_SCORE_OUTPUT, maxMinScoreFolder);
        confMap.put(Constants.SHIFU_NAMESPACE_STRICT_MODE, Boolean.TRUE.toString());
        confMap.put(Constants.SHIFU_OUTPUT_DATA_DELIMITER, Base64Utils.base64Encode(
                Environment.getProperty(Constants.SHIFU_OUTPUT_DATA_DELIMITER, Constants.DEFAULT_DELIMITER)));

        try {
            PigExecutor.getExecutor().submitJob(modelConfig, pathFinder.getScriptPath(pigScript), paramsMap,
                    evalConfig.getDataSet().getSource(), confMap, super.pathFinder);
        } catch (IOException e) {
            throw new ShifuException(ShifuErrorCode.ERROR_RUNNING_PIG_JOB, e);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }

        Iterator<JobStats> iter = PigStats.get().getJobGraph().iterator();

        while(iter.hasNext()) {
            JobStats jobStats = iter.next();
            long evalRecords = jobStats.getHadoopCounters().getGroup(Constants.SHIFU_GROUP_COUNTER)
                    .getCounter(Constants.COUNTER_RECORDS);
            LOG.info("Total valid eval records is : {}", evalRecords);
            // If no basic record counter, check next one
            if(evalRecords == 0L) {
                continue;
            }
            this.evalRecords = evalRecords;

            long pigPosTags = jobStats.getHadoopCounters().getGroup(Constants.SHIFU_GROUP_COUNTER)
                    .getCounter(Constants.COUNTER_POSTAGS);
            long pigNegTags = jobStats.getHadoopCounters().getGroup(Constants.SHIFU_GROUP_COUNTER)
                    .getCounter(Constants.COUNTER_NEGTAGS);
            double pigPosWeightTags = jobStats.getHadoopCounters().getGroup(Constants.SHIFU_GROUP_COUNTER)
                    .getCounter(Constants.COUNTER_WPOSTAGS) / (Constants.EVAL_COUNTER_WEIGHT_SCALE * 1.0d);
            double pigNegWeightTags = jobStats.getHadoopCounters().getGroup(Constants.SHIFU_GROUP_COUNTER)
                    .getCounter(Constants.COUNTER_WNEGTAGS) / (Constants.EVAL_COUNTER_WEIGHT_SCALE * 1.0d);

            double maxScore = Integer.MIN_VALUE;
            double minScore = Integer.MAX_VALUE;
            if(modelConfig.isRegression()) {
                double[] maxMinScores = locateMaxMinScoreFromFile(sourceType, maxMinScoreFolder);
                maxScore = maxMinScores[0];
                minScore = maxMinScores[1];
                LOG.info("Max score is {}, min score is {}", maxScore, minScore);
                ShifuFileUtils.deleteFile(maxMinScoreFolder, sourceType);
            }

            long badMetaScores = jobStats.getHadoopCounters().getGroup(Constants.SHIFU_GROUP_COUNTER)
                    .getCounter("BAD_META_SCORE");

            // Get score status from Counter to avoid re-computing such metrics
            LOG.info("Eval records is {}; and bad meta score is {}.", evalRecords, badMetaScores);

            return new ScoreStatus(pigPosTags, pigNegTags, pigPosWeightTags, pigNegWeightTags, maxScore, minScore,
                    evalRecords);
        }
        return null;
    }

    /**
     * Use akka to run model evaluation
     * 
     * @param evalConfig
     *            the evaluation instance
     * @throws IOException
     *             the error while create data scanner for input data
     */
    private void runAkkaEval(EvalConfig evalConfig) throws IOException {
        runAkkaScore(evalConfig);
        runConfusionMatrix(evalConfig);
        runPerformance(evalConfig);
    }

    /**
     * Running the performance matrices
     * 
     * @param evalSetList
     *            EvalConfig list
     * @throws IOException
     */
    private void runPerformance(List<EvalConfig> evalSetList) throws IOException {
        for(EvalConfig evalConfig: evalSetList) {
            runPerformance(evalConfig);
        }
    }

    /**
     * Running the performance matrices
     * 
     * @param evalConfig
     *            the name for evaluation
     * @throws IOException
     *             any io exception
     */
    private void runPerformance(EvalConfig evalConfig) throws IOException {
        PerformanceEvaluator perfEval = new PerformanceEvaluator(modelConfig, evalConfig);
        switch(modelConfig.getBasic().getRunMode()) {
            case DIST:
            case MAPRED:
                // FIXME here, this,evalRecords is 0 in initialzation
                perfEval.review(this.evalRecords);
                break;
            case LOCAL:
            default:
                perfEval.review();
                break;
        }
    }

    /**
     * Compute confusion matrix
     * 
     * @param evalSetList
     *            a List of EvalConfig
     * @throws IOException
     *             any io exception
     */
    private void runConfusionMatrix(List<EvalConfig> evalSetList) throws IOException {
        for(EvalConfig config: evalSetList) {
            runConfusionMatrix(config);
        }
    }

    /**
     * Compute confusion matrix
     * 
     * @param config
     *            eval config
     * @param ss
     *            the score stats
     * @return List of ConfusionMatrixObject
     * @throws IOException
     *             any io exception
     */
    private PerformanceResult runConfusionMatrix(EvalConfig config, ScoreStatus ss, boolean isUseMaxMinScore,
            boolean isMTL, int mtlIndex) throws IOException {
        return runConfusionMatrix(config, ss, pathFinder.getEvalScorePath(config),
                pathFinder.getEvalPerformancePath(config, config.getDataSet().getSource()), true, true,
                isUseMaxMinScore, isMTL, mtlIndex);
    }

    private PerformanceResult runConfusionMatrix(EvalConfig config, ScoreStatus ss, String scoreDataPath,
            String evalPerformancePath, boolean isPrint, boolean isGenerateChart, boolean isUseMaxMinScore,
            boolean isMTL, int mtlIndex) throws IOException {
        ConfusionMatrix worker = new ConfusionMatrix(modelConfig, columnConfigList, config, this);
        switch(modelConfig.getBasic().getRunMode()) {
            case DIST:
            case MAPRED:
                if(modelConfig.isRegression()) {
                    return worker.bufferedComputeConfusionMatrixAndPerformance(ss.pigPosTags, ss.pigNegTags,
                            ss.pigPosWeightTags, ss.pigNegWeightTags, ss.evalRecords, ss.maxScore, ss.minScore,
                            scoreDataPath, evalPerformancePath, isPrint, isGenerateChart, isUseMaxMinScore, isMTL,
                            mtlIndex);
                } else {
                    worker.computeConfusionMatixForMultipleClassification(this.evalRecords);
                    return null;
                }
            default:
                worker.computeConfusionMatrix();
                return null;
        }
    }

    private PerformanceResult runConfusionMatrix(EvalConfig config, ScoreStatus ss, String scoreDataPath,
            String evalPerformancePath, boolean isPrint, boolean isGenerateChart, int targetColumnIndex,
            int scoreColumnIndex, int weightColumnIndex, boolean isMultiTask, int mtlIndex) throws IOException {
        ConfusionMatrix worker = new ConfusionMatrix(modelConfig, columnConfigList, config, this);
        switch(modelConfig.getBasic().getRunMode()) {
            case DIST:
            case MAPRED:
                if(modelConfig.isRegression()) {
                    return worker.bufferedComputeConfusionMatrixAndPerformance(ss.pigPosTags, ss.pigNegTags,
                            ss.pigPosWeightTags, ss.pigNegWeightTags, ss.evalRecords, ss.maxScore, ss.minScore,
                            scoreDataPath, evalPerformancePath, isPrint, isGenerateChart, targetColumnIndex,
                            scoreColumnIndex, weightColumnIndex, true, isMultiTask, mtlIndex);
                } else {
                    worker.computeConfusionMatixForMultipleClassification(this.evalRecords);
                    return null;
                }
            case LOCAL:
            default:
                worker.computeConfusionMatrix();
                return null;
        }
    }

    private void runGenAudit(List<EvalConfig> evalSetList) throws IOException {
        this.params.put(NOSORT, Boolean.TRUE);
        if(CollectionUtils.isNotEmpty(evalSetList)) {
            for(EvalConfig evalConfig: evalSetList) {
                doGenAuditData(evalConfig);
            }
        }
    }

    private void doGenAuditData(EvalConfig evalConfig) throws IOException {
        // generate audit meta columns
        List<String> evalMetaColumns = evalConfig.getAllMetaColumns(this.modelConfig);
        final Set<String> metaColumnSet = new HashSet<>(evalMetaColumns);
        this.columnConfigList.stream().filter(columnConfig -> columnConfig.isFinalSelect())
                .map(columnConfig -> columnConfig.getColumnName()).forEach(finalVar -> {
                    if(!metaColumnSet.contains(finalVar)) {
                        evalMetaColumns.add(finalVar);
                    }
                });

        File columns = new File("columns");
        columns.mkdirs(); // create folder if it doesn't exist
        String newEvalMetaFile = "columns" + File.separator + evalConfig.getName() + ".audit.names";
        ShifuFileUtils.writeLines(evalMetaColumns, newEvalMetaFile, SourceType.LOCAL);

        String originalMetaFileName = evalConfig.getDataSet().getMetaColumnNameFile();
        String originalScoreFileName = evalConfig.getScoreMetaColumnNameFile();

        evalConfig.getDataSet().setMetaColumnNameFile(newEvalMetaFile);
        evalConfig.setScoreMetaColumnNameFile(null);
        saveModelConfig(); // update ModelConfig
        syncDataToHdfs(Arrays.asList(new EvalConfig[] { evalConfig }));
        runScore(evalConfig);

        // recover setting
        evalConfig.getDataSet().setMetaColumnNameFile(originalMetaFileName);
        evalConfig.setScoreMetaColumnNameFile(originalScoreFileName);
        saveModelConfig();

        int auditRecordsCount = getExpectAuditCount();

        File auditFile = new File("tmp", modelConfig.getModelSetName() + "_" + evalConfig.getName() + "_audit.data");
        BufferedWriter writer = null;
        try {
            writer = new BufferedWriter(new FileWriter(auditFile));
            if(ModelBasicConf.RunMode.LOCAL.equals(this.modelConfig.getBasic().getRunMode())) {
                String evalSorePah = this.pathFinder.getEvalScorePath(evalConfig);
                writeFileLines(writer, evalSorePah, evalConfig.getDataSet().getSource(), false, auditRecordsCount + 1);
            } else {
                String headerPath = this.pathFinder.getEvalScoreHeaderPath(evalConfig);
                writeFileLines(writer, headerPath, evalConfig.getDataSet().getSource(), false, 1);
                String evalSorePah = this.pathFinder.getEvalScorePath(evalConfig);
                writeFileLines(writer, evalSorePah, evalConfig.getDataSet().getSource(), true, auditRecordsCount);
            }

            LOG.info("Generate audit file {} successfully", auditFile.getCanonicalFile());
        } catch (IOException e) {
            LOG.error("Error occurred when generating audit file - {}", auditFile.getCanonicalPath());
        } finally {
            IOUtils.closeQuietly(writer);
        }
    }

    private void writeFileLines(BufferedWriter writer, String filePath, SourceType sourceType, boolean isPartFile,
            int linesCount) {
        BufferedReader reader = null;
        HdfsPartFile hdfsPartFile = null;
        int currentNumOfLine = 0;
        try {
            if(!isPartFile) {
                reader = ShifuFileUtils.getReader(filePath, sourceType);
                String line = null;
                while(currentNumOfLine++ < linesCount && (line = reader.readLine()) != null) {
                    writer.write(line);
                    writer.newLine();
                }
            } else {
                hdfsPartFile = new HdfsPartFile(filePath, sourceType);
                String line = null;
                while(currentNumOfLine++ < linesCount && (line = hdfsPartFile.readLine()) != null) {
                    writer.write(line);
                    writer.newLine();
                }
            }
        } catch (IOException e) {
            LOG.error("Fail to read data from {}.", filePath, e);
        } finally {
            IOUtils.closeQuietly(reader);
            if(hdfsPartFile != null) {
                hdfsPartFile.close();
            }
        }
    }

    /**
     * Run confusion matrix
     * 
     * @param config
     *            eval config
     * @return List of ConfusionMatrixObject
     * @throws IOException
     *             any io exception
     */
    private void runConfusionMatrix(EvalConfig config) throws IOException {
        runConfusionMatrix(config, null, false, false, -1);
    }

    /**
     * Check user set the expect audit count or not
     *
     * @return the expect audit records count, if user doesn't set, return default value - 10k
     */
    private int getExpectAuditCount() {
        return getIntParam(this.params, EXPECT_AUDIT_CNT, 10000);
    }

    /**
     * Check "-nosort" is specified or not
     * 
     * @return true if nosort is specified, or false
     */
    private boolean isNoSort() {
        return getBooleanParam(this.params, NOSORT);
    }

    /**
     * Add ref models as score column for performance comparision
     * 
     * @param scoreMetaColumns
     *            - the score columns to add into
     */
    private void addReferModelScoreColumns(List<String> scoreMetaColumns) {
        List<String> referModels = getRefModels();
        if(CollectionUtils.isNotEmpty(referModels)) {
            referModels.stream().forEach(referModel -> {
                File referModelFile = new File(referModel);
                scoreMetaColumns.add(genRefModelScoreName(referModelFile.getName()) + "::mean");
            });
        }
    }

    /**
     * Copy refer models as sub-models
     * 
     * @return
     *         true - if copy refer models successfully
     *         false - if some refer models doesn't exist
     * @throws IOException
     */
    private boolean copyRefModels() throws IOException {
        List<String> refModels = getRefModels();
        for(String refModel: refModels) {
            File refModelFile = new File(refModel);
            if(!refModelFile.exists()) {
                return false;
            }

            String refModelName = refModelFile.getName();
            File subModel = new File(Constants.MODELS, genRefModelScoreName(refModelName));
            subModel.mkdirs(); // create sub model in current project

            FileUtils.copyFile(new File(refModelFile, Constants.MODEL_CONFIG_JSON_FILE_NAME),
                    new File(subModel, Constants.MODEL_CONFIG_JSON_FILE_NAME));
            FileUtils.copyFile(new File(refModelFile, Constants.COLUMN_CONFIG_JSON_FILE_NAME),
                    new File(subModel, Constants.COLUMN_CONFIG_JSON_FILE_NAME));
            subModel.deleteOnExit();

            File modelsDir = new File(refModelFile, Constants.MODELS);
            if(modelsDir.exists()) {
                Arrays.stream(modelsDir.listFiles()).forEach(modelFile -> {
                    try {
                        FileUtils.copyFile(modelFile, new File(subModel, modelFile.getName()));
                    } catch (IOException e) {
                        LOG.error("Fail to copy file {}", modelFile.getAbsolutePath());
                    }
                });
            }

            Runtime.getRuntime().addShutdownHook(new Thread() {
                @Override
                public void run() {
                    try {
                        FileUtils.deleteDirectory(subModel);
                    } catch (IOException e) {
                        LOG.error("Fail to remove file {} after running.", subModel.getAbsolutePath());
                    }
                }
            });
        }
        return true;
    }

    /**
     * Check whether need to add refer models or not
     * 
     * @param evalStep
     *            - current step of Eval
     * @return
     *         true - if needs to copy refer models as sub models, else false
     */
    private boolean needsToCopyRefModels(EvalStep evalStep) {
        return CollectionUtils.isNotEmpty(getRefModels())
                && (EvalStep.RUN.equals(evalStep) || EvalStep.SCORE.equals(evalStep) || EvalStep.AUDIT.equals(evalStep)
                        || EvalStep.NORM.equals(evalStep));
    }

    /**
     * Add "ref_" as prefix for ref model name to avoid some models start with numbers
     * 
     * @param modelName
     *            - ref model name
     * @return "ref_" + modelName
     */
    private String genRefModelScoreName(String modelName) {
        return "ref_" + CommonUtils.normColumnName(StringUtils.trimToEmpty(modelName));
    }

    /**
     * Get the reference models to run eval step
     * 
     * @return reference models list
     */
    private List<String> getRefModels() {
        return getStringList(this.params, EvalModelProcessor.REF_MODEL, ",");
    }

    /**
     * Check "-strict" is specified or not. This is used when normalize the evaluation data set.
     * The Strict model - means output the data just as input, and append weight column only.
     * 
     * @return true if strict is specified, or false
     */
    private boolean isStrict() {
        return getBooleanParam(this.params, Constants.STRICT_MODE);
    }

    private static class ScoreStatus {

        public long pigPosTags = 0l;

        public long pigNegTags = 0l;

        public double pigPosWeightTags = 0d;

        public double pigNegWeightTags = 0d;

        public double maxScore = Integer.MIN_VALUE;

        public double minScore = Integer.MAX_VALUE;

        public long evalRecords = 0l;

        public ScoreStatus(long pigPosTags, long pigNegTags, double pigPosWeightTags, double pigNegWeightTags,
                double maxScore, double minScore, long evalRecords) {
            this.pigPosTags = pigPosTags;
            this.pigNegTags = pigNegTags;
            this.pigPosWeightTags = pigPosWeightTags;
            this.pigNegWeightTags = pigNegWeightTags;
            this.maxScore = maxScore;
            this.minScore = minScore;
            this.evalRecords = evalRecords;
        }
    }

}
