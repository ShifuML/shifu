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

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;
import java.util.concurrent.CountDownLatch;

import ml.shifu.shifu.actor.AkkaSystemExecutor;
import ml.shifu.shifu.column.NSColumn;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.container.obj.PerformanceResult;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.ConfusionMatrix;
import ml.shifu.shifu.core.PerformanceEvaluator;
import ml.shifu.shifu.core.Scorer;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.dt.IndependentTreeModel;
import ml.shifu.shifu.core.eval.GainChart;
import ml.shifu.shifu.core.validator.ModelInspector.ModelStep;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.pig.PigExecutor;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.Environment;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.fs.Path;
import org.apache.pig.tools.pigstats.JobStats;
import org.apache.pig.tools.pigstats.PigStats;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
        LIST, NEW, DELETE, RUN, PERF, SCORE, CONFMAT, NORM, GAINCHART;
    }

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
     * Runner for evaluation
     */
    @Override
    public int run() throws Exception {
        LOG.info("Step Start: eval");
        long start = System.currentTimeMillis();
        try {
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
                case CONFMAT:
                    // FIXME, here should be failed
                    runConfusionMatrix(getEvalConfigListFromInput());
                    break;
                default:
                    break;
            }

            syncDataToHdfs(modelConfig.getDataSet().getSource());

            clearUp(ModelStep.EVAL);
        } catch (Exception e) {
            LOG.error("Error:", e);
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
                runDistScore(config);
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
    private ScoreStatus runDistScore(EvalConfig evalConfig) throws IOException {
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
        paramsMap.put("columnIndex", evalConfig.getPerformanceScoreSelector().trim());
        paramsMap.put("scale",
                Environment.getProperty(Constants.SHIFU_SCORE_SCALE, Integer.toString(Scorer.DEFAULT_SCORE_SCALE)));

        String expressionsAsString = super.modelConfig.getSegmentFilterExpressionsAsString();
        Environment.getProperties().put("shifu.segment.expressions", expressionsAsString);

        String pigScript = "scripts/Eval.pig";
        Map<String, String> confMap = new HashMap<String, String>();

        // max min score folder
        String maxMinScoreFolder = ShifuFileUtils
                .getFileSystemBySourceType(sourceType)
                .makeQualified(
                        new Path("tmp" + File.separator + "maxmin_score_" + System.currentTimeMillis() + "_"
                                + RANDOM.nextLong())).toString();
        confMap.put(Constants.SHIFU_EVAL_MAXMIN_SCORE_OUTPUT, maxMinScoreFolder);
        if(modelConfig.isClassification()) {
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

            long pigPosTags = jobStats.getHadoopCounters().getGroup(Constants.SHIFU_GROUP_COUNTER)
                    .getCounter(Constants.COUNTER_POSTAGS);
            long pigNegTags = jobStats.getHadoopCounters().getGroup(Constants.SHIFU_GROUP_COUNTER)
                    .getCounter(Constants.COUNTER_NEGTAGS);
            double pigPosWeightTags = jobStats.getHadoopCounters().getGroup(Constants.SHIFU_GROUP_COUNTER)
                    .getCounter(Constants.COUNTER_WPOSTAGS)
                    / (Constants.EVAL_COUNTER_WEIGHT_SCALE * 1.0d);
            double pigNegWeightTags = jobStats.getHadoopCounters().getGroup(Constants.SHIFU_GROUP_COUNTER)
                    .getCounter(Constants.COUNTER_WNEGTAGS)
                    / (Constants.EVAL_COUNTER_WEIGHT_SCALE * 1.0d);

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
                ShifuFileUtils.deleteFile(maxMinScoreFolder, sourceType);
            }
            // only one pig job with such counters, return
            return new ScoreStatus(pigPosTags, pigNegTags, pigPosWeightTags, pigNegWeightTags, maxScore, minScore,
                    evalRecords);
        }
        return null;
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
        List<Scanner> scanners = ShifuFileUtils.getDataScanners(
                ShifuFileUtils.expandPath(config.getDataSet().getDataPath(), sourceType), sourceType);

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
            throw new ShifuException(ShifuErrorCode.ERROR_MODEL_EVALSET_ALREADY_EXIST, "EvalSet - " + name
                    + " already exists in ModelConfig. Please use another evalset name");
        }

        evalConfig = new EvalConfig();
        evalConfig.setName(name);
        evalConfig.setDataSet(modelConfig.getDataSet().cloneRawSourceData());
        // create empty <EvalSetName>Score.meta.column.names
        ShifuFileUtils.createFileIfNotExists(new Path(evalConfig.getName()
                + Constants.DEFAULT_EVALSCORE_META_COLUMN_FILE).toString(), SourceType.LOCAL);

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
                LOG.warn("Starting from 0.10.x, 'scoreMetaColumns' is used for benchmark score columns and limited to at most 5.");
                LOG.warn("If meta columns are set in file of 'scoreMetaColumns', please move meta column config to 'eval#dataSet#metaColumnNameFile' part.");
                LOG.warn("If 'eval#dataSet#metaColumnNameFile' is duplicated with training 'metaColumnNameFile', you can rename it to another file with different name.");
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

    private void validateEvalColumnConfig(EvalConfig evalConfig) throws IOException {
        if(this.columnConfigList == null) {
            return;
        }

        String[] evalColumnNames = null;

        if(StringUtils.isNotBlank(evalConfig.getDataSet().getHeaderPath())) {
            String delimiter = StringUtils.isBlank(evalConfig.getDataSet().getHeaderDelimiter()) // get header delimiter
            ? evalConfig.getDataSet().getDataDelimiter()
                    : evalConfig.getDataSet().getHeaderDelimiter();
            evalColumnNames = CommonUtils.getHeaders(evalConfig.getDataSet().getHeaderPath(), delimiter, evalConfig
                    .getDataSet().getSource());
        } else {
            String delimiter = StringUtils.isBlank(evalConfig.getDataSet().getHeaderDelimiter()) // get header delimiter
            ? evalConfig.getDataSet().getDataDelimiter()
                    : evalConfig.getDataSet().getHeaderDelimiter();
            String[] fields = CommonUtils.takeFirstLine(evalConfig.getDataSet().getDataPath(), delimiter, evalConfig
                    .getDataSet().getSource());
            // if first line contains target column name, we guess it is csv format and first line is header.
            String evalTargetColumnName = ((StringUtils.isBlank(evalConfig.getDataSet().getTargetColumnName())) ? modelConfig
                    .getTargetColumnName() : evalConfig.getDataSet().getTargetColumnName());
            if(StringUtils.join(fields, "").contains(evalTargetColumnName)) {
                // first line of data meaning second line in data files excluding first header line
                String[] dataInFirstLine = CommonUtils.takeFirstTwoLines(evalConfig.getDataSet().getDataPath(),
                        delimiter, evalConfig.getDataSet().getSource())[1];
                if(dataInFirstLine != null && fields.length != dataInFirstLine.length) {
                    throw new IllegalArgumentException(
                            "Eval header length and eval data length are not consistent, please check you header setting and data set setting in eval.");
                }

                evalColumnNames = fields;
                // for(int i = 0; i < fields.length; i++) {
                // evalColumnNames[i] = CommonUtils.getRelativePigHeaderColumnName(fields[i]);
                // }
                LOG.warn("No header path is provided, we will try to read first line and detect schema.");
                LOG.warn("Schema in ColumnConfig.json are named as first line of data set path.");
            } else {
                LOG.warn("No header path is provided, we will try to read first line and detect schema.");
                LOG.warn("Schema in ColumnConfig.json are named as  index 0, 1, 2, 3 ...");
                LOG.warn("Please make sure weight column and tag column are also taking index as name.");
                evalColumnNames = new String[fields.length];
                for(int i = 0; i < fields.length; i++) {
                    evalColumnNames[i] = i + "";
                }
            }
        }

        Set<NSColumn> names = new HashSet<NSColumn>();
        for(String evalColumnName: evalColumnNames) {
            names.add(new NSColumn(evalColumnName));
        }

        String filterExpressions = super.modelConfig.getSegmentFilterExpressionsAsString();
        if(StringUtils.isNotBlank(filterExpressions)) {
            int segFilterSize = CommonUtils
                    .split(filterExpressions, Constants.SHIFU_STATS_FILTER_EXPRESSIONS_DELIMETER).length;
            for(int i = 0; i < segFilterSize; i++) {
                for(int j = 0; j < evalColumnNames.length; j++) {
                    names.add(new NSColumn(evalColumnNames[j] + "_" + (i + 1)));
                }
            }
        }

        for(ColumnConfig config: this.columnConfigList) {
            NSColumn nsColumn = new NSColumn(config.getColumnName());
            if(config.isFinalSelect() && !names.contains(nsColumn)
                    && !names.contains(new NSColumn(nsColumn.getSimpleName()))) {
                throw new IllegalArgumentException("Final selected column " + config.getColumnName()
                        + " does not exist in - " + evalConfig.getDataSet().getHeaderPath());
            }
        }

        NSColumn targetColumn = new NSColumn(evalConfig.getDataSet().getTargetColumnName());
        if(StringUtils.isNotBlank(evalConfig.getDataSet().getTargetColumnName()) && !names.contains(targetColumn)
                && !names.contains(new NSColumn(targetColumn.getSimpleName()))) {
            throw new IllegalArgumentException("Target column " + evalConfig.getDataSet().getTargetColumnName()
                    + " does not exist in - " + evalConfig.getDataSet().getHeaderPath());
        }

        NSColumn weightColumn = new NSColumn(evalConfig.getDataSet().getTargetColumnName());
        if(StringUtils.isNotBlank(evalConfig.getDataSet().getWeightColumnName()) && !names.contains(weightColumn)
                && !names.contains(new NSColumn(weightColumn.getSimpleName()))) {
            throw new IllegalArgumentException("Weight column " + evalConfig.getDataSet().getWeightColumnName()
                    + " does not exist in - " + evalConfig.getDataSet().getHeaderPath());
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
        ScoreStatus ss = runDistScore(evalConfig);

        List<String> scoreMetaColumns = evalConfig.getScoreMetaColumns(modelConfig);
        if(scoreMetaColumns == null || scoreMetaColumns.isEmpty() || !modelConfig.isRegression()) {
            // if no any champion score column set, go to previous evaluation with only challendge model
            runConfusionMatrix(evalConfig, ss, isGBTNotConvertToProb(evalConfig));
            return;
        }

        // 1. Get challendge model performance
        PerformanceResult challendgeModelPerformance = runConfusionMatrix(evalConfig, ss,
                pathFinder.getEvalScorePath(evalConfig), pathFinder.getEvalPerformancePath(evalConfig), false, false,
                isGBTNotConvertToProb(evalConfig));

        List<PerformanceResult> prList = new ArrayList<PerformanceResult>();
        prList.add(challendgeModelPerformance);

        // 2. Get all champion model performance
        List<String> names = new ArrayList<String>();
        names.add(modelConfig.getBasic().getName() + "-" + evalConfig.getName());
        for(String metaScoreColumn: scoreMetaColumns) {
            if(StringUtils.isBlank(metaScoreColumn)) {
                continue;
            }
            names.add(metaScoreColumn);

            LOG.info("Model score sort for {} in eval {} is started.", metaScoreColumn, evalConfig.getName());
            ScoreStatus newScoreStatus = runDistMetaScore(evalConfig, metaScoreColumn);

            PerformanceResult championModelPerformance = runConfusionMatrix(evalConfig, newScoreStatus,
                    pathFinder.getEvalMetaScorePath(evalConfig, metaScoreColumn),
                    pathFinder.getEvalMetaPerformancePath(evalConfig, metaScoreColumn), false, false, 0, 1, 2);
            prList.add(championModelPerformance);
        }

        synchronized(this) {
            GainChart gc = new GainChart();
            boolean hasWeight = StringUtils.isNotBlank(evalConfig.getDataSet().getWeightColumnName());

            // 3. Compute gain chart and other eval performance files only in local.
            String htmlGainChart = pathFinder.getEvalFilePath(evalConfig.getName(), evalConfig.getName()
                    + "_gainchart.html", SourceType.LOCAL);
            LOG.info("Gain chart is generated in {}.", htmlGainChart);
            gc.generateHtml(evalConfig, modelConfig, htmlGainChart, prList, names);

            String hrmlPrRoc = pathFinder.getEvalFilePath(evalConfig.getName(), evalConfig.getName() + "_prroc.html",
                    SourceType.LOCAL);
            LOG.info("PR & ROC chart is generated in {}.", hrmlPrRoc);
            gc.generateHtml4PrAndRoc(evalConfig, modelConfig, hrmlPrRoc, prList, names);

            for(int i = 0; i < names.size(); i++) {
                String name = names.get(i);
                PerformanceResult pr = prList.get(i);
                String unitGainChartCsv = pathFinder.getEvalFilePath(evalConfig.getName(), name
                        + "_unit_wise_gainchart.csv", SourceType.LOCAL);
                LOG.info("Unit-wise gain chart data is generated in {} for eval {} and name {}.", unitGainChartCsv,
                        evalConfig.getName(), name);
                gc.generateCsv(evalConfig, modelConfig, unitGainChartCsv, pr.gains);
                if(hasWeight) {
                    String weightedGainChartCsv = pathFinder.getEvalFilePath(evalConfig.getName(), name
                            + "_weighted_gainchart.csv", SourceType.LOCAL);
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
                    String weightedPrCsvFile = pathFinder.getEvalFilePath(evalConfig.getName(), name
                            + "_weighted_pr.csv", SourceType.LOCAL);
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
                    String weightedRocCsvFile = pathFinder.getEvalFilePath(evalConfig.getName(), name
                            + "_weighted_roc.csv", SourceType.LOCAL);
                    LOG.info("Weighted roc data is generated in {} for eval {} and name {}.", weightedRocCsvFile,
                            evalConfig.getName(), name);
                    gc.generateCsv(evalConfig, modelConfig, weightedRocCsvFile, pr.weightedRoc);
                }

                String modelScoreGainChartCsv = pathFinder.getEvalFilePath(evalConfig.getName(), name
                        + "_modelscore_gainchart.csv", SourceType.LOCAL);
                LOG.info("Model score gain chart data is generated in {} for eval {} and name {}.",
                        modelScoreGainChartCsv, evalConfig.getName(), name);
                gc.generateCsv(evalConfig, modelConfig, modelScoreGainChartCsv, pr.modelScoreList);
            }
            LOG.info("Performance Evaluation is done for {}.", evalConfig.getName());
        }
    }

    @SuppressWarnings("deprecation")
    private ScoreStatus runDistMetaScore(EvalConfig evalConfig, String metaScore) throws IOException {
        SourceType sourceType = evalConfig.getDataSet().getSource();

        // clean up output directories
        ShifuFileUtils.deleteFile(pathFinder.getEvalMetaScorePath(evalConfig, metaScore), sourceType);

        // prepare special parameters and execute pig
        Map<String, String> paramsMap = new HashMap<String, String>();

        paramsMap.put(Constants.SOURCE_TYPE, sourceType.toString());
        paramsMap.put("pathEvalRawData", evalConfig.getDataSet().getDataPath());
        paramsMap.put("pathSortScoreData", pathFinder.getEvalMetaScorePath(evalConfig, metaScore));
        paramsMap.put("eval_set_name", evalConfig.getName());
        paramsMap.put("delimiter", evalConfig.getDataSet().getDataDelimiter());
        paramsMap.put("column_name", metaScore);

        String pigScript = "scripts/EvalScoreMetaSort.pig";
        Map<String, String> confMap = new HashMap<String, String>();
        // max min score folder
        String maxMinScoreFolder = ShifuFileUtils
                .getFileSystemBySourceType(sourceType)
                .makeQualified(
                        new Path("tmp" + File.separator + "maxmin_score_" + System.currentTimeMillis() + "_"
                                + RANDOM.nextLong())).toString();
        confMap.put(Constants.SHIFU_EVAL_MAXMIN_SCORE_OUTPUT, maxMinScoreFolder);

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
                    .getCounter(Constants.COUNTER_WPOSTAGS)
                    / (Constants.EVAL_COUNTER_WEIGHT_SCALE * 1.0d);
            double pigNegWeightTags = jobStats.getHadoopCounters().getGroup(Constants.SHIFU_GROUP_COUNTER)
                    .getCounter(Constants.COUNTER_WNEGTAGS)
                    / (Constants.EVAL_COUNTER_WEIGHT_SCALE * 1.0d);

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
    private PerformanceResult runConfusionMatrix(EvalConfig config, ScoreStatus ss, boolean isUseMaxMinScore)
            throws IOException {
        return runConfusionMatrix(config, ss, pathFinder.getEvalScorePath(config),
                pathFinder.getEvalPerformancePath(config, config.getDataSet().getSource()), true, true,
                isUseMaxMinScore);
    }

    private PerformanceResult runConfusionMatrix(EvalConfig config, ScoreStatus ss, String scoreDataPath,
            String evalPerformancePath, boolean isPrint, boolean isGenerateChart, boolean isUseMaxMinScore)
            throws IOException {
        ConfusionMatrix worker = new ConfusionMatrix(modelConfig, config, this);
        switch(modelConfig.getBasic().getRunMode()) {
            case DIST:
            case MAPRED:
                if(modelConfig.isRegression()) {
                    return worker.bufferedComputeConfusionMatrixAndPerformance(ss.pigPosTags, ss.pigNegTags,
                            ss.pigPosWeightTags, ss.pigNegWeightTags, ss.evalRecords, ss.maxScore, ss.minScore,
                            scoreDataPath, evalPerformancePath, isPrint, isGenerateChart, isUseMaxMinScore);
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
            int scoreColumnIndex, int weightColumnIndex) throws IOException {
        ConfusionMatrix worker = new ConfusionMatrix(modelConfig, config, this);
        switch(modelConfig.getBasic().getRunMode()) {
            case DIST:
            case MAPRED:
                if(modelConfig.isRegression()) {
                    return worker.bufferedComputeConfusionMatrixAndPerformance(ss.pigPosTags, ss.pigNegTags,
                            ss.pigPosWeightTags, ss.pigNegWeightTags, ss.evalRecords, ss.maxScore, ss.minScore,
                            scoreDataPath, evalPerformancePath, isPrint, isGenerateChart, targetColumnIndex,
                            scoreColumnIndex, weightColumnIndex, true);
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
        runConfusionMatrix(config, null, false);
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
