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
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;

import ml.shifu.shifu.actor.AkkaSystemExecutor;
import ml.shifu.shifu.column.NSColumn;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.container.obj.PerformanceResult;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.ConfusionMatrix;
import ml.shifu.shifu.core.PerformanceEvaluator;
import ml.shifu.shifu.core.Scorer;
import ml.shifu.shifu.core.validator.ModelInspector.ModelStep;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.fs.PathFinder;
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
    private final static Logger log = LoggerFactory.getLogger(EvalModelProcessor.class);

    /**
     * Step for evaluation
     */
    public enum EvalStep {
        LIST, NEW, DELETE, RUN, PERF, SCORE, CONFMAT, NORM, GAINCHART;
    }

    private String evalName = null;

    private EvalStep evalStep;

    private long evalRecords = 0l;

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
        log.info("Step Start: eval");
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
                    runPerformance(getEvalConfigListFromInput());
                    break;
                case SCORE:
                    runScore(getEvalConfigListFromInput());
                    break;
                case CONFMAT:
                    runConfusionMatrix(getEvalConfigListFromInput());
                    break;
                default:
                    break;
            }

            syncDataToHdfs(modelConfig.getDataSet().getSource());

            clearUp(ModelStep.EVAL);
        } catch (Exception e) {
            log.error("Error:", e);
            return -1;
        }
        log.info("Step Finished: eval with {} ms", (System.currentTimeMillis() - start));
        return 0;
    }

    private void deleteEvalSet(String evalSetName) {
        EvalConfig evalConfig = modelConfig.getEvalConfigByName(evalSetName);
        if(evalConfig == null) {
            log.error("{} eval set doesn't exist.", evalSetName);
        } else {
            modelConfig.getEvals().remove(evalConfig);
            try {
                saveModelConfig();
            } catch (IOException e) {
                throw new ShifuException(ShifuErrorCode.ERROR_WRITE_MODELCONFIG, e);
            }
            log.info("Done. Delete eval set - " + evalSetName);
        }
    }

    private void listEvalSet() {
        List<EvalConfig> evals = modelConfig.getEvals();
        if(CollectionUtils.isNotEmpty(evals)) {
            log.info("There are {} eval sets.", evals.size());
            for(EvalConfig evalConfig: evals) {
                log.info("\t - {}", evalConfig.getName());
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
                    log.error("The evalset - " + eval + " doesn't exist!");
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
     * @param evalConfigList
     *            eval config list
     * @throws IOException
     *             any io exception
     */
    private void runScore(List<EvalConfig> evalSetList) throws IOException {
        for(EvalConfig config: evalSetList) {
            runScore(config);
        }
    }

    /**
     * Run score only
     * 
     * @param evalConfig
     *            the eval config instance
     * @throws IOException
     *             any io exception
     */
    private void runScore(EvalConfig config) throws IOException {
        // create evalset home directory firstly in local file system
        PathFinder pathFinder = new PathFinder(modelConfig);
        String evalSetPath = pathFinder.getEvalSetPath(config, SourceType.LOCAL);
        FileUtils.forceMkdir(new File(evalSetPath));
        syncDataToHdfs(config.getDataSet().getSource());

        switch(modelConfig.getBasic().getRunMode()) {
            case DIST:
            case MAPRED:
                runPigScore(config);
                break;
            case LOCAL:
                runAkkaScore(config);
                break;
            default:
                break;
        }
    }

    /**
     * Run normalization against the evaluation data sets
     * 
     * @param evalConfigList
     *            eval config list
     * @throws IOException
     *             any io exception
     */
    private void runNormalize(List<EvalConfig> evalConfigList) throws IOException {
        for(EvalConfig evalConfig: evalConfigList) {
            runNormalize(evalConfig);
        }
    }

    /**
     * Run normalization against the evaluation data set
     * 
     * @param evalConfig
     *            the eval config instance
     * @throws IOException
     *             any io exception
     */
    private void runNormalize(EvalConfig evalConfig) throws IOException {
        PathFinder pathFinder = new PathFinder(modelConfig);
        String evalSetPath = pathFinder.getEvalSetPath(evalConfig, SourceType.LOCAL);
        FileUtils.forceMkdir(new File(evalSetPath));
        syncDataToHdfs(evalConfig.getDataSet().getSource());

        switch(modelConfig.getBasic().getRunMode()) {
            case DIST:
            case MAPRED:
                runPigNormalize(evalConfig);
                break;
            default:
                break;
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
    private ScoreStatus runPigScore(EvalConfig evalConfig) throws IOException {
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
        paramsMap.put("delimiter", evalConfig.getDataSet().getDataDelimiter());
        paramsMap.put("columnIndex", evalConfig.getPerformanceScoreSelector().trim());
        paramsMap.put("scale",
                Environment.getProperty(Constants.SHIFU_SCORE_SCALE, Integer.toString(Scorer.DEFAULT_SCORE_SCALE)));

        String pigScript = "scripts/Eval.pig";
        Map<String, String> confMap = new HashMap<String, String>();

        // max min score folder
        String maxMinScoreFolder = ShifuFileUtils
                .getFileSystemBySourceType(sourceType)
                .makeQualified(
                        new Path("tmp" + File.separator + "maxmin_score_" + System.currentTimeMillis() + "_"
                                + new Random().nextLong())).toString();
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
            log.info("Total valid eval records is : {}", evalRecords);
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

            log.info("Avg SLA for eval model scoring is {} micro seconds", totalRunTime / evalRecords);

            double maxScore = Integer.MIN_VALUE;
            double minScore = Integer.MAX_VALUE;
            if(modelConfig.isRegression()) {
                double[] maxMinScores = locateMaxMinScoreFromFile(sourceType, maxMinScoreFolder);
                maxScore = maxMinScores[0];
                minScore = maxMinScores[1];
                log.info("Max score is {}, min score is {}", maxScore, minScore);
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
     * Run pig code to normalize evaluation dataset
     * 
     * @param evalConfig
     *            the name for evaluation
     * @throws IOException
     *             any io exception
     */
    private void runPigNormalize(EvalConfig evalConfig) throws IOException {
        // clean up output directories
        SourceType sourceType = evalConfig.getDataSet().getSource();

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

        modelConfig.getEvals().add(evalConfig);

        try {
            saveModelConfig();
        } catch (IOException e) {
            throw new ShifuException(ShifuErrorCode.ERROR_WRITE_MODELCONFIG, e);
        }
        log.info("Create Eval - " + name);
    }

    /**
     * Running evaluation entry function
     * <p>
     * this function will switch to pig or akka evaluation depends on the modelConfig running mode
     * </p>
     * 
     * @throws IOException
     *             any exception in running pig evaluation or akka evaluation
     */
    private void runEval(List<EvalConfig> evalSetList) throws IOException {
        for(EvalConfig evalConfig: evalSetList) {
            runEval(evalConfig);
        }
    }

    private void validateEvalColumnConfig(EvalConfig evalConfig) throws IOException {
        if(this.columnConfigList == null) {
            return;
        }

        String[] evalColumnNames = null;

        if(StringUtils.isNotBlank(evalConfig.getDataSet().getHeaderPath())) {
            String delimiter = StringUtils.isBlank(evalConfig.getDataSet().getHeaderDelimiter()) ? evalConfig
                    .getDataSet().getDataDelimiter() : evalConfig.getDataSet().getHeaderDelimiter();
            evalColumnNames = CommonUtils.getHeaders(evalConfig.getDataSet().getHeaderPath(), delimiter, evalConfig
                    .getDataSet().getSource());
        } else {
            String delimiter = StringUtils.isBlank(evalConfig.getDataSet().getHeaderDelimiter()) ? evalConfig
                    .getDataSet().getDataDelimiter() : evalConfig.getDataSet().getHeaderDelimiter();
            String[] fields = CommonUtils.takeFirstLine(evalConfig.getDataSet().getDataPath(), delimiter, evalConfig
                    .getDataSet().getSource());
            // if first line contains target column name, we guess it is csv format and first line is header.
            if(StringUtils.join(fields, "").contains(modelConfig.getTargetColumnName())) {
                // first line of data meaning second line in data files excluding first header line
                String[] dataInFirstLine = CommonUtils.takeFirstTwoLines(evalConfig.getDataSet().getDataPath(),
                        delimiter, evalConfig.getDataSet().getSource())[1];
                if(dataInFirstLine != null && fields.length != dataInFirstLine.length) {
                    throw new IllegalArgumentException(
                            "Eval header length and eval data length are not consistent, please check you header setting and data set setting in eval.");
                }

                evalColumnNames = new String[fields.length];
                for(int i = 0; i < fields.length; i++) {
                    evalColumnNames[i] = CommonUtils.getRelativePigHeaderColumnName(fields[i]);
                }
                log.warn("No header path is provided, we will try to read first line and detect schema.");
                log.warn("Schema in ColumnConfig.json are named as first line of data set path.");
            } else {
                log.warn("No header path is provided, we will try to read first line and detect schema.");
                log.warn("Schema in ColumnConfig.json are named as  index 0, 1, 2, 3 ...");
                log.warn("Please make sure weight column and tag column are also taking index as name.");
                evalColumnNames = new String[fields.length];
                for(int i = 0; i < fields.length; i++) {
                    evalColumnNames[i] = i + "";
                }
            }
        }

        Set<NSColumn> names = new HashSet<NSColumn>();
        for ( String evalColumnName : evalColumnNames ) {
            names.add(new NSColumn(evalColumnName));
        }

        for(ColumnConfig config: this.columnConfigList) {
            if(config.isFinalSelect() && !names.contains(new NSColumn(config.getColumnName()))) {
                throw new IllegalArgumentException("Final selected column " + config.getColumnName()
                        + " does not exist in - " + evalConfig.getDataSet().getHeaderPath());
            }
        }

        if(StringUtils.isNotBlank(evalConfig.getDataSet().getTargetColumnName())
                && !names.contains(new NSColumn(evalConfig.getDataSet().getTargetColumnName()))) {
            throw new IllegalArgumentException("Target column " + evalConfig.getDataSet().getTargetColumnName()
                    + " does not exist in - " + evalConfig.getDataSet().getHeaderPath());
        }

        if(StringUtils.isNotBlank(evalConfig.getDataSet().getWeightColumnName())
                && !names.contains(new NSColumn(evalConfig.getDataSet().getWeightColumnName()))) {
            throw new IllegalArgumentException("Weight column " + evalConfig.getDataSet().getWeightColumnName()
                    + " does not exist in - " + evalConfig.getDataSet().getHeaderPath());
        }
    }

    /**
     * Run evaluation by @EvalConfig
     * 
     * @param evalConfig
     *            the name for evaluation
     * @throws IOException
     *             any io exception
     */
    private void runEval(EvalConfig evalConfig) throws IOException {
        // create evalset home directory firstly in local file system
        validateEvalColumnConfig(evalConfig);
        PathFinder pathFinder = new PathFinder(modelConfig);
        String evalSetPath = pathFinder.getEvalSetPath(evalConfig, SourceType.LOCAL);
        FileUtils.forceMkdir(new File(evalSetPath));
        syncDataToHdfs(evalConfig.getDataSet().getSource());

        switch(modelConfig.getBasic().getRunMode()) {
            case DIST:
            case MAPRED:
                runPigEval(evalConfig);
                break;
            case LOCAL:
                runAkkaEval(evalConfig);
                break;
            default:
                break;
        }
    }

    /**
     * Use pig to run model evaluation
     * 
     * @param evalConfig
     *            the evaluation instance
     * @throws IOException
     *             any exception in delete the old tmp files
     */
    private void runPigEval(EvalConfig evalConfig) throws IOException {
        ScoreStatus ss = runPigScore(evalConfig);
        runConfusionMatrix(evalConfig, ss);

        // ScoreStatus ss = runPigScore(evalConfig);
        //
        // List<String> scoreMetaColumns = evalConfig.getScoreMetaColumns(modelConfig);
        // if(scoreMetaColumns == null || scoreMetaColumns.isEmpty() || !modelConfig.isRegression()) {
        // runConfusionMatrix(evalConfig, ss);
        // return;
        // }
        //
        // PerformanceResult challendgeModelPerformance = runConfusionMatrix(evalConfig, ss,
        // pathFinder.getEvalScorePath(evalConfig), pathFinder.getEvalPerformancePath(evalConfig), false, false);
        // List<PerformanceResult> prList = new ArrayList<PerformanceResult>();
        // prList.add(challendgeModelPerformance);
        //
        // List<String> names = new ArrayList<String>();
        // names.add(modelConfig.getBasic().getName() + "::" + evalConfig.getName());
        // for(String metaScoreColumn: scoreMetaColumns) {
        // if(StringUtils.isBlank(metaScoreColumn)) {
        // continue;
        // }
        // names.add(metaScoreColumn);
        // log.info("Meta score performance for {} started.", metaScoreColumn);
        // ScoreStatus newScoreStatus = runPigMetaScore(evalConfig, metaScoreColumn);
        // PerformanceResult championModelPerformance = runConfusionMatrix(evalConfig, newScoreStatus,
        // pathFinder.getEvalScorePath(evalConfig, metaScoreColumn),
        // pathFinder.getEvalPerformancePath(evalConfig, metaScoreColumn), false, false, 0, 1, 2);
        // prList.add(championModelPerformance);
        // }
        // String htmlGainChart = pathFinder.getEvalFilePath(evalConfig.getName(), evalConfig.getName()
        // + "_gainchart.html", SourceType.LOCAL);
        // log.info("Gain chart is generated in {}.", htmlGainChart);
        // new GainChart().generateHtml(evalConfig, modelConfig, htmlGainChart, prList, names);
    }

    @SuppressWarnings({ "deprecation", "unused" })
    private ScoreStatus runPigMetaScore(EvalConfig evalConfig, String metaScore) throws IOException {
        // clean up output directories
        SourceType sourceType = evalConfig.getDataSet().getSource();

        ShifuFileUtils.deleteFile(pathFinder.getEvalScorePath(evalConfig, metaScore), sourceType);

        // prepare special parameters and execute pig
        Map<String, String> paramsMap = new HashMap<String, String>();

        paramsMap.put(Constants.SOURCE_TYPE, sourceType.toString());
        paramsMap.put("pathEvalRawData", evalConfig.getDataSet().getDataPath());
        paramsMap.put("pathSortScoreData", pathFinder.getEvalScorePath(evalConfig, metaScore));
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
                                + new Random().nextLong())).toString();
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
            log.info("Total valid eval records is : {}", evalRecords);
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
                log.info("Max score is {}, min score is {}", maxScore, minScore);
                ShifuFileUtils.deleteFile(maxMinScoreFolder, sourceType);
            }

            long badMetaScores = jobStats.getHadoopCounters().getGroup(Constants.SHIFU_GROUP_COUNTER)
                    .getCounter("BAD_META_SCORE");
            log.info("Eval records is {}; and bad meta score is {}.", evalRecords, badMetaScores);

            // only one pig job with such counters, return
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
                perfEval.review(this.evalRecords);
                break;
            case LOCAL:
            default:
                perfEval.review();
                break;
        }
    }

    /**
     * run confusion matrix
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
     * Run confusion matrix
     * 
     * @param config
     *            eval config
     * @param ss
     *            the score stats
     * @return List of ConfusionMatrixObject
     * @throws IOException
     *             any io exception
     */
    private PerformanceResult runConfusionMatrix(EvalConfig config, ScoreStatus ss) throws IOException {
        return runConfusionMatrix(config, ss, pathFinder.getEvalScorePath(config),
                pathFinder.getEvalPerformancePath(config, config.getDataSet().getSource()), true, true);
    }

    private PerformanceResult runConfusionMatrix(EvalConfig config, ScoreStatus ss, String scoreDataPath,
            String evalPerformancePath, boolean isPrint, boolean isGenerateChart) throws IOException {
        ConfusionMatrix worker = new ConfusionMatrix(modelConfig, config);
        switch(modelConfig.getBasic().getRunMode()) {
            case DIST:
            case MAPRED:
                if(modelConfig.isRegression()) {
                    return worker.bufferedComputeConfusionMatrixAndPerformance(ss.pigPosTags, ss.pigNegTags,
                            ss.pigPosWeightTags, ss.pigNegWeightTags, ss.evalRecords, ss.maxScore, ss.minScore,
                            scoreDataPath, evalPerformancePath, isPrint, isGenerateChart, false);
                } else {
                    worker.computeConfusionMatixForMultipleClassification(this.evalRecords);
                    return null;
                }
            default:
                worker.computeConfusionMatrix();
                return null;
        }
    }

    @SuppressWarnings("unused")
    private PerformanceResult runConfusionMatrix(EvalConfig config, ScoreStatus ss, String scoreDataPath,
            String evalPerformancePath, boolean isPrint, boolean isGenerateChart, int targetColumnIndex,
            int scoreColumnIndex, int weightColumnIndex) throws IOException {
        ConfusionMatrix worker = new ConfusionMatrix(modelConfig, config);
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
        runConfusionMatrix(config, null);
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
