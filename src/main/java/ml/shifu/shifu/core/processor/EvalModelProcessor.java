/**
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
import java.util.*;

import ml.shifu.shifu.actor.AkkaSystemExecutor;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.ConfusionMatrix;
import ml.shifu.shifu.core.PerformanceEvaluator;
import ml.shifu.shifu.core.validator.ModelInspector.ModelStep;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.fs.PathFinder;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.pig.PigExecutor;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;

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

    private long pigPosTags = 0l;

    private long pigNegTags = 0l;

    private double pigPosWeightTags = 0d;

    private double pigNegWeightTags = 0d;

    private int maxScore = 0;

    private int minScore = 0;

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
                case GAINCHART:
                    runGainChart(getEvalConfigListFromInput());
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

    /**
     * @param evalSetName
     */
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

    /**
     *
     */
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
     * @param evalSetList
     * @throws IOException
     */
    private void runScore(List<EvalConfig> evalSetList) throws IOException {
        for(EvalConfig config: evalSetList) {
            runScore(config);
        }
    }

    /**
     * run score only
     * 
     * @param config
     * @throws IOException
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
     * @throws IOException
     */
    @SuppressWarnings("deprecation")
    private void runPigScore(EvalConfig evalConfig) throws IOException {
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

        String pigScript = "scripts/Eval.pig";
        if(modelConfig.isClassification()) {
            pigScript = "scripts/EvalScore.pig";
        }
        try {
            PigExecutor.getExecutor().submitJob(modelConfig, pathFinder.getAbsolutePath(pigScript), paramsMap,
                    evalConfig.getDataSet().getSource());
        } catch (IOException e) {
            throw new ShifuException(ShifuErrorCode.ERROR_RUNNING_PIG_JOB, e);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }

        Iterator<JobStats> iter = PigStats.get().getJobGraph().iterator();

        while(iter.hasNext()) {
            JobStats jobStats = iter.next();
            this.evalRecords = jobStats.getHadoopCounters().getGroup(Constants.SHIFU_GROUP_COUNTER)
                    .getCounter(Constants.COUNTER_RECORDS);
            log.info("evalRecords:" + evalRecords);
            // If no basic record counter, check next one
            if(this.evalRecords == 0L) {
                continue;
            }
            this.pigPosTags = jobStats.getHadoopCounters().getGroup(Constants.SHIFU_GROUP_COUNTER)
                    .getCounter(Constants.COUNTER_POSTAGS);
            this.pigNegTags = jobStats.getHadoopCounters().getGroup(Constants.SHIFU_GROUP_COUNTER)
                    .getCounter(Constants.COUNTER_NEGTAGS);
            this.pigPosWeightTags = jobStats.getHadoopCounters().getGroup(Constants.SHIFU_GROUP_COUNTER)
                    .getCounter(Constants.COUNTER_WPOSTAGS)
                    / (Constants.EVAL_COUNTER_WEIGHT_SCALE * 1.0d);
            this.pigNegWeightTags = jobStats.getHadoopCounters().getGroup(Constants.SHIFU_GROUP_COUNTER)
                    .getCounter(Constants.COUNTER_WNEGTAGS)
                    / (Constants.EVAL_COUNTER_WEIGHT_SCALE * 1.0d);
            this.maxScore = (int) (jobStats.getHadoopCounters().getGroup(Constants.SHIFU_GROUP_COUNTER)
                    .getCounter(Constants.COUNTER_MAX_SCORE));
            this.minScore = (int) (jobStats.getHadoopCounters().getGroup(Constants.SHIFU_GROUP_COUNTER)
                    .getCounter(Constants.COUNTER_MIN_SCORE));
            // only one pig job with such counters, break
            break;
        }
    }

    /**
     * Run pig code to normalize evaluation dataset
     * 
     * @param evalConfig
     * @throws IOException
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

        String pigScript = "scripts/EvalNorm.pig";

        try {
            PigExecutor.getExecutor().submitJob(modelConfig, pathFinder.getAbsolutePath(pigScript), paramsMap,
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
     * @throws IOException
     */
    private void runAkkaScore(EvalConfig config) throws IOException {
        SourceType sourceType = config.getDataSet().getSource();
        List<Scanner> scanners = ShifuFileUtils.getDataScanners(
                ShifuFileUtils.expandPath(config.getDataSet().getDataPath(), sourceType), sourceType);

        AkkaSystemExecutor.getExecutor().submitModelEvalJob(modelConfig,
                ShifuFileUtils.searchColumnConfig(config, this.columnConfigList), config, scanners);

        // TODO A bug here in local mode
        // this.evalRecords = ...;
        closeScanners(scanners);
    }

    /**
     * Create a evaluation with <code>name</code>
     * 
     * @param name
     *            - the evaluation set name
     * @throws IOException
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
    private void runGainChart(List<EvalConfig> evalSetList) throws IOException {
        for(EvalConfig evalConfig: evalSetList) {
            runGainChart(evalConfig);
        }
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

    /**
     * Generate gain chart with highchart.js
     * 
     * @param evalConfig
     * @throws IOException
     */
    private void runGainChart(EvalConfig evalConfig) throws IOException {
        // create evalset home directory firstly in local file system
        validateEvalColumnConfig(evalConfig);
        PathFinder pathFinder = new PathFinder(modelConfig);
        String evalSetPath = pathFinder.getEvalSetPath(evalConfig, SourceType.LOCAL);
        FileUtils.forceMkdir(new File(evalSetPath));
        syncDataToHdfs(evalConfig.getDataSet().getSource());

        switch(modelConfig.getBasic().getRunMode()) {
            case DIST:
            case MAPRED:
                runPigScore(evalConfig);
                generateGainChart(evalConfig);
                break;
            case LOCAL:
            default:
                throw new RuntimeException("Local mode is not supported so far.");
        }
    }

    private void generateGainChart(EvalConfig evalConfig) {

    }

    private void validateEvalColumnConfig(EvalConfig evalConfig) throws IOException {
        if(this.columnConfigList == null) {
            return;
        }

        String[] evalColumnNames = CommonUtils.getHeaders(evalConfig.getDataSet().getHeaderPath(), evalConfig
                .getDataSet().getHeaderDelimiter(), evalConfig.getDataSet().getSource());
        Set<String> names = new HashSet<String>();
        names.addAll(Arrays.asList(evalColumnNames));

        for(ColumnConfig config: this.columnConfigList) {
            if(config.isFinalSelect() && !names.contains(config.getColumnName())) {
                throw new IllegalArgumentException("Final selected column " + config.getColumnName()
                        + " does not exist in - " + evalConfig.getDataSet().getHeaderPath());
            }
        }

        if(StringUtils.isNotBlank(evalConfig.getDataSet().getTargetColumnName())
                && !names.contains(evalConfig.getDataSet().getTargetColumnName())) {
            throw new IllegalArgumentException("Target column " + evalConfig.getDataSet().getTargetColumnName()
                    + " does not exist in - " + evalConfig.getDataSet().getHeaderPath());
        }

        if(StringUtils.isNotBlank(evalConfig.getDataSet().getWeightColumnName())
                && !names.contains(evalConfig.getDataSet().getWeightColumnName())) {
            throw new IllegalArgumentException("Weight column " + evalConfig.getDataSet().getWeightColumnName()
                    + " does not exist in - " + evalConfig.getDataSet().getHeaderPath());
        }
    }

    /**
     * Run evaluation by @EvalConfig
     * 
     * @param evalConfig
     * @throws IOException
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
        runPigScore(evalConfig);
        // TODO code refacter because of several magic numbers and not good name functions ...
        runConfusionMatrix(evalConfig);
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
     * @return List of ConfusionMatrixObject
     * @throws IOException
     */
    private void runConfusionMatrix(EvalConfig config) throws IOException {
        ConfusionMatrix worker = new ConfusionMatrix(modelConfig, config);
        switch(modelConfig.getBasic().getRunMode()) {
            case DIST:
            case MAPRED:
                if(modelConfig.isRegression()) {
                    worker.bufferedComputeConfusionMatrixAndPerformance(this.pigPosTags, this.pigNegTags,
                            this.pigPosWeightTags, this.pigNegWeightTags, this.evalRecords, this.maxScore,
                            this.minScore);
                } else {
                    worker.computeConfusionMatixForMultipleClassification(this.evalRecords);
                }
                break;
            default:
                worker.computeConfusionMatrix();
                break;
        }
    }
}
