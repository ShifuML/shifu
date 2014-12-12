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

import ml.shifu.shifu.actor.AkkaSystemExecutor;
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
import ml.shifu.shifu.util.Constants;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.fs.Path;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * EvalModelProcessor class
 */
public class EvalModelProcessor extends BasicModelProcessor implements Processor {

    /**
     * Step for evaluation
     */
    public enum EvalStep {
        LIST, NEW, DELETE, RUN, PERF, SCORE, CONFMAT;
    }

    /**
     * log object
     */
    private final static Logger log = LoggerFactory.getLogger(EvalModelProcessor.class);

    private String evalName = null;

    private EvalStep evalStep;

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

        setUp(ModelStep.EVAL);

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

        clearUp(ModelStep.EVAL);
        return 0;
    }

    /**
     * @param evalName
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
     * @param evalName
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
        (new File(evalSetPath)).mkdirs();
        
        syncDataToHdfs(config.getDataSet().getSource());

        switch(modelConfig.getBasic().getRunMode()) {
            case mapred:
                runPigScore(config);
                break;
            case local:
                runAkkaScore(config);
                break;
            default:
                break;
        }
    }

    /**
     * run pig mode scoring
     * 
     * @param config
     * @throws IOException
     */
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
        paramsMap.put("pathHeader", evalConfig.getDataSet().getHeaderPath());
        paramsMap.put("pathEvalScore", pathFinder.getEvalScorePath(evalConfig));
        paramsMap.put("pathEvalPerformance", pathFinder.getEvalPerformancePath(evalConfig));
        paramsMap.put("eval_set_name", evalConfig.getName());
        paramsMap.put("delimiter", evalConfig.getDataSet().getDataDelimiter());

        try {
            PigExecutor.getExecutor().submitJob(modelConfig, pathFinder.getAbsolutePath("scripts/Eval.pig"), paramsMap,
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
    private void runEval(List<EvalConfig> evalSetList) throws IOException {
        for(EvalConfig evalConfig: evalSetList) {
            runEval(evalConfig);
        }
        log.info("Step Finished: eval");
    }

    /**
     * Run evaluation by @EvalConfig
     * 
     * @param evalConfig
     * @throws IOException
     */
    private void runEval(EvalConfig evalConfig) throws IOException {
        // create evalset home directory firstly in local file system
        PathFinder pathFinder = new PathFinder(modelConfig);
        String evalSetPath = pathFinder.getEvalSetPath(evalConfig, SourceType.LOCAL);
        (new File(evalSetPath)).mkdirs();

        syncDataToHdfs(evalConfig.getDataSet().getSource());

        switch(modelConfig.getBasic().getRunMode()) {
            case mapred:
                runPigEval(evalConfig);
                break;
            case local:
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
        runConfusionMatrix(evalConfig);
        runPerformance(evalConfig);
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
     * @param evalSetName
     *            the name for evaluation
     * @param scoreColumn
     *            the performance score target
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

        perfEval.review();
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
     * @param EvalConfig
     * @return List of ConfusionMatrixObject
     * @throws IOException
     */
    private void runConfusionMatrix(EvalConfig config) throws IOException {
        ConfusionMatrix worker = new ConfusionMatrix(modelConfig, config);
        worker.computeConfusionMatrix();
    }

}
