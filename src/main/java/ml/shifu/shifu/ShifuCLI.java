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
package ml.shifu.shifu;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.*;
import java.util.jar.JarFile;
import java.util.jar.Manifest;

import ml.shifu.shifu.core.TreeModel;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.util.*;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.commons.lang3.tuple.MutablePair;
import org.apache.pig.impl.util.JarManager;
import org.encog.ml.BasicML;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ml.shifu.shifu.container.obj.ModelTrainConf.ALGORITHM;
import ml.shifu.shifu.core.processor.BasicModelProcessor;
import ml.shifu.shifu.core.processor.ComboModelProcessor;
import ml.shifu.shifu.core.processor.CreateModelProcessor;
import ml.shifu.shifu.core.processor.EvalModelProcessor;
import ml.shifu.shifu.core.processor.EvalModelProcessor.EvalStep;
import ml.shifu.shifu.core.processor.ExportModelProcessor;
import ml.shifu.shifu.core.processor.InitModelProcessor;
import ml.shifu.shifu.core.processor.ManageModelProcessor;
import ml.shifu.shifu.core.processor.ManageModelProcessor.ModelAction;
import ml.shifu.shifu.core.processor.ModelDataEncodeProcessor;
import ml.shifu.shifu.core.processor.NormalizeModelProcessor;
import ml.shifu.shifu.core.processor.PostTrainModelProcessor;
import ml.shifu.shifu.core.processor.Processor;
import ml.shifu.shifu.core.processor.ShifuTestProcessor;
import ml.shifu.shifu.core.processor.StatsModelProcessor;
import ml.shifu.shifu.core.processor.TrainModelProcessor;
import ml.shifu.shifu.core.processor.VarSelectModelProcessor;
import ml.shifu.shifu.exception.ShifuException;

/**
 * ShifuCLI class is the MAIN class for whole project
 * It will read and analysis the parameters from command line
 * and execute corresponding functions
 */
public class ShifuCLI {

    private static final String MODELSET_CMD_M = "m";
    private static final String EVAL_CMD_RUN = "run";
    private static final String EVAL_CMD = "eval";
    private static final String POSTTRAIN_CMD = "posttrain";
    private static final String TRAIN_CMD = "train";
    private static final String VARSELECT_CMD = "varselect";
    private static final String VARSEL_CMD = "varsel";
    private static final String NORMALIZE_CMD = "normalize";
    private static final String NORM_CMD = "norm";
    private static final String TRANSFORM_CMD = "transform";
    private static final String STATS_CMD = "stats";
    private static final String INIT_CMD_MODEL = "model";
    private static final String INIT_CMD = "init";
    private static final String MODELSET_CMD_CP = "cp";
    private static final String MODELSET_CMD_NEW = "new";
    private static final String MODELSET_CMD_TYPE = "t";
    private static final String EXPORT_CONCISE = "c";
    private static final String NEW = "new";

    private static final String CMD_EXPORT = "export";
    private static final String CMD_COMBO = "combo";
    private static final String CMD_ENCODE = "encode";
    private static final String CMD_TEST = "test";
    private static final String CMD_CONVERT = "convert";
    private static final String CMD_ANALYSIS = "analysis";

    // options for stats
    private static final String CORRELATION = "correlation";
    private static final String SHORT_CORRELATION = "c";
    private static final String PSI = "psi";
    private static final String UPDATE_STATS_ONLY = "updatestatsonly";
    private static final String SHORT_UPDATE_STATS_ONLY = "u";

    private static final String SHORT_PSI = "p";
    // options for variable re-binning
    private static final String REBIN = "rebin";
    private static final String VARS = "vars";
    private static final String N = "n";
    private static final String IVR = "ivr";
    private static final String BIC = "bic";

    // options for variable select
    private static final String RESET = "reset";
    private static final String FILTER_AUTO = "autofilter";
    private static final String RECOVER_AUTO = "recoverauto";
    private static final String RECURSIVE = "r";
    private static final String VAR_SEL_FILE = "f";

    // for evaluation
    private static final String LIST = "list";
    private static final String DELETE = "delete";
    private static final String SCORE = "score";
    private static final String CONFMAT = "confmat";
    private static final String PERF = "perf";
    private static final String NORM = "norm";
    private static final String NOSORT = "nosort";
    private static final String REF = "ref";
    private static final String STRICT = "strict";
    private static final String AUDIT = "audit";

    private static final String SAVE = "save";
    private static final String SWITCH = "switch";
    private static final String EVAL_MODEL = "model";
    private static final String SHOW = "show";

    private static final String SHUFFLE = "shuffle";
    private static final String REBALANCE = "rebalance";
    private static final String UPDATE_WEIGHT = "updateweight";
    private static final String RESUME = "resume";

    // for test function
    private static final String FILTER = "filter";
    // for model spec convert
    private static final String TO_ZIPB = "tozipb";
    private static final String TO_TREEB = "totreeb";
    // for model spec analysis
    private static final String FI = "fi";
    // for model name
    private static final String NAME = "name";

    static private final Logger log = LoggerFactory.getLogger(ShifuCLI.class);

    public static void main(String[] args) {
        String[] cleanedArgs = cleanArgs(args);
        // invalid input and help options
        if(cleanedArgs.length < 1 || (isHelpOption(cleanedArgs[0]))) {
            printUsage();
            System.exit(cleanedArgs.length < 1 ? -1 : 0);
        }

        // process -v and -version conditions manually
        if(isVersionOption(cleanedArgs[0])) {
            printLogoAndVersion();
            System.exit(0);
        }

        CommandLineParser parser = new GnuParser();
        Options opts = buildModelSetOptions();
        CommandLine cmd = null;

        try {
            cmd = parser.parse(opts, cleanedArgs);
        } catch (ParseException e) {
            log.error("Invalid command options. Please check help message.");
            printUsage();
            System.exit(1);
        }

        int status = 0;

        try {
            if(cleanedArgs[0].equals(NEW) && cleanedArgs.length >= 2 && StringUtils.isNotEmpty(cleanedArgs[1])) {
                // modelset step
                String modelName = cleanedArgs[1];
                status = createNewModel(modelName, cmd.getOptionValue(MODELSET_CMD_TYPE),
                        cmd.getOptionValue(MODELSET_CMD_M));
                if(status == 0) {
                    printModelSetCreatedSuccessfulLog(modelName);
                } else {
                    log.warn("Error in create new model set, please check your shifu config or report issue");
                }
                System.exit(status);
                // copyModel(manager, cmd.getOptionValues(MODELSET_CMD_CP));
            } else {
                if(cleanedArgs[0].equals(MODELSET_CMD_CP) && cleanedArgs.length >= 3
                        && StringUtils.isNotEmpty(cleanedArgs[1]) && StringUtils.isNotEmpty(cleanedArgs[2])) {
                    String newModelSetName = cleanedArgs[2];
                    // modelset step
                    copyModel(new String[] { cleanedArgs[1], newModelSetName });
                    printModelSetCopiedSuccessfulLog(newModelSetName);
                } else if(cleanedArgs[0].equals(INIT_CMD)) {
                    // init step
                    if(cmd.getOptions() == null || cmd.getOptions().length == 0) {
                        status = initializeModel();
                        if(status == 0) {
                            log.info(
                                    "ModelSet initialization is successful. Please continue next step by using 'shifu stats'.");
                        } else {
                            log.warn(
                                    "Error in ModelSet initialization, please check your shifu config or report issue");
                        }
                    } else if(cmd.hasOption(INIT_CMD_MODEL)) {
                        initializeModelParam();
                    } else {
                        log.error("Invalid command, please check help message.");
                        printUsage();
                    }
                } else if(cleanedArgs[0].equals(STATS_CMD)) {
                    Map<String, Object> params = new HashMap<String, Object>();
                    params.put(Constants.IS_COMPUTE_CORR,
                            cmd.hasOption(CORRELATION) || cmd.hasOption(SHORT_CORRELATION));
                    params.put(Constants.IS_REBIN, cmd.hasOption(REBIN));
                    params.put(Constants.REQUEST_VARS, cmd.getOptionValue(VARS));
                    params.put(Constants.EXPECTED_BIN_NUM, cmd.getOptionValue(N));
                    params.put(Constants.IV_KEEP_RATIO, cmd.getOptionValue(IVR));
                    params.put(Constants.MINIMUM_BIN_INST_CNT, cmd.getOptionValue(BIC));
                    params.put(Constants.IS_COMPUTE_PSI, cmd.hasOption(PSI) || cmd.hasOption(SHORT_PSI));
                    params.put(Constants.IS_UPDATE_STATS_ONLY,
                            cmd.hasOption(UPDATE_STATS_ONLY) || cmd.hasOption(SHORT_UPDATE_STATS_ONLY));

                    // stats step
                    status = calModelStats(params);
                    if(status == 0) {
                        if(cmd.hasOption(CORRELATION) || cmd.hasOption(SHORT_CORRELATION)) {
                            log.info(
                                    "Do model set correlation computing successfully. Please continue next step by using 'shifu normalize or shifu norm'. For tree ensemble model, no need do norm, please continue next step by using 'shifu varsel'");
                        }
                        if(cmd.hasOption(PSI) || cmd.hasOption(SHORT_PSI)) {
                            log.info(
                                    "Do model set psi computing successfully. Please continue next step by using 'shifu normalize or shifu norm'. For tree ensemble model, no need do norm, please continue next step by using 'shifu varsel'");
                        } else {
                            log.info(
                                    "Do model set statistic successfully. Please continue next step by using 'shifu normalize or shifu norm or shifu transform'. For tree ensemble model, no need do norm, please continue next step by using 'shifu varsel'");
                        }
                    } else {
                        log.warn(
                                "Error in model set stats computation, please report issue on http:/github.com/shifuml/shifu/issues.");
                    }
                } else if(cleanedArgs[0].equals(NORMALIZE_CMD) || cleanedArgs[0].equals(NORM_CMD)
                        || cleanedArgs[0].equals(TRANSFORM_CMD)) {
                    // normalize step
                    Map<String, Object> params = new HashMap<String, Object>();
                    params.put(Constants.IS_TO_SHUFFLE_DATA, cmd.hasOption(SHUFFLE));
                    params.put(Constants.EXPECT_POS_RATIO, cmd.getOptionValue(REBALANCE));
                    params.put(Constants.RBL_UPDATE_WEIGHT, cmd.hasOption(UPDATE_WEIGHT));
                    status = normalizeTrainData(params);
                    if(status == 0) {
                        log.info(
                                "Do model set normalization successfully. Please continue next step by using 'shifu varselect or shifu varsel'.");
                    } else {
                        log.warn(
                                "Error in model set stats computation, please report issue on http:/github.com/shifuml/shifu/issues.");
                    }
                } else if(cleanedArgs[0].equals(VARSELECT_CMD) || cleanedArgs[0].equals(VARSEL_CMD)) {
                    Map<String, Object> params = new HashMap<String, Object>();
                    params.put(Constants.IS_TO_RESET, cmd.hasOption(RESET));
                    params.put(Constants.IS_TO_LIST, cmd.hasOption(LIST));
                    params.put(Constants.IS_TO_FILTER_AUTO, cmd.hasOption(FILTER_AUTO));
                    params.put(Constants.IS_TO_RECOVER_AUTO, cmd.hasOption(RECOVER_AUTO));
                    params.put(Constants.RECURSIVE_CNT, cmd.getOptionValue(RECURSIVE));
                    params.put(Constants.VAR_SEL_FILE, cmd.getOptionValue(VAR_SEL_FILE));

                    // variable selected step
                    status = selectModelVar(params);
                    if(status == 0) {
                        log.info(
                                "Do model set variables selection successfully. Please continue next step by using 'shifu train'.");
                    } else {
                        log.info("Do variable selection with error, please check error message or report issue.");
                    }
                } else if(cleanedArgs[0].equals(TRAIN_CMD)) {
                    // train step
                    status = trainModel(cmd.hasOption(SHUFFLE));
                    if(status == 0) {
                        log.info(
                                "Do model set training successfully. Please continue next step by using 'shifu posttrain' or if no need posttrain you can go through with 'shifu eval'.");
                    } else {
                        log.info("Do model training with error, please check error message or report issue.");
                    }
                } else if(cleanedArgs[0].equals(CMD_ENCODE)) {
                    Map<String, Object> params = new HashMap<String, Object>();
                    params.put(ModelDataEncodeProcessor.ENCODE_DATA_SET, cmd.getOptionValue(EVAL_CMD_RUN));
                    params.put(ModelDataEncodeProcessor.ENCODE_REF_MODEL, cmd.getOptionValue(REF));
                    status = runEncode(params);
                } else if(cleanedArgs[0].equals(CMD_COMBO)) {
                    if(cmd.hasOption(MODELSET_CMD_NEW)) {
                        log.info("Create new commbo models");
                        status = createNewCombo(cmd.getOptionValue(MODELSET_CMD_NEW));
                    } else if(cmd.hasOption(INIT_CMD)) {
                        log.info("Init commbo models");
                        status = initComboModels();
                    } else if(cmd.hasOption(EVAL_CMD_RUN)) {
                        log.info("Run combo model - with toShuffle: {}, with toResume: {}", opts.hasOption(SHUFFLE),
                                opts.hasOption(RESUME));
                        status = runComboModels(cmd.hasOption(SHUFFLE), cmd.hasOption(RESUME));
                        // train combo models
                    } else if(cmd.hasOption(EVAL_CMD)) {
                        log.info("Eval combo model.");
                        // eval combo model performance
                        status = evalComboModels(cmd.hasOption(RESUME));
                    } else {
                        log.error("Invalid command usage.");
                        printUsage();
                    }
                } else if(cleanedArgs[0].equals(POSTTRAIN_CMD)) {
                    // post train step
                    status = postTrainModel();
                    if(status == 0) {
                        log.info(
                                "Do model set post-training successfully. Please configure your eval set in ModelConfig.json and continue next step by using 'shifu eval' or 'shifu eval -new <eval set>' to create a new eval set.");
                    } else {
                        log.info("Do model post training with error, please check error message or report issue.");
                    }
                } else if(cleanedArgs[0].equals(SAVE)) {
                    String newModelSetName = cleanedArgs.length >= 2 ? cleanedArgs[1] : null;
                    saveCurrentModel(newModelSetName);
                } else if(cleanedArgs[0].equals(SWITCH)) {
                    String newModelSetName = cleanedArgs[1];
                    switchCurrentModel(newModelSetName);
                } else if(cleanedArgs[0].equals(SHOW)) {
                    ManageModelProcessor p = new ManageModelProcessor(ModelAction.SHOW, null);
                    p.run();
                } else if(cleanedArgs[0].equals(EVAL_CMD)) {
                    Map<String, Object> params = new HashMap<String, Object>();
                    params.put(EvalModelProcessor.REF_MODEL, cmd.getOptionValue(REF));

                    // eval step
                    if(cleanedArgs.length == 1) {
                        // run everything
                        status = runEvalSet(params);
                        if(status == 0) {
                            log.info("Run eval performance with all eval sets successfully.");
                        } else {
                            log.info("Do evaluation with error, please check error message or report issue.");
                        }
                    } else if(cmd.getOptionValue(MODELSET_CMD_NEW) != null) {
                        // create new eval
                        createNewEvalSet(cmd.getOptionValue(MODELSET_CMD_NEW));
                        log.info(
                                "Create eval set successfully. You can configure EvalConfig.json or directly run 'shifu eval -run <evalSetName>' to get performance info.");
                    } else if(cmd.hasOption(EVAL_CMD_RUN)) {
                        runEvalSet(cmd.getOptionValue(EVAL_CMD_RUN), params);
                        log.info("Finish run eval performance with eval set {}.", cmd.getOptionValue(EVAL_CMD_RUN));
                    } else if(cmd.hasOption(SCORE)) {
                        params.put(EvalModelProcessor.NOSORT, cmd.hasOption(NOSORT));
                        // run score
                        runEvalScore(cmd.getOptionValue(SCORE), params);
                        log.info("Finish run score with eval set {}.", cmd.getOptionValue(SCORE));
                    } else if(cmd.hasOption(CONFMAT)) {
                        // run confusion matrix
                        runEvalConfMat(cmd.getOptionValue(CONFMAT));
                        log.info("Finish run confusion matrix with eval set {}.", cmd.getOptionValue(CONFMAT));
                    } else if(cmd.hasOption(PERF)) {
                        // run perfermance
                        runEvalPerf(cmd.getOptionValue(PERF));
                        log.info("Finish run performance maxtrix with eval set {}.", cmd.getOptionValue(PERF));
                    } else if(cmd.hasOption(LIST)) {
                        // list all evaluation sets
                        listEvalSet();
                    } else if(cmd.hasOption(DELETE)) {
                        // delete some evaluation set
                        deleteEvalSet(cmd.getOptionValue(DELETE));
                    } else if(cmd.hasOption(NORM)) {
                        params.put(Constants.STRICT_MODE, cmd.hasOption(STRICT));
                        runEvalNorm(cmd.getOptionValue(NORM), params);
                    } else if (cmd.hasOption(AUDIT)) {
                        params.put(EvalModelProcessor.EXPECT_AUDIT_CNT, cmd.getOptionValue(N));
                        runAuditEval(cmd.getOptionValue(AUDIT), params);
                    } else {
                        log.error("Invalid command, please check help message.");
                        printUsage();
                    }
                } else if(cleanedArgs[0].equals(CMD_EXPORT)) {
                    Map<String, Object> params = new HashMap<String, Object>();
                    params.put(ExportModelProcessor.IS_CONCISE, cmd.hasOption(EXPORT_CONCISE));
                    params.put(ExportModelProcessor.REQUEST_VARS, cmd.getOptionValue(VARS));
                    params.put(ExportModelProcessor.EXPECTED_BIN_NUM, cmd.getOptionValue(N));
                    params.put(ExportModelProcessor.IV_KEEP_RATIO, cmd.getOptionValue(IVR));
                    params.put(ExportModelProcessor.MINIMUM_BIN_INST_CNT, cmd.getOptionValue(BIC));
                    params.put(ExportModelProcessor.EXPORT_MODEL_NAME, cmd.getOptionValue(NAME));
                    status = exportModel(cmd.getOptionValue(MODELSET_CMD_TYPE), params);
                    if(status == 0) {
                        log.info("Export models/columnstats/corr successfully.");
                    } else {
                        log.warn("Fail to export models/columnstats/corr, please check or report issue.");
                    }
                } else if(cleanedArgs[0].equals(CMD_TEST)) {
                    Map<String, Object> params = new HashMap<String, Object>();
                    params.put(ShifuTestProcessor.IS_TO_TEST_FILTER, cmd.hasOption(FILTER));
                    params.put(ShifuTestProcessor.TEST_TARGET, cmd.getOptionValue(FILTER));
                    params.put(ShifuTestProcessor.TEST_RECORD_CNT, cmd.getOptionValue(N));
                    status = runShifuTest(params);
                    if(status == 0) {
                        log.info("Run test for Shifu Successfully.");
                    } else {
                        log.warn("Fail to run Shifu test.");
                    }
                } else if(cleanedArgs[0].equals(CMD_CONVERT)) {
                    int optType = -1;
                    if(cmd.hasOption(TO_ZIPB)) {
                        optType = 1;
                    } else if(cmd.hasOption(TO_TREEB)) {
                        optType = 2;
                    }

                    String[] convertArgs = new String[2];
                    int j = 0;
                    for(int i = 1; i < cleanedArgs.length; i++) {
                        if(!cleanedArgs[i].startsWith("-")) {
                            convertArgs[j++] = cleanedArgs[i];
                        }
                    }

                    if(optType < 0 || StringUtils.isBlank(convertArgs[0]) || StringUtils.isBlank(convertArgs[1])) {
                        printUsage();
                    } else {
                        status = runShifuConvert(optType, convertArgs[0], convertArgs[1]);
                    }
                } else if(cleanedArgs[0].equals(CMD_ANALYSIS)) {
                    if(cmd.hasOption(FI)) {
                        String modelPath = cmd.getOptionValue(FI);
                        analysisModelFi(modelPath);
                    }
                } else {
                    log.error("Invalid command, please check help message.");
                    printUsage();
                }
            }
            // for some case jvm cannot stop
            System.exit(status);
        } catch (ShifuException e) {
            // need define error code in each step.
            log.error(e.getError().toString() + "; msg: " + e.getMessage(), e.getCause());
            exceptionExit(e);
        } catch (Exception e) {
            exceptionExit(e);
        }
    }
    private static String[] cleanArgs(String[] args) {
        // get -D parameters at first and set it in Environment then clean args
        List<String> cleanedArgsList = new ArrayList<>();
        for(int i = 0; i < args.length; i++) {
            if(args[i].startsWith("-D")) {
                // remove '-D' at first
                String keyValue = args[i].substring(2);
                int index = keyValue.indexOf("=");
                String key = keyValue.substring(0, index).trim();
                String value = "";
                if(keyValue.length() >= index + 1) {
                    value = keyValue.substring(index + 1).trim();
                }
                // set to Environment for others to read
                Environment.setProperty(key, value);
                // such parameter will also be set in system properties for later reference in correlation and others
                System.setProperty(key, value);
            } else {
                cleanedArgsList.add(args[i]);
            }
        }

        return cleanedArgsList.toArray(new String[0]);
    }

    /*
     * switch model - switch the current model to</p>
     * <p>
     * <li>master if it's not current model existing</li>
     * <li><code>modelName</code> if you already save it with name <code>modelName</code></li>
     * <p>
     * then create a new branch with naming <code>newModelSetName</code>
     */
    private static void switchCurrentModel(String newModelSetName) throws Exception {
        ManageModelProcessor p = new ManageModelProcessor(ModelAction.SWITCH, newModelSetName);
        p.run();
    }

    /*
     * save model - save current mode or save to a specially name <code>newModelSetName</code>
     */
    private static void saveCurrentModel(String newModelSetName) throws Exception {
        ManageModelProcessor p = new ManageModelProcessor(ModelAction.SAVE, newModelSetName);
        p.run();
    }

    /*
     * Create new model - create directory and ModelConfig for the model
     */
    public static int createNewModel(String modelSetName, String modelType, String description) throws Exception {
        ALGORITHM modelAlg = null;
        if(modelType != null) {
            for(ALGORITHM alg: ALGORITHM.values()) {
                if(alg.name().equalsIgnoreCase(modelType.trim())) {
                    modelAlg = alg;
                }
            }
        } else {
            modelAlg = ALGORITHM.NN;
        }

        if(modelAlg == null) {
            log.error("Unsupported algirithm - {}", modelType);
            return 2;
        }

        CreateModelProcessor p = new CreateModelProcessor(modelSetName, modelAlg, description);
        return p.run();
    }

    /*
     * Load the column definition and do the training data purification
     */
    public static int initializeModel() throws Exception {
        InitModelProcessor processor = new InitModelProcessor();
        return processor.run();
    }

    /*
     * Calculate variables stats for model - ks/iv/mean/max/min
     */
    public static int calModelStats(Map<String, Object> params) throws Exception {
        StatsModelProcessor p = new StatsModelProcessor(params);
        return p.run();
    }

    /*
     * Select variables for model
     */
    public static int selectModelVar(Map<String, Object> params) throws Exception {
        VarSelectModelProcessor p = new VarSelectModelProcessor(params);
        return p.run();
    }

    /*
     * Normalize the training data
     */
    public static int normalizeTrainData() throws Exception {
        return normalizeTrainData(null);
    }

    /*
     * Normalize the training data
     */
    public static int normalizeTrainData(Map<String, Object> params) throws Exception {
        NormalizeModelProcessor p = new NormalizeModelProcessor(params);
        return p.run();
    }

    public static int trainModel(boolean isToShuffle) throws Exception {
        TrainModelProcessor p = new TrainModelProcessor();
        p.setToShuffle(isToShuffle);
        return p.run();
    }

    public static int postTrainModel() throws Exception {
        PostTrainModelProcessor p = new PostTrainModelProcessor();
        return p.run();
    }

    public static int createNewEvalSet(String evalSetName) throws Exception {
        EvalModelProcessor p = new EvalModelProcessor(EvalStep.NEW, evalSetName);
        return p.run();
    }

    public static int runEvalSet(Map<String, Object> params) throws Exception {
        EvalModelProcessor p = new EvalModelProcessor(EvalStep.RUN, params);
        return p.run();
    }

    public static int runEvalSet(String evalSetName, Map<String, Object> params) throws Exception {
        log.info("Run evaluation set with {}", evalSetName);
        EvalModelProcessor p = new EvalModelProcessor(EvalStep.RUN, evalSetName, params);
        return p.run();
    }

    public static int runEvalScore(String evalSetNames, Map<String, Object> params) throws Exception {
        EvalModelProcessor p = new EvalModelProcessor(EvalStep.SCORE, evalSetNames, params);
        return p.run();
    }

    private static void runAuditEval(String evalSetNames, Map<String, Object> params) throws Exception {
        EvalModelProcessor p = new EvalModelProcessor(EvalStep.AUDIT, evalSetNames, params);
        p.run();
    }

    private static int runEvalConfMat(String evalSetNames) throws Exception {
        EvalModelProcessor p = new EvalModelProcessor(EvalStep.CONFMAT, evalSetNames);
        return p.run();
    }

    private static int runEvalPerf(String evalSetNames) throws Exception {
        EvalModelProcessor p = new EvalModelProcessor(EvalStep.PERF, evalSetNames);
        return p.run();
    }

    private static int runEvalNorm(String evalSetNames, Map<String, Object> params) throws Exception {
        EvalModelProcessor p = new EvalModelProcessor(EvalStep.NORM, evalSetNames, params);
        return p.run();
    }

    private static int listEvalSet() throws Exception {
        EvalModelProcessor p = new EvalModelProcessor(EvalStep.LIST);
        return p.run();
    }

    private static int deleteEvalSet(String evalSetName) throws Exception {
        EvalModelProcessor p = new EvalModelProcessor(EvalStep.DELETE, evalSetName);
        return p.run();
    }

    private static void copyModel(String[] cmdArgs) throws IOException, ShifuException {
        BasicModelProcessor p = new BasicModelProcessor();

        p.copyModelFiles(cmdArgs[0], cmdArgs[1]);
    }

    public static int exportModel(String type, Map<String, Object> params) throws Exception {
        ExportModelProcessor p = new ExportModelProcessor(type, params);
        return p.run();
    }

    private static int createNewCombo(String algorithms) throws Exception {
        Processor processor = new ComboModelProcessor(ComboModelProcessor.ComboStep.NEW, algorithms);
        return processor.run();
    }

    private static int initComboModels() throws Exception {
        Processor processor = new ComboModelProcessor(ComboModelProcessor.ComboStep.INIT);
        return processor.run();
    }

    private static int runComboModels(boolean isToShuffleData, boolean isToResume) throws Exception {
        ComboModelProcessor processor = new ComboModelProcessor(ComboModelProcessor.ComboStep.RUN);
        processor.setToShuffleData(isToShuffleData);
        processor.setToResume(isToResume);
        return processor.run();
    }

    private static int evalComboModels(boolean isToResume) throws Exception {
        ComboModelProcessor processor = new ComboModelProcessor(ComboModelProcessor.ComboStep.EVAL);
        processor.setToResume(isToResume);
        return processor.run();
    }

    private static void initializeModelParam() throws Exception {
        InitModelProcessor p = new InitModelProcessor();
        p.checkAlgorithmParam();
    }

    private static int runEncode(Map<String, Object> params) {
        ModelDataEncodeProcessor processor = new ModelDataEncodeProcessor(params);
        return processor.run();
    }

    private static int runShifuTest(Map<String, Object> params) {
        ShifuTestProcessor processor = new ShifuTestProcessor(params);
        return processor.run();
    }

    public static int runShifuConvert(int optType, String fromFilePath, String toFilePath) {
        IndependentTreeModelUtils modelUtils = new IndependentTreeModelUtils();
        boolean status = false;
        if(optType == 1) {
            status = modelUtils.convertBinaryToZipSpec(new File(fromFilePath), new File(toFilePath));
        } else if(optType == 2) {
            status = modelUtils.convertZipSpecToBinary(new File(fromFilePath), new File(toFilePath));
        }
        return (status ? 0 : 1);
    }

    public static int analysisModelFi(String modelPath) {
        File modelFile = new File(modelPath);
        if(!modelFile.exists() || !(modelPath.toUpperCase().endsWith("." + CommonConstants.GBT_ALG_NAME)
                || modelPath.toUpperCase().endsWith("." + CommonConstants.RF_ALG_NAME))) {
            log.error("The model {} doesn't exist or it isn't GBT/RF model.", modelPath);
            return 1;
        }

        FileInputStream inputStream = null;
        String fiFileName = modelFile.getName() + ".fi";

        try {
            inputStream = new FileInputStream(modelFile);
            BasicML basicML = TreeModel.loadFromStream(inputStream);
            Map<Integer, MutablePair<String, Double>> featureImportances = CommonUtils
                    .computeTreeModelFeatureImportance(Arrays.asList(new BasicML[] { basicML }));
            CommonUtils.writeFeatureImportance(fiFileName, featureImportances);
        } catch (IOException e) {
            log.error("Fail to analysis model FI for {}", modelPath);
            return 1;
        } finally {
            IOUtils.closeQuietly(inputStream);
        }

        return 0;
    }

    private static void printModelSetCopiedSuccessfulLog(String newModelSetName) {
        log.info(String.format("ModelSet %s is copied successfully with ModelConfig.json in %s folder.",
                newModelSetName, newModelSetName));
        log.info(String.format(
                "Please change your folder to %s and then configure your ModelConfig.json or directly do initialization step by 'shifu init.'",
                newModelSetName));
    }

    private static void printModelSetCreatedSuccessfulLog(String modelName) {
        log.info(String.format("ModelSet %s is created successfully with ModelConfig.json in %s folder.", modelName,
                modelName));
        log.info(String.format(
                "Please change your folder to %s and then configure your ModelConfig.json or directly do initialization step by 'shifu init.'",
                modelName));
    }

    @SuppressWarnings("static-access")
    private static Options buildModelSetOptions() {
        Options opts = new Options();

        Option opt_cmt = OptionBuilder.hasArg().withDescription("The description for new model").create(MODELSET_CMD_M);
        Option opt_new = OptionBuilder.hasArg().withDescription("To create an eval set").create(NEW);
        Option opt_type = OptionBuilder.hasArg().withDescription("Specify model type").create(MODELSET_CMD_TYPE);
        Option opt_run = OptionBuilder.hasOptionalArg().withDescription("To run eval set").create(EVAL_CMD_RUN);
        Option opt_model = OptionBuilder.hasArg(false).withDescription("Init model").create(INIT_CMD_MODEL);
        Option opt_concise = OptionBuilder.hasArg(false).withDescription("Export concise PMML").create(EXPORT_CONCISE);

        // options for variable selection
        Option opt_reset = OptionBuilder.hasArg(false).withDescription("Reset all variables to finalSelect = false")
                .create(RESET);
        Option opt_filter_auto = OptionBuilder.hasArg(false)
                .withDescription("Auto filter variables by MissingRate, IV/KS, Correlation").create(FILTER_AUTO);
        Option opt_recover_auto = OptionBuilder.hasArg(false)
                .withDescription("Recover auto filtered variables from history.").create(RECOVER_AUTO);
        Option opt_recursive = OptionBuilder.hasArg().withDescription("Run variable selection recursively")
                .create(RECURSIVE);
        Option opt_varsel_file = OptionBuilder.hasArg().withDescription("Run variable selection based on file")
                .create(VAR_SEL_FILE);

        Option opt_correlation = OptionBuilder.hasArg(false)
                .withDescription("Compute correlation value for all column pairs.").create(CORRELATION);
        Option opt_correlation_short = OptionBuilder.hasArg(false)
                .withDescription("Compute correlation value for all column pairs.").create("c");
        Option opt_psi = OptionBuilder.hasArg(false).withDescription("Compute psi value.").create(PSI);
        Option opt_psi_short = OptionBuilder.hasArg(false).withDescription("Compute psi value.").create(SHORT_PSI);
        Option opt_uso = OptionBuilder.hasArg(false)
                .withDescription("Compute stats value with given binning in local ColumnConfig.json.")
                .create(UPDATE_STATS_ONLY);
        Option opt_uso_short = OptionBuilder.hasArg(false)
                .withDescription("Compute stats value with given binning in local ColumnConfig.json.")
                .create(SHORT_UPDATE_STATS_ONLY);

        Option opt_shuffle = OptionBuilder.hasArg(false).withDescription("Shuffle data after normalization")
                .create(SHUFFLE);
        Option opt_rebalance = OptionBuilder.hasArg().withDescription("rebalance ratio for positive instances")
                .create(REBALANCE);
        Option opt_update_weight = OptionBuilder.hasArg(false).withDescription("re-balance data by updating weights")
                .create(UPDATE_WEIGHT);
        Option opt_resume = OptionBuilder.hasArg(false).withDescription("Resume combo model training.").create(RESUME);

        Option opt_list = OptionBuilder.hasArg(false).create(LIST);
        Option opt_delete = OptionBuilder.hasArg().create(DELETE);
        Option opt_score = OptionBuilder.hasOptionalArg().create(SCORE);
        Option opt_confmat = OptionBuilder.hasArg().create(CONFMAT);
        Option opt_perf = OptionBuilder.hasArg().create(PERF);
        Option opt_norm = OptionBuilder.hasArg().create(NORM);
        Option opt_eval = OptionBuilder.hasArg(false).create(EVAL_CMD);
        Option opt_init = OptionBuilder.hasArg(false).create(INIT_CMD);
        Option opt_nosort = OptionBuilder.hasArg(false).create(NOSORT);
        Option opt_ref = OptionBuilder.hasArg(true).create(REF);
        Option opt_filter = OptionBuilder.hasOptionalArg().create(FILTER);
        Option opt_strict = OptionBuilder.hasArg(false).create(STRICT);
        Option opt_audit = OptionBuilder.hasOptionalArg().create(AUDIT);

        // options for variable re-binning
        Option opt_rebin = OptionBuilder.hasArg(false).create(REBIN);
        Option opt_vars = OptionBuilder.hasArg().create(VARS);
        Option opt_n = OptionBuilder.hasArg().create(N);
        Option opt_ivr = OptionBuilder.hasArg().create(IVR);
        Option opt_bic = OptionBuilder.hasArg().create(BIC);

        Option opt_save = OptionBuilder.hasArg(false).withDescription("save model").create(SAVE);
        Option opt_switch = OptionBuilder.hasArg(false).withDescription("switch model").create(SWITCH);
        Option opt_eval_model = OptionBuilder.hasArg().withDescription("").create(EVAL_MODEL);

        Option opt_tozipb = OptionBuilder.hasArg(false).create(TO_ZIPB);
        Option opt_totreeb = OptionBuilder.hasArg(false).create(TO_TREEB);

        Option opt_fi = OptionBuilder.hasArg(true).create(FI);

        Option opt_name = OptionBuilder.hasArg(true).withDescription("New model name for model spec.").create(NAME);

        opts.addOption(opt_cmt);
        opts.addOption(opt_new);
        opts.addOption(opt_type);
        opts.addOption(opt_run);
        opts.addOption(opt_perf);
        opts.addOption(opt_norm);
        opts.addOption(opt_model);
        opts.addOption(opt_concise);
        opts.addOption(opt_nosort);
        opts.addOption(opt_ref);
        opts.addOption(opt_filter);
        opts.addOption(opt_strict);

        opts.addOption(opt_reset);
        opts.addOption(opt_filter_auto);
        opts.addOption(opt_recover_auto);
        opts.addOption(opt_recursive);
        opts.addOption(opt_varsel_file);

        opts.addOption(opt_eval);
        opts.addOption(opt_init);
        opts.addOption(opt_shuffle);
        opts.addOption(opt_update_weight);
        opts.addOption(opt_rebalance);
        opts.addOption(opt_resume);

        opts.addOption(opt_list);
        opts.addOption(opt_delete);
        opts.addOption(opt_score);
        opts.addOption(opt_confmat);
        opts.addOption(opt_save);
        opts.addOption(opt_switch);
        opts.addOption(opt_eval_model);
        opts.addOption(opt_correlation);
        opts.addOption(opt_correlation_short);
        opts.addOption(opt_psi);
        opts.addOption(opt_psi_short);
        opts.addOption(opt_audit);

        opts.addOption(opt_uso);
        opts.addOption(opt_uso_short);

        opts.addOption(opt_rebin);
        opts.addOption(opt_vars);
        opts.addOption(opt_n);
        opts.addOption(opt_ivr);
        opts.addOption(opt_bic);

        opts.addOption(opt_tozipb);
        opts.addOption(opt_totreeb);

        opts.addOption(opt_fi);
        opts.addOption(opt_name);

        return opts;
    }

    /*
     * print usage
     */
    private static void printUsage() {
        System.out.println("Usage: shifu [-Dkey=value] COMMAND");
        System.out.println("where COMMAND is one of:");
        System.out.println("\tnew <ModelSetName> [-t <NN|LR|RF|GBT>]  Create a new model set.");
        System.out.println(
                "\tinit                                    Create initial ColumnConfig.json and upload to HDFS.");
        System.out.println(
                "\tstats                                   Calculate statistics on HDFS and update local ColumnConfig.json.");
        System.out.println(
                "\tstats -correlation(c)                   Calculate correlation values between column pairs.");
        System.out.println("\tstats -psi(p)                           Calculate psi values if psi column is provided.");
        System.out.println(
                "\tstats -updatestatsonly                  Calculate stats values if given bin boundaries in ColumnConfig.json.");
        System.out.println("\tstats -rebin [-vars var1,var1] [-ivr <ratio>] [-bic <bic>] [-n <expectBinCnt>]");
        System.out.println("\t                                        Do the variable Re-bin.");
        System.out.println(
                "\tvarselect/varsel                        Variable selection, will update finalSelect in ColumnConfig.json.");
        System.out.println("\tvarselect/varsel -list                  List all finalSelect=true variables");
        System.out.println("\tvarselect/varsel -reset                 Set finalSelect=false for all variables");
        System.out.println(
                "\tvarselect/varsel -autofilter            Auto filter variables by MissingRate, KS/IV, and Correlation.");
        System.out.println("\tvarselect/varsel -recoverauto           Recover those variables that are auto-filtered.");
        System.out.println("\tvarselect/varsel -r                     Run variable selection recursively.");
        System.out.println(
                "\tvarselect/varsel -f <file>              Run variable selection based on some file. The file could be raw file, model spec or ColumnConfig.json.");
        System.out.println("\tnormalize/norm/transform [-shuffle] [-rebalance <ratio>] [-updateweight]");
        System.out.println("\t                                        Normalize the columns with finalSelect as true.");
        System.out.println("\ttrain [-dry] [-shuffle]                 Train the model with the normalized data.");
        System.out.println("\tposttrain                               Post-process data after training models.");
        System.out.println("\teval                                    Run all eval sets.");
        System.out.println("\teval -list                              Lis all eval set.");
        System.out.println("\teval -new     <EvalSetName>             Create a new eval set.");
        System.out.println("\teval -delete  <EvalSetName>             Delete an eval set.");
        System.out.println("\teval -run     <EvalSetName>             Run eval set evaluation.");
        System.out.println("\teval -score   <EvalSetName> [-nosort]   Scoring evaluation dataset.");
        System.out.println("\teval -norm    <EvalSetName>             Normalize evaluation dataset.");
        System.out.println("\teval -confmat <EvalSetName>             Compute the TP/FP/TN/FN based on scoring.");
        System.out
                .println("\teval -perf <EvalSetName>                Calculate the model performance based on confmat.");
        System.out.println("\teval -audit [-n <#numofrecords>]        Score eval data and generate audit dataset.");
        System.out.println(
                "\texport [-t pmml|columnstats|woemapping|bagging|baggingpmml|corr|woe|ume|baggingume|normume] [-c] [-vars var1,var1] [-ivr <ratio>] [-bic <bic>] [-name <modelName>]");
        System.out.println(
                "\t                                        Export model to PMML format or export ColumnConfig.");
        System.out.println(
                "\tcombo -new    <Algorithm List>          Create a combo model train. Algorithm lis should be NN,LR,RF,GBT");
        System.out.println("\tcombo -init                             Generate sub-models.");
        System.out.println("\tcombo -run [-shuffle] [-resume]         Run Combo-Model train.");
        System.out.println("\tcombo -eval [-resume]                   Evaluate Combo-Model performance.");
        System.out.println("\tencode -run [TDS|EvalSetNames] [-ref encode_ref_model]");
        System.out.println(
                "\t                                        Run encode on training or evaluation datasets and set them to encode_ref_model.");
        System.out.println("\ttest [-filter [EvalSetNames]] [-n]      Run testing for Shifu to detect error early.");
        System.out.println("\tversion|v|-v|-version                   Print version of current package.");
        System.out.println("\thelp|h|-h|-help                         Help message.");
    }

    /**
     * print version info for shifu
     */
    private static void printLogoAndVersion() {
        String findContainingJar = JarManager.findContainingJar(ShifuCLI.class);
        JarFile jar = null;
        try {
            jar = new JarFile(findContainingJar);
            final Manifest manifest = jar.getManifest();

            String vendor = manifest.getMainAttributes().getValue("vendor");
            String title = manifest.getMainAttributes().getValue("title");
            String version = manifest.getMainAttributes().getValue("version");
            String timestamp = manifest.getMainAttributes().getValue("timestamp");
            System.out.println(" ____  _   _ ___ _____ _   _ ");
            System.out.println("/ ___|| | | |_ _|  ___| | | |");
            System.out.println("\\___ \\| |_| || || |_  | | | |");
            System.out.println(" ___) |  _  || ||  _| | |_| |");
            System.out.println("|____/|_| |_|___|_|    \\___/ ");
            System.out.println("                             ");
            System.out.println(vendor + " " + title + " version " + version + " \ncompiled " + timestamp);
        } catch (Exception e) {
            throw new RuntimeException("unable to read pigs manifest file", e);
        } finally {
            if(jar != null) {
                try {
                    jar.close();
                } catch (IOException e) {
                    throw new RuntimeException("jar closed failed", e);
                }
            }
        }
    }

    private static boolean isVersionOption(String arg) {
        return arg.equalsIgnoreCase("v") || arg.equalsIgnoreCase("version") || arg.equalsIgnoreCase("-version")
                || arg.equalsIgnoreCase("-v");
    }

    private static boolean isHelpOption(String str) {
        return "h".equalsIgnoreCase(str) || "-h".equalsIgnoreCase(str) || "help".equalsIgnoreCase(str)
                || "-help".equalsIgnoreCase(str);
    }

    private static void exceptionExit(Exception e) {
        log.error("Error in running, please check the stack, msg:" + e.toString(), e);
        System.err.println(Constants.CONTACT_MESSAGE);
        System.exit(-1);
    }
}