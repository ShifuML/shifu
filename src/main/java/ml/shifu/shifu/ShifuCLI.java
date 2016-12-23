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

import java.io.IOException;
import java.util.jar.JarFile;
import java.util.jar.Manifest;

import ml.shifu.shifu.container.obj.ModelTrainConf.ALGORITHM;
import ml.shifu.shifu.core.processor.*;
import ml.shifu.shifu.core.processor.EvalModelProcessor.EvalStep;
import ml.shifu.shifu.core.processor.ManageModelProcessor.ModelAction;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.util.Constants;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.lang.StringUtils;
import org.apache.pig.impl.util.JarManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
    private static final String TRAIN_CMD_DEBUG = "debug";
    private static final String TRAIN_CMD_DRY = "dry";
    private static final String TRAIN_CMD = "train";
    private static final String VARSELECT_CMD = "varselect";
    private static final String VARSEL_CMD = "varsel";
    private static final String NORMALIZE_CMD = "normalize";
    private static final String NORM_CMD = "norm";
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

    private static final String RESET = "reset";

    // for evaluation
    private static final String LIST = "list";
    private static final String DELETE = "delete";
    private static final String SCORE = "score";
    private static final String CONFMAT = "confmat";
    private static final String PERF = "perf";
    private static final String NORM = "norm";

    private static final String SAVE = "save";
    private static final String SWITCH = "switch";
    private static final String EVAL_MODEL = "model";
    private static final String SHOW = "show";

    static private final Logger log = LoggerFactory.getLogger(ShifuCLI.class);

    /**
     * Main entry for the whole framework.
     * 
     * @throws IOException
     */
    public static void main(String[] args) {
        // invalid input and help options
        if(args.length < 1 || (isHelpOption(args[0]))) {
            printUsage();
            System.exit(args.length < 1 ? -1 : 0);
        }

        // process -v and -version conditions manually
        if(isVersionOption(args[0])) {
            printLogoAndVersion();
            System.exit(0);
        }

        CommandLineParser parser = new GnuParser();
        Options opts = buildModelSetOptions(args);
        CommandLine cmd = null;

        try {
            cmd = parser.parse(opts, args);
        } catch (ParseException e) {
            log.error("Invalid command options. Please check help message.");
            printUsage();
            System.exit(1);
        }

        int status = 0;

        try {
            if(args[0].equals(NEW) && args.length >= 2 && StringUtils.isNotEmpty(args[1])) {
                // modelset step
                String modelName = args[1];
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
                if(args[0].equals(MODELSET_CMD_CP) && args.length >= 3 && StringUtils.isNotEmpty(args[1])
                        && StringUtils.isNotEmpty(args[2])) {
                    String newModelSetName = args[2];
                    // modelset step
                    copyModel(new String[] { args[1], newModelSetName });
                    printModelSetCopiedSuccessfulLog(newModelSetName);
                } else if(args[0].equals(INIT_CMD)) {
                    // init step
                    if(cmd.getOptions() == null || cmd.getOptions().length == 0) {
                        status = initializeModel();
                        if(status == 0) {
                            log.info("ModelSet initilization is successful. Please continue next step by using 'shifu stats'.");
                        } else {
                            log.warn("Error in ModelSet initilization, please check your shifu config or report issue");
                        }
                    } else if(cmd.hasOption(INIT_CMD_MODEL)) {
                        initializeModelParam();
                    } else {
                        log.error("Invalid command, please check help message.");
                        printUsage();
                    }
                } else if(args[0].equals(STATS_CMD)) {
                    // stats step
                    status = calModelStats();
                    if(status == 0) {
                        log.info("Do model set statistics successfully. Please continue next step by using 'shifu normalize or shifu norm'. For tree ensemble model, no need do norm, please continue next step by using 'shifu varsel'");
                    } else {
                        log.warn("Error in model set stats computation, please report issue on http:/github.com/shifuml/shifu/issues.");
                    }
                } else if(args[0].equals(NORMALIZE_CMD) || args[0].equals(NORM_CMD)) {
                    // normalize step
                    status = normalizeTrainData();
                    if(status == 0) {
                        log.info("Do model set normalization successfully. Please continue next step by using 'shifu varselect or shifu varsel'.");
                    } else {
                        log.warn("Error in model set stats computation, please report issue on http:/github.com/shifuml/shifu/issues.");
                    }
                } else if(args[0].equals(VARSELECT_CMD) || args[0].equals(VARSEL_CMD)) {
                    // variable selected step
                    status = selectModelVar(cmd.hasOption(RESET));
                    if(status == 0) {
                        log.info("Do model set variables selection successfully. Please continue next step by using 'shifu train'.");
                    } else {
                        log.info("Do variable selection with error, please check error message or report issue.");
                    }
                } else if(args[0].equals(TRAIN_CMD)) {
                    // train step
                    status = trainModel(cmd.hasOption(TRAIN_CMD_DRY), cmd.hasOption(TRAIN_CMD_DEBUG));
                    if(status == 0) {
                        log.info("Do model set training successfully. Please continue next step by using 'shifu posttrain' or if no need posttrain you can go through with 'shifu eval'.");
                    } else {
                        log.info("Do model training with error, please check error message or report issue.");
                    }
                } else if(args[0].equals(CMD_COMBO)) {
                    if ( cmd.hasOption(MODELSET_CMD_NEW) ) {
                        log.info("Create new commbo models");
                        status = createNewCombo(cmd.getOptionValue(MODELSET_CMD_NEW));
                    } else if ( cmd.hasOption(INIT_CMD)) {
                        log.info("Init commbo models");
                        status = initComboModels();
                    } else if ( cmd.hasOption(EVAL_CMD_RUN) ) {
                        log.info("Run combo model.");
                        status = runComboModels();
                        // train combo models
                    } else if ( cmd.hasOption(EVAL_CMD) ) {
                        log.info("Eval combo model.");
                        // eval combo model performance
                        status = evalComboModels();
                    } else {
                        log.error("Invalid command usage.");
                        printUsage();
                    }
                } else if(args[0].equals(POSTTRAIN_CMD)) {
                    // post train step
                    status = postTrainModel();
                    if(status == 0) {
                        log.info("Do model set post-training successfully. Please configurate your eval set in ModelConfig.json and continue next step by using 'shifu eval' or 'shifu eval -new <eval set>' to create a new eval set.");
                    } else {
                        log.info("Do model post training with error, please check error message or report issue.");
                    }
                } else if(args[0].equals(SAVE)) {
                    String newModelSetName = args.length >= 2 ? args[1] : null;
                    saveCurrentModel(newModelSetName);
                } else if(args[0].equals(SWITCH)) {
                    String newModelSetName = args[1];
                    switchCurrentModel(newModelSetName);
                } else if(args[0].equals(SHOW)) {
                    ManageModelProcessor p = new ManageModelProcessor(ModelAction.SHOW, null);
                    p.run();
                } else if(args[0].equals(EVAL_CMD)) {
                    // eval step
                    if(args.length == 1) {
                        // run everything
                        status = runEvalSet(cmd.hasOption(TRAIN_CMD_DRY));
                        if(status == 0) {
                            log.info("Run eval performance with all eval sets successfully.");
                        } else {
                            log.info("Do evaluation with error, please check error message or report issue.");
                        }
                    } else if(cmd.getOptionValue(MODELSET_CMD_NEW) != null) {
                        // create new eval
                        createNewEvalSet(cmd.getOptionValue(MODELSET_CMD_NEW));
                        log.info("Create eval set successfully. You can configurate EvalConfig.json or directly run 'shifu eval -run <evalSetName>' to get performance info.");
                    } else if(cmd.hasOption(EVAL_CMD_RUN)) {
                        runEvalSet(cmd.getOptionValue(EVAL_CMD_RUN), cmd.hasOption(TRAIN_CMD_DRY));
                        log.info("Finish run eval performance with eval set {}.", cmd.getOptionValue(EVAL_CMD_RUN));
                    } else if(cmd.hasOption(SCORE)) {

                        // run score
                        runEvalScore(cmd.getOptionValue(SCORE));
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
                    } else if (cmd.hasOption(NORM)) {
                        runEvalNorm(cmd.getOptionValue(NORM));
                    } else {
                        log.error("Invalid command, please check help message.");
                        printUsage();
                    }
                } else if(args[0].equals(CMD_EXPORT)) {
                    boolean isConcise = cmd.hasOption(EXPORT_CONCISE);
                    status = exportModel(cmd.getOptionValue(MODELSET_CMD_TYPE), isConcise);
                    if(status == 0) {
                        log.info("Export models/columnstats to PMML/csv format successfully in current folder.");
                    } else {
                        log.warn("Export models/columnstats to PMML/csv format with error, please check or report issue.");
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
            log.error(e.getError().toString(), e.getCause());
            exceptionExit(e);
        } catch (Exception e) {
            exceptionExit(e);
        }
    }

    /**
     * switch model - switch the current model to</p>
     * <p/>
     * <li>master if it's not current model existing</li>
     * <li><code>modelName</code> if you already save it with name <code>modelName</code></li>
     * <p/>
     * then create a new branch with naming <code>newModelSetName</code>
     * 
     * @param newModelSetName
     * @throws Exception
     */
    private static void switchCurrentModel(String newModelSetName) throws Exception {
        ManageModelProcessor p = new ManageModelProcessor(ModelAction.SWITCH, newModelSetName);
        p.run();
    }

    /**
     * save model - save current mode or save to a specially name <code>newModelSetName</code>
     * 
     * @param newModelSetName
     * @throws Exception
     */
    private static void saveCurrentModel(String newModelSetName) throws Exception {
        ManageModelProcessor p = new ManageModelProcessor(ModelAction.SAVE, newModelSetName);
        p.run();
    }

    /**
     * Create new model - create directory and ModelConfig for the model
     * 
     * @throws Exception
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

    /**
     * Load the column definition and do the training data purification
     * 
     * @throws Exception
     */
    public static int initializeModel() throws Exception {
        InitModelProcessor processor = new InitModelProcessor();
        return processor.run();
    }

    /**
     * Calculate variables stats for model - ks/iv/mean/max/min
     */
    public static int calModelStats() throws Exception {
        StatsModelProcessor p = new StatsModelProcessor();
        return p.run();
    }

    /**
     * Select variables for model
     * 
     * @throws Exception
     * @throws ShifuException
     */
    public static int selectModelVar(boolean isToReset) throws Exception {
        VarSelectModelProcessor p = new VarSelectModelProcessor(isToReset);
        return p.run();
    }

    /**
     * Normalize the training data
     * 
     * @throws Exception
     */
    public static int normalizeTrainData() throws Exception {
        NormalizeModelProcessor p = new NormalizeModelProcessor();
        return p.run();
    }

    /**
     * Train model
     * 
     * @throws Exception
     */
    public static int trainModel(boolean isDryTrain, boolean isDebug) throws Exception {
        TrainModelProcessor p = new TrainModelProcessor(isDryTrain, isDebug);
        return p.run();
    }

    /**
     * Run post-train step
     */
    public static int postTrainModel() throws Exception {
        PostTrainModelProcessor p = new PostTrainModelProcessor();
        return p.run();
    }

    /**
     * Create new evalset
     * 
     * @throws Exception
     */
    public static int createNewEvalSet(String evalSetName) throws Exception {
        EvalModelProcessor p = new EvalModelProcessor(EvalStep.NEW, evalSetName);
        return p.run();
    }

    /**
     * Run the evalset to test the model with isDry switch
     */
    public static int runEvalSet(boolean isDryRun) throws Exception {
        EvalModelProcessor p = new EvalModelProcessor(EvalStep.RUN);
        return p.run();
    }

    /**
     * @param evalSetName
     * @param isDryRun
     * @throws Exception
     */
    public static int runEvalSet(String evalSetName, boolean isDryRun) throws Exception {
        log.info("Run evaluation set with {}", evalSetName);
        EvalModelProcessor p = new EvalModelProcessor(EvalStep.RUN, evalSetName);
        return p.run();
    }

    /**
     * @param evalSetNames
     * @throws Exception
     */
    public static int runEvalScore(String evalSetNames) throws Exception {
        EvalModelProcessor p = new EvalModelProcessor(EvalStep.SCORE, evalSetNames);
        return p.run();
    }

    /**
     * @param evalSetNames
     * @throws Exception
     */
    private static int runEvalConfMat(String evalSetNames) throws Exception {
        EvalModelProcessor p = new EvalModelProcessor(EvalStep.CONFMAT, evalSetNames);
        return p.run();
    }

    /**
     * @param evalSetNames
     * @throws Exception
     */
    private static int runEvalPerf(String evalSetNames) throws Exception {
        EvalModelProcessor p = new EvalModelProcessor(EvalStep.PERF, evalSetNames);
        return p.run();
    }

    /**
     * @param evalSetNames
     * @return
     */
    private static int runEvalNorm(String evalSetNames) throws Exception {
        EvalModelProcessor p = new EvalModelProcessor(EvalStep.NORM, evalSetNames);
        return p.run();
    }

    /**
     * list all evaluation set
     * 
     * @throws Exception
     */
    private static int listEvalSet() throws Exception {
        EvalModelProcessor p = new EvalModelProcessor(EvalStep.LIST);
        return p.run();
    }

    /**
     * delete some evaluation set
     * 
     * @param evalSetName
     * @throws Exception
     */
    private static int deleteEvalSet(String evalSetName) throws Exception {
        EvalModelProcessor p = new EvalModelProcessor(EvalStep.DELETE, evalSetName);
        return p.run();
    }

    /**
     * create a new model from existing model
     * 
     * @throws ShifuException
     */
    private static void copyModel(String[] cmdArgs) throws IOException, ShifuException {
        BasicModelProcessor p = new BasicModelProcessor();

        p.copyModelFiles(cmdArgs[0], cmdArgs[1]);
    }

    /**
     * export Shifu model into other format, i.e. PMML
     * 
     * @param type
     * @throws Exception
     */
    public static int exportModel(String type, boolean isConcise) throws Exception {
        ExportModelProcessor p = new ExportModelProcessor(type, isConcise);
        return p.run();
    }

    /**
     * create ComboTrain.json, when user provide the algorithms to combo
     * @param algorithms
     * @return
     * @throws Exception
     */
    private static int createNewCombo(String algorithms) throws Exception {
        Processor processor = new ComboModelProcessor(ComboModelProcessor.ComboStep.NEW, algorithms);
        return processor.run();
    }

    /**
     * create each sub-models, assemble model and generate corresponding configurations
     * @return
     * @throws Exception
     */
    private static int initComboModels() throws Exception {
        Processor processor = new ComboModelProcessor(ComboModelProcessor.ComboStep.INIT);
        return processor.run();
    }

    /**
     * train each sub-models, and use train data as evaluation set to generate model score.
     *      And join the evaluation result to train assemble model
     * @return
     * @throws Exception
     */
    private static int runComboModels() throws Exception {
        Processor processor = new ComboModelProcessor(ComboModelProcessor.ComboStep.RUN);
        return processor.run();
    }

    /**
     * evaluate each sub-models, join data and evaluate assemble model
     * @return
     * @throws Exception
     */
    private static int evalComboModels() throws Exception {
        Processor processor = new ComboModelProcessor(ComboModelProcessor.ComboStep.EVAL);
        return processor.run();
    }

    /**
     * Load and test ModelConfig
     * 
     * @throws Exception
     */
    private static void initializeModelParam() throws Exception {
        InitModelProcessor p = new InitModelProcessor();
        p.checkAlgorithmParam();
    }

    private static void printModelSetCopiedSuccessfulLog(String newModelSetName) {
        log.info(String.format("ModelSet %s is copied successfully with ModelConfig.json in %s folder.",
                newModelSetName, newModelSetName));
        log.info(String
                .format("Please change your folder to %s and then configurate your ModelConfig.json or dirctly do initilization step by 'shifu init.'",
                        newModelSetName));
    }

    private static void printModelSetCreatedSuccessfulLog(String modelName) {
        log.info(String.format("ModelSet %s is created successfully with ModelConfig.json in %s folder.", modelName,
                modelName));
        log.info(String
                .format("Please change your folder to %s and then configurate your ModelConfig.json or dirctly do initilization step by 'shifu init.'",
                        modelName));
    }

    /**
     * Build the usage option for parameter check
     */
    @SuppressWarnings("static-access")
    private static Options buildModelSetOptions(String[] args) {
        Options opts = new Options();

        Option opt_cmt = OptionBuilder.hasArg()
                .withDescription("The description for new model").create(MODELSET_CMD_M);
        Option opt_new = OptionBuilder.hasArg()
                .withDescription("To create an eval set").create(NEW);
        Option opt_type = OptionBuilder.hasArg()
                .withDescription("Specify model type").create(MODELSET_CMD_TYPE);
        Option opt_run = OptionBuilder.hasOptionalArg()
                .withDescription("To run eval set").create(EVAL_CMD_RUN);
        Option opt_dry = OptionBuilder.hasArg(false)
                .withDescription("Dry run the train").create(TRAIN_CMD_DRY);
        Option opt_debug = OptionBuilder.hasArg(false)
                .withDescription("Save the log of train process").create(TRAIN_CMD_DEBUG);
        Option opt_model = OptionBuilder.hasArg(false)
                .withDescription("Init model").create(INIT_CMD_MODEL);
        Option opt_concise = OptionBuilder.hasArg(false)
                .withDescription("Export concise PMML").create(EXPORT_CONCISE);
        Option opt_reset = OptionBuilder.hasArg(false)
                .withDescription("Reset all variables to finalSelect = false").create(RESET);

        Option opt_list = OptionBuilder.hasArg(false).create(LIST);
        Option opt_delete = OptionBuilder.hasArg().create(DELETE);
        Option opt_score = OptionBuilder.hasArg().create(SCORE);
        Option opt_confmat = OptionBuilder.hasArg().create(CONFMAT);
        Option opt_perf = OptionBuilder.hasArg().create(PERF);
        Option opt_norm = OptionBuilder.hasArg().create(NORM);
        Option opt_eval = OptionBuilder.hasArg(false).create(EVAL_CMD);
        Option opt_init = OptionBuilder.hasArg(false).create(INIT_CMD);

        Option opt_save = OptionBuilder.hasArg(false).withDescription("save model").create(SAVE);
        Option opt_switch = OptionBuilder.hasArg(false).withDescription("switch model").create(SWITCH);
        Option opt_eval_model = OptionBuilder.hasArg().withDescription("").create(EVAL_MODEL);

        opts.addOption(opt_cmt);
        opts.addOption(opt_new);
        opts.addOption(opt_type);
        opts.addOption(opt_run);
        opts.addOption(opt_perf);
        opts.addOption(opt_norm);
        opts.addOption(opt_dry);
        opts.addOption(opt_debug);
        opts.addOption(opt_model);
        opts.addOption(opt_concise);
        opts.addOption(opt_reset);
        opts.addOption(opt_eval);
        opts.addOption(opt_init);

        opts.addOption(opt_list);
        opts.addOption(opt_delete);
        opts.addOption(opt_score);
        opts.addOption(opt_confmat);
        opts.addOption(opt_save);
        opts.addOption(opt_switch);
        opts.addOption(opt_eval_model);

        return opts;
    }

    /**
     * print usage
     */
    private static void printUsage() {
        System.out.println("Usage: shifu COMMAND");
        System.out.println("where COMMAND is one of:");
        System.out.println("\tnew <ModelSetName> [-t <NN|LR|SVM|DT>]  Create a new model set.");
        System.out.println("\tinit                                    Create initial ColumnConfig.json and upload to HDFS.");
        System.out.println("\tstats                                   Calculate statistics on HDFS and update local ColumnConfig.json.");
        System.out.println("\tvarselect/varsel [-reset]               Variable selection, will update finalSelect in ColumnConfig.json.");
        System.out.println("\tnormalize/norm                          Normalize the columns with finalSelect as true.");
        System.out.println("\ttrain [-dry]                            Train the model with the normalized data.");
        System.out.println("\tposttrain                               Post-process data after training models.");
        System.out.println("\teval                                    Run all eval sets.");
        System.out.println("\teval -list                              Lis all eval set.");
        System.out.println("\teval -new     <EvalSetName>             Create a new eval set.");
        System.out.println("\teval -delete  <EvalSetName>             Delete an eval set.");
        System.out.println("\teval -run     <EvalSetName>             Run eval set evaluation.");
        System.out.println("\teval -score   <EvalSetName>             Scoring evaluation dataset.");
        System.out.println("\teval -norm    <EvalSetName>             Normalize evaluation dataset.");
        System.out.println("\teval -confmat <EvalSetName>             Compute the TP/FP/TN/FN based on scoring");
        System.out.println("\teval -perf <EvalSetName>                Calculate the model performance based on confmat");
        System.out.println("\texport [-t pmml|columnstats] [-c]       Export model to PMML format or export ColumnConfig.");
        System.out.println("\tcombo -new    <Algorithm List>          Create a combo model train. Algorithm lis should be NN,LR,RF,GBT,LR");
        System.out.println("\tcombo -init                             Generate sub-models.");
        System.out.println("\tcombo -run                              Run Combo-Model train.");
        System.out.println("\tcombo -eval                             Evaluate Combo-Model performance.");
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

    /**
     * check the argument is for listing version or not
     * 
     * @param arg
     *            input option
     * @return true - if arg is v/version/-v/-version, or return false
     */
    private static boolean isVersionOption(String arg) {
        return arg.equalsIgnoreCase("v")
                || arg.equalsIgnoreCase("version")
                || arg.equalsIgnoreCase("-version")
                || arg.equalsIgnoreCase("-v");
    }

    /**
     * check the argument is for listing help info or not
     * 
     * @param str
     * @return true - if arg is h/-h/help/-help, or return false
     */
    private static boolean isHelpOption(String str) {
        return "h".equalsIgnoreCase(str)
                || "-h".equalsIgnoreCase(str)
                || "help".equalsIgnoreCase(str)
                || "-help".equalsIgnoreCase(str);
    }

    /**
     * print exception and contact message, then quit program
     * 
     * @param e
     */
    private static void exceptionExit(Exception e) {
        log.error("Error in running, please check the stack, msg:" + e.toString(), e);
        System.err.println(Constants.CONTACT_MESSAGE);
        System.exit(-1);
    }
}