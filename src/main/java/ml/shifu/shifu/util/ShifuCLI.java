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
package ml.shifu.shifu.util;

import java.io.IOException;
import java.util.jar.JarFile;
import java.util.jar.Manifest;

import ml.shifu.shifu.container.obj.ModelTrainConf.ALGORITHM;
import ml.shifu.shifu.core.processor.BasicModelProcessor;
import ml.shifu.shifu.core.processor.CreateModelProcessor;
import ml.shifu.shifu.core.processor.EvalModelProcessor;
import ml.shifu.shifu.core.processor.EvalModelProcessor.EvalStep;
import ml.shifu.shifu.core.processor.ExportModelProcessor;
import ml.shifu.shifu.core.processor.InitModelProcessor;
import ml.shifu.shifu.core.processor.ManageModelProcessor;
import ml.shifu.shifu.core.processor.ManageModelProcessor.ModelAction;
import ml.shifu.shifu.core.processor.NormalizeModelProcessor;
import ml.shifu.shifu.core.processor.PostTrainModelProcessor;
import ml.shifu.shifu.core.processor.StatsModelProcessor;
import ml.shifu.shifu.core.processor.TrainModelProcessor;
import ml.shifu.shifu.core.processor.VarSelectModelProcessor;
import ml.shifu.shifu.exception.ShifuException;

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
    private static final String NEW = "new";

    private static final String CMD_EXPORT = "export";
    
    // for evaluation
    private static final String LIST = "list";
    private static final String DELETE = "delete";
    private static final String SCORE = "score";
    private static final String CONFMAT = "confmat";
    private static final String PERF = "perf";

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
        if (args.length < 1 || (isHelpOption(args[0]))) {
            printUsage();
            System.exit(args.length < 1 ? -1 : 0);
        }

        // process -v and -version conditions manually
        if (isVersionOption(args[0])) {
            printVersionString();
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

        try {
            if (args[0].equals(NEW) && args.length >= 2 && StringUtils.isNotEmpty(args[1])) {
                // modelset step
                String modelName = args[1];
                int status = createNewModel(modelName, cmd.getOptionValue(MODELSET_CMD_TYPE), cmd.getOptionValue(MODELSET_CMD_M));
                if (status == 0) {
                    printModelSetCreatedSuccessfulLog(modelName);
                }
                // copyModel(manager, cmd.getOptionValues(MODELSET_CMD_CP));
            } else {
                if (args[0].equals(MODELSET_CMD_CP) && args.length >= 3 && StringUtils.isNotEmpty(args[1])
                        && StringUtils.isNotEmpty(args[2])) {
                    String newModelSetName = args[2];
                    // modelset step
                    copyModel(new String[]{args[1], newModelSetName});
                    printModelSetCopiedSuccessfulLog(newModelSetName);
                } else if (args[0].equals(INIT_CMD)) {
                    // init step
                    if (cmd.getOptions() == null || cmd.getOptions().length == 0) {
                        int status = initializeModel();
                        if (status == 0) {
                            log.info("ModelSet initilization is successful. Please continue next step by using 'shifu stats'.");
                        }
                    } else if (cmd.hasOption(INIT_CMD_MODEL)) {
                        initializeModelParam();
                    } else {
                        log.error("Invalid command, please check help message.");
                        printUsage();
                    }
                } else if (args[0].equals(STATS_CMD)) {
                    // stats step
                    calModelStats();
                    log.info("Do model set statistics successfully. Please continue next step by using 'shifu normalize or shifu norm'.");
                } else if (args[0].equals(NORMALIZE_CMD) || args[0].equals(NORM_CMD)) {
                    // normalize step
                    normalizeTrainData();
                    log.info("Do model set normalization successfully. Please continue next step by using 'shifu varselect or shifu varsel'.");
                } else if (args[0].equals(VARSELECT_CMD) || args[0].equals(VARSEL_CMD)) {
                    // variable selected step
                    selectModelVar();
                    log.info("Do model set variables selection successfully. Please continue next step by using 'shifu train'.");
                } else if (args[0].equals(TRAIN_CMD)) {
                    // train step
                    trainModel(cmd.hasOption(TRAIN_CMD_DRY), cmd.hasOption(TRAIN_CMD_DEBUG));
                    log.info("Do model set training successfully. Please continue next step by using 'shifu posttrain' or if no need posttrain you can go through with 'shifu eval'.");
                } else if (args[0].equals(POSTTRAIN_CMD)) {
                    // post train step
                    postTrainModel();
                    log.info("Do model set post-training successfully. Please configurate your eval set in ModelConfig.json and continue next step by using 'shifu eval' or 'shifu eval -new <eval set>' to create a new eval set.");
                } else if (args[0].equals(SAVE)) {
                    String newModelSetName = args.length >= 2 ? args[1] : null;
                    saveCurrentModel(newModelSetName);
                } else if (args[0].equals(SWITCH)) {
                    String newModelSetName = args[1];
                    switchCurrentModel(newModelSetName);
                } else if (args[0].equals(SHOW)) {
                    ManageModelProcessor p = new ManageModelProcessor(ModelAction.SHOW, null);
                    p.run();
                } else if (args[0].equals(EVAL_CMD)) {
                    // eval step
                    if (args.length == 1) {
                        //run everything
                        runEvalSet(cmd.hasOption(TRAIN_CMD_DRY));
                        log.info("Run eval performance with all eval sets successfully.");
                    } else if (cmd.getOptionValue(MODELSET_CMD_NEW) != null) {
                        //create new eval
                        createNewEvalSet(cmd.getOptionValue(MODELSET_CMD_NEW));
                        log.info("Create eval set successfully. You can configurate EvalConfig.json or directly run 'shifu eval -run <evalSetName>' to get performance info.");
                    } else if (cmd.hasOption(EVAL_CMD_RUN)) {
                        runEvalSet(cmd.getOptionValue(EVAL_CMD_RUN), cmd.hasOption(TRAIN_CMD_DRY));
                        log.info("Finish run eval performance with eval set {}.",
                                cmd.getOptionValue(EVAL_CMD_RUN));
                    } else if (cmd.hasOption(SCORE)) {

                        //run score
                        runEvalScore(cmd.getOptionValue(SCORE));
                        log.info("Finish run score with eval set {}.", cmd.getOptionValue(SCORE));
                    } else if (cmd.hasOption(CONFMAT)) {

                        //run confusion matrix
                        runEvalConfMat(cmd.getOptionValue(CONFMAT));
                        log.info("Finish run confusion matrix with eval set {}.", cmd.getOptionValue(CONFMAT));

                    } else if (cmd.hasOption(PERF)) {
                        //run perfermance
                        runEvalPerf(cmd.getOptionValue(PERF));
                        log.info("Finish run performance maxtrix with eval set {}.", cmd.getOptionValue(PERF));

                    } else if (cmd.hasOption(LIST)) {
                        //list all evaluation sets
                        listEvalSet();
                    } else if (cmd.hasOption(DELETE)) {
                        // delete some evaluation set
                        deleteEvalSet(cmd.getOptionValue(DELETE));
                    } else {
                        log.error("Invalid command, please check help message.");
                        printUsage();
                    }
                } else if (args[0].equals(CMD_EXPORT) ) {
                    exportModel(cmd.getOptionValue(MODELSET_CMD_TYPE));
                } else {
                    log.error("Invalid command, please check help message.");
                    printUsage();
                }
            }
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
        if (modelType != null) {
            for (ALGORITHM alg : ALGORITHM.values()) {
                if (alg.name().equalsIgnoreCase(modelType.trim())) {
                    modelAlg = alg;
                }
            }
        } else {
            modelAlg = ALGORITHM.NN;
        }

        if (modelAlg == null) {
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
    public static void calModelStats() throws Exception {
        StatsModelProcessor p = new StatsModelProcessor();
        p.run();
    }

    /**
     * Select variables for model
     *
     * @throws Exception
     * @throws ShifuException
     */
    public static void selectModelVar() throws Exception {
        VarSelectModelProcessor p = new VarSelectModelProcessor();
        p.run();
    }

    /**
     * Normalize the training data
     *
     * @throws Exception
     */
    public static void normalizeTrainData() throws Exception {
        NormalizeModelProcessor p = new NormalizeModelProcessor();
        p.run();
    }

    /**
     * Train model
     *
     * @throws Exception
     */
    public static void trainModel(boolean isDryTrain, boolean isDebug) throws Exception {
        TrainModelProcessor p = new TrainModelProcessor(isDryTrain, isDebug);
        p.run();
    }

    /**
     * Run post-train step
     */
    public static void postTrainModel() throws Exception {
        PostTrainModelProcessor p = new PostTrainModelProcessor();
        p.run();
    }

    /**
     * Create new evalset
     *
     * @throws Exception
     */
    public static void createNewEvalSet(String evalSetName) throws Exception {
        EvalModelProcessor p = new EvalModelProcessor(EvalStep.NEW, evalSetName);

        p.run();
    }

    /**
     * Run the evalset to test the model with isDry switch
     */
    public static void runEvalSet(boolean isDryRun) throws Exception {
        EvalModelProcessor p = new EvalModelProcessor(EvalStep.RUN);
        p.run();
    }

    /**
     * @param evalSetName
     * @param isDryRun
     * @throws Exception
     */
    public static void runEvalSet(String evalSetName, boolean isDryRun) throws Exception {
        //TODO dry run useful?
        EvalModelProcessor p = new EvalModelProcessor(EvalStep.RUN, evalSetName);
        p.run();
    }

    /**
     * @param evalSetNames
     * @throws Exception
     */
    public static void runEvalScore(String evalSetNames) throws Exception {
        EvalModelProcessor p = new EvalModelProcessor(EvalStep.SCORE, evalSetNames);
        p.run();
    }


    /**
     * @param evalSetNames
     * @throws Exception
     */
    private static void runEvalConfMat(String evalSetNames) throws Exception {
        EvalModelProcessor p = new EvalModelProcessor(EvalStep.CONFMAT, evalSetNames);
        p.run();
    }


    /**
     * @param evalSetNames
     * @throws Exception
     */
    private static void runEvalPerf(String evalSetNames) throws Exception {
        EvalModelProcessor p = new EvalModelProcessor(EvalStep.PERF, evalSetNames);
        p.run();
    }


    /**
     * list all evaluation set
     *
     * @throws Exception
     */
    private static void listEvalSet() throws Exception {
        EvalModelProcessor p = new EvalModelProcessor(EvalStep.LIST);
        p.run();
    }


    /**
     * delete some evaluation set
     *
     * @param optionValue
     * @throws Exception
     */
    private static void deleteEvalSet(String evalSetName) throws Exception {
        EvalModelProcessor p = new EvalModelProcessor(EvalStep.DELETE, evalSetName);
        p.run();
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
     * @param optionValue
     * @throws Exception 
     */
    public static void exportModel(String type) throws Exception {
        ExportModelProcessor p = new ExportModelProcessor(type);
        p.run();
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

        Option opt_cmt = OptionBuilder.hasArg().withDescription("The description for new model").create(MODELSET_CMD_M);
        Option opt_new = OptionBuilder.hasArg().withDescription("To create an eval set").create(NEW);
        Option opt_type = OptionBuilder.hasArg().withDescription("Specify model type").create(MODELSET_CMD_TYPE);
        Option opt_run = OptionBuilder.hasArg().withDescription("To run eval set").create(EVAL_CMD_RUN);
        Option opt_dry = OptionBuilder.hasArg(false).withDescription("Dry run the train").create(TRAIN_CMD_DRY);
        Option opt_debug = OptionBuilder.hasArg(false).withDescription("Save the log of train process").create(TRAIN_CMD_DEBUG);
        Option opt_model = OptionBuilder.hasArg(false).withDescription("Init model").create(INIT_CMD_MODEL);

        Option opt_list = OptionBuilder.hasArg(false).create(LIST);
        Option opt_delete = OptionBuilder.hasArg().create(DELETE);
        Option opt_score = OptionBuilder.hasArg().create(SCORE);
        Option opt_confmat = OptionBuilder.hasArg().create(CONFMAT);
        Option opt_perf = OptionBuilder.hasArg().create(PERF);

        Option opt_save = OptionBuilder.hasArg(false).withDescription("save model").create(SAVE);
        Option opt_switch = OptionBuilder.hasArg(false).withDescription("switch model").create(SWITCH);
        Option opt_eval_model = OptionBuilder.hasArg().withDescription("").create(EVAL_MODEL);

        opts.addOption(opt_cmt);
        opts.addOption(opt_new);
        opts.addOption(opt_type);
        opts.addOption(opt_run);
        opts.addOption(opt_perf);
        opts.addOption(opt_dry);
        opts.addOption(opt_debug);
        opts.addOption(opt_model);

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
        System.out.println("\tvarselect/varsel                        Variable selection, will update finalSelect in ColumnConfig.json.");
        System.out.println("\tnormalize/norm                          Normalize the columns with finalSelect as true.");
        System.out.println("\ttrain [-dry]                            Train the model with the normalized data.");
        System.out.println("\tposttrain                               Post-process data after training models.");
        System.out.println("\teval                                    Run all eval sets.");
        System.out.println("\teval -list                              Lis all eval set.");
        System.out.println("\teval -new     <EvalSetName>             Create a new eval set.");
        System.out.println("\teval -delete  <EvalSetName>             Delete an eval set.");
        System.out.println("\teval -run     <EvalSetName>             Run eval set evaluation.");
        System.out.println("\teval -score   <EvalSetName>             Scoring evaluation dataset.");
        System.out.println("\teval -confmat <EvalSetName>             Compute the TP/FP/TN/FN based on scoring");
        System.out.println("\teval -perf <EvalSetName>                Calculate the model performance based on confmat");
        System.out.println("\texport [-t pmml]                        Export model to PMML format.");
        System.out.println("\tversion|v|-v|-version                   Print version of current package.");
        System.out.println("\thelp|h|-h|-help                         Help message.");
    }

    /**
     * print version info for shifu
     */
    private static void printVersionString() {
        String findContainingJar = JarManager.findContainingJar(ShifuCLI.class);
        JarFile jar = null;
        try {
            jar = new JarFile(findContainingJar);
            final Manifest manifest = jar.getManifest();

            String vendor = manifest.getMainAttributes().getValue("vendor");
            String title = manifest.getMainAttributes().getValue("title");
            String version = manifest.getMainAttributes().getValue("version");
            String timestamp = manifest.getMainAttributes().getValue("timestamp");
            System.out.println(vendor + " " + title + " version " + version + " \ncompiled " + timestamp);
        } catch (Exception e) {
            throw new RuntimeException("unable to read pigs manifest file", e);
        } finally {
            if (jar != null) {
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
     * @param arg input option
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
     * @param arg
     * @return true - if arg is h/-h/help/-help, or return false
     */
    private static boolean isHelpOption(String string) {
        return "h".equalsIgnoreCase(string)
                || "-h".equalsIgnoreCase(string)
                || "help".equalsIgnoreCase(string)
                || "-help".equalsIgnoreCase(string);
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