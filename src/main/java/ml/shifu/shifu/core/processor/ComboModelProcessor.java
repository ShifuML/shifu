/*
 * Copyright [2013-2015] PayPal Software Foundation
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

import ml.shifu.shifu.combo.ColumnFile;
import ml.shifu.shifu.container.obj.*;
import ml.shifu.shifu.core.validator.ModelInspector;
import ml.shifu.shifu.executor.ExecutorManager;
import ml.shifu.shifu.executor.ProcessManager;
import ml.shifu.shifu.fs.PathFinder;
import ml.shifu.shifu.fs.SourceFile;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.Environment;
import ml.shifu.shifu.util.JSONUtils;
import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;

/**
 * Created by zhanhu on 12/5/16.
 */
public class ComboModelProcessor extends BasicModelProcessor implements Processor {

    private static Logger LOG = LoggerFactory.getLogger(ComboModelProcessor.class);

    public enum ComboStep {
        NEW, INIT, RUN, EVAL
    }

    public static final String ALG_DELIMITER = ",";
    public static final String SCORE_FIELD = "mean";

    private ComboStep comboStep;
    private String algorithms;
    private boolean isToShuffleData;
    private boolean isToResume;
    private int comboMaxRetryTimes = 3;

    private List<ModelTrainConf.ALGORITHM> comboAlgs;
    private ComboModelTrain comboModelTrain;

    private ExecutorManager<Integer> excutorManager = new ExecutorManager<Integer>();

    public ComboModelProcessor(ComboStep comboStep) {
        this.comboStep = comboStep;
    }

    public ComboModelProcessor(ComboStep comboStep, String algorithms) {
        this(comboStep);
        this.algorithms = algorithms;
        this.comboMaxRetryTimes = Environment.getInt("shifu.combo.max.retry", 3);
    }

    @Override
    public int run() throws Exception {
        LOG.info("Start to run combo, step - {}", this.comboStep);

        int status = 0;

        setUp(ModelInspector.ModelStep.COMBO);

        if ((status = validate()) > 0) {
            LOG.error("Validation Fail.");
            return status;
        }

        switch (comboStep) {
            case NEW:
                status = createNewCombo();
                break;
            case INIT:
                this.comboModelTrain = loadComboTrain();
                status = initComboModels();
                break;
            case RUN:
                this.comboModelTrain = loadComboTrain();
                status = runComboModels();
                break;
            case EVAL:
                this.comboModelTrain = loadComboTrain();
                status = evalComboModels();
                break;
        }

        clearUp(ModelInspector.ModelStep.COMBO);
        return status;
    }

    public void setToShuffleData(boolean toShuffleData) {
        this.isToShuffleData = toShuffleData;
    }

    public void setToResume(boolean toResume) {
        this.isToResume = toResume;
    }

    @Override
    public void clearUp(ModelInspector.ModelStep modelStep) {
        this.excutorManager.forceShutDown();
    }

    /**
     * Create ComboTrain.json according algorithm list.
     * ModelTrain configuration is set from template.
     *
     * @return if create new combo successful
     */
    private int createNewCombo() {
        ComboModelTrain comboModelTrain = new ComboModelTrain();
        List<SubTrainConf> subTrainConfList = new ArrayList<SubTrainConf>(this.comboAlgs.size());
        for (int i = 0; i < this.comboAlgs.size(); i++) {
            subTrainConfList.add(createSubTrainConf(this.comboAlgs.get(i)));
        }
        comboModelTrain.setSubTrainConfList(subTrainConfList);
        return saveComboTrain(comboModelTrain);
    }

    /**
     * Create folder for sub-models, and create related files for sub-models.
     * All settings in sub-model will use parent model as reference.
     *
     * @return 0 successful, otherwise failed
     * @throws IOException any io exception
     */
    private int initComboModels() throws IOException {
        if (this.comboModelTrain == null) {
            LOG.error("ComboModelTrain doesn't exists.");
            return 1;
        }

        String[] subModelNames = new String[this.comboModelTrain.getSubTrainConfList().size() - 1];

        for (int i = 0; i < this.comboModelTrain.getSubTrainConfList().size() - 1; i++) {
            SubTrainConf subTrainConf = this.comboModelTrain.getSubTrainConfList().get(i);
            String subModelName = genSubModelName(i, subTrainConf);

            // 0) save all sub model names, it will used as variables for assemble model
            subModelNames[i] = subModelName;

            // 1) create folder for sub-model
            new File(subModelName).mkdirs();

            // 2) create ModelConfig
            ModelConfig subModelConfig = this.modelConfig.clone();
            subModelConfig.getBasic().setName(subModelName);
            subModelConfig.setStats(subTrainConf.getModelStatsConf());
            subModelConfig.setNormalize(subTrainConf.getModelNormalizeConf());
            subModelConfig.setVarSelect(subTrainConf.getModelVarSelectConf());
            subModelConfig.setTrain(subTrainConf.getModelTrainConf());

            List<EvalConfig> subEvalConfigs = new ArrayList<EvalConfig>();
            for ( EvalConfig eval : modelConfig.getEvals() ) {
                if ( !eval.getName().equalsIgnoreCase("EvalTrain") ) {
                    subEvalConfigs.add(eval.clone());
                }
            }
            subModelConfig.setEvals(subEvalConfigs);

            // 3) save to sub-model's ModelConfig.json
            saveModelConfig(subModelName, subModelConfig);
            FileUtils.copyDirectory(new File(Constants.COLUMN_META_FOLDER_NAME),
                    new File(subModelName, Constants.COLUMN_META_FOLDER_NAME));
        }

        EvalConfig evalTrain = modelConfig.getEvalConfigByName("EvalTrain");
        if (evalTrain == null) {
            evalTrain = new EvalConfig();
            evalTrain.setName("EvalTrain");
            evalTrain.setDataSet(modelConfig.getDataSet().cloneRawSourceData());
            modelConfig.getEvals().add(evalTrain);
            saveModelConfig();
        }

        // create ModelConfig for assemble model
        ModelConfig assembleModelConfig = this.modelConfig.clone();
        String assembleModelName = genAssembleModelName(this.modelConfig.getModelSetName());
        new File(assembleModelName).mkdirs();

        assembleModelConfig.setModelSetName(assembleModelName);
        if ( RawSourceData.SourceType.HDFS.equals(evalTrain.getDataSet().getSource()) ) {
            assembleModelConfig.getDataSet().setDataPath(pathFinder.getEvalScorePath(evalTrain));
            assembleModelConfig.getDataSet().setHeaderPath(pathFinder.getEvalScoreHeaderPath(evalTrain));
        } else if (RawSourceData.SourceType.LOCAL.equals(evalTrain.getDataSet().getSource()) ) {
            File evalScoreFile = new File(pathFinder.getEvalScorePath(evalTrain));
            assembleModelConfig.getDataSet().setDataPath(evalScoreFile.getAbsolutePath());
            assembleModelConfig.getDataSet().setHeaderPath("");
        }
        assembleModelConfig.getDataSet().setDataDelimiter("|");
        assembleModelConfig.getDataSet().setHeaderDelimiter("|");
        assembleModelConfig.getDataSet().setCategoricalColumnNameFile(null);
        FileUtils.copyDirectory(new File(Constants.COLUMN_META_FOLDER_NAME),
                new File(assembleModelName, Constants.COLUMN_META_FOLDER_NAME));

        // create force selects file
        String forceSelectNames = createModelNamesFile(assembleModelName, assembleModelName + ".forceselect", null,
                CommonUtils.genPigFieldName(subModelNames));
        assembleModelConfig.getVarSelect().setForceSelectColumnNameFile(forceSelectNames);
        assembleModelConfig.getVarSelect().setForceRemoveColumnNameFile(null);
        assembleModelConfig.getVarSelect().setForceEnable(true);
        assembleModelConfig.getVarSelect().setFilterNum(subModelNames.length);
        assembleModelConfig.getVarSelect().setFilterEnable(true);

        List<EvalConfig> assembleEvalConfigs = new ArrayList<EvalConfig>();
        for ( EvalConfig eval: modelConfig.getEvals() ) {
            if ( !eval.getName().equalsIgnoreCase("EvalTrain") ) {
                EvalConfig assembleEval = eval.clone();
                if ( RawSourceData.SourceType.HDFS.equals(eval.getDataSet().getSource()) ) {
                    assembleEval.getDataSet().setDataPath(pathFinder.getEvalScorePath(eval));
                    assembleEval.getDataSet().setHeaderPath(pathFinder.getEvalScoreHeaderPath(eval));
                } else if (RawSourceData.SourceType.LOCAL.equals(eval.getDataSet().getSource()) ) {
                    File evalScoreFile = new File(pathFinder.getEvalScorePath(eval));
                    assembleEval.getDataSet().setDataPath(evalScoreFile.getAbsolutePath());
                    assembleEval.getDataSet().setHeaderPath("");
                }

                assembleEval.getDataSet().setDataDelimiter("|");
                assembleEval.getDataSet().setHeaderDelimiter("|");
                assembleEvalConfigs.add(assembleEval);
            }
        }
        assembleModelConfig.setEvals(assembleEvalConfigs);

        // create folder and save ModelConfig.json
        new File(assembleModelName).mkdirs();
        saveModelConfig(assembleModelName, assembleModelConfig);

        return 0;
    }

    /**
     * Start to train combo models
     * 1) train sub-models and evaluate sub-model train-eval set (train data as evaluation data)
     * 2) join train-eval set output for training assemble model
     * 3) train assemble model
     *
     * @return status of execution
     * 0 - success
     * others - fail
     * @throws IOException any io exception
     */
    public int runComboModels() throws IOException {
        int status = 0;

        // 1. train sub models and evaluate sub models using training data
        List<Callable<Integer>> tasks = new ArrayList<Callable<Integer>>();
        for (int i = 0; i < this.comboModelTrain.getSubTrainConfList().size() - 1; i++) {
            SubTrainConf subTrainConf = this.comboModelTrain.getSubTrainConfList().get(i);
            String subModelName = genSubModelName(i, subTrainConf);
            Callable<Integer> task = createSubModelTrainTasks(subModelName, genEvalTrainName());
            if (task != null) {
                tasks.add(task);
            }
        }
        if (hasFailTaskResults(this.excutorManager.submitTasksAndRetryIfFail(tasks, this.comboMaxRetryTimes))) {
            LOG.error("There are errors when training and evaluating sub-models. Please check log.");
            return 1;
        }

        // copy sub model specs to main model directory
        File modelsDir = new File(Constants.MODELS);
        modelsDir.mkdirs();

        for (int i = 0; i < this.comboModelTrain.getSubTrainConfList().size() - 1; i++) {
            SubTrainConf subTrainConf = this.comboModelTrain.getSubTrainConfList().get(i);
            String subModelName = genSubModelName(i, subTrainConf);

            File subModelsDir = new File(modelsDir, subModelName);
            subModelsDir.mkdirs();

            FileUtils.copyFile(new File(subModelName, Constants.MODEL_CONFIG_JSON_FILE_NAME),
                    new File(subModelsDir, Constants.MODEL_CONFIG_JSON_FILE_NAME));
            FileUtils.copyFile(new File(subModelName, Constants.COLUMN_CONFIG_JSON_FILE_NAME),
                    new File(subModelsDir, Constants.COLUMN_CONFIG_JSON_FILE_NAME));
            File[] modelFiles = (new File(subModelName, Constants.MODELS)).listFiles();
            for (File modelFile : modelFiles) {
                FileUtils.copyFile(modelFile, new File(new File(Constants.MODELS, subModelName), modelFile.getName()));
            }
        }

        ProcessManager.runShellProcess(".",
                new String[][]{
                        new String[]{"shifu", "eval"}});

        // 4.2 run the whole process for assemble model
        ProcessManager.runShellProcess(genAssembleModelName(modelConfig.getModelSetName()),
                new String[][]{
                        new String[]{"shifu", "init"},
                        new String[]{"shifu", "stats"},
                        new String[]{"shifu", "norm"},
                        new String[]{"shifu", "varsel"},
                        new String[]{"shifu", "train"}});

        LOG.info("Finish to run combo train.");
        return status;
    }

    /**
     * Evaluate the Combo model performance
     * 1. Evaluate all evaluation sets in sub models;
     * 2. Join the evaluation result data for assemble model;
     * 3. Run evaluation for assemble model
     *
     * @return 0 success, otherwise failed
     * @throws IOException any io exception
     */
    private int evalComboModels() throws IOException {
        int status = 0;

        // 1. For all sub-models, evaluate all evaluation sets except training data set
        List<Callable<Integer>> tasks = new ArrayList<Callable<Integer>>();
        for (EvalConfig evalConfig : this.modelConfig.getEvals()) {
            if ( !evalConfig.getName().equalsIgnoreCase("EvalTrain") ) {
                tasks.addAll(createEvaluateTasks(evalConfig.getName()));
                if (hasFailTaskResults(this.excutorManager.submitTasksAndRetryIfFail(tasks, this.comboMaxRetryTimes))) {
                    LOG.error("Error occurred when evaluate sub-models. Please check log!");
                    return 1;
                }
                tasks.clear();
            }
        }

        LOG.info("Finish to eval combo train.");
        return status;
    }

    /**
     * get the eval data path for sub model
     *
     * @param subModelConfig - @ModelConfig for sub model
     * @param evalConfig     eval config
     * @return eval data path
     */
    private String getEvalDataPath(ModelConfig subModelConfig, EvalConfig evalConfig) {
        PathFinder pathFinder = new PathFinder(subModelConfig);
        String evalDataPath = pathFinder.getEvalScorePath(evalConfig);

        // if it is local, and the sub-model name as directory
        if (RawSourceData.SourceType.LOCAL.equals(evalConfig.getDataSet().getSource())) {
            evalDataPath = subModelConfig.getModelSetName() + File.separator + evalDataPath;
        }

        return evalDataPath;
    }

    /**
     * get @EvalConfig from @ModelConfig by evalset name
     *
     * @param mconfig model config
     * @param name    eval name
     * @return eval config instance
     */
    private EvalConfig getEvalConfigByName(ModelConfig mconfig, String name) {
        for (EvalConfig evalConfig : mconfig.getEvals()) {
            if (evalConfig.getName().equalsIgnoreCase(name)) {
                return evalConfig;
            }
        }
        return null;
    }

    /**
     * Create train and eval task for sub-model
     *
     * @param subModelName sub model name
     * @param evalSetName  eval set name
     * @return callable instance
     */
    private Callable<Integer> createSubModelTrainTasks(final String subModelName, final String evalSetName)
            throws IOException {
        return new Callable<Integer>() {
            @Override
            public Integer call() {
                try {
                    if (isToShuffleData) {
                        return ProcessManager.runShellProcess(subModelName, new String[][]{
                                new String[]{"shifu", "init"},
                                new String[]{"shifu", "stats"},
                                new String[]{"shifu", "norm", "-shuffle"},
                                new String[]{"shifu", "varsel"},
                                new String[]{"shifu", "train", "-shuffle"}});
                    } else {
                        return ProcessManager.runShellProcess(subModelName, new String[][]{
                                new String[]{"shifu", "init"},
                                new String[]{"shifu", "stats"},
                                new String[]{"shifu", "norm",},
                                new String[]{"shifu", "varsel"},
                                new String[]{"shifu", "train"}});
                    }
                } catch (IOException e) {
                    LOG.error("Fail to run commands.", e);
                    return 1;
                }
            }
        };
    }

    /**
     * Create evaluation tasks for all sub-models
     *
     * @param evalName - the evalset to evaluate
     * @return list of callable instance
     * @throws IOException any io exception
     */
    private List<Callable<Integer>> createEvaluateTasks(final String evalName) throws IOException {
        List<Callable<Integer>> tasks = new ArrayList<Callable<Integer>>();

        for (int i = 0; i < this.comboModelTrain.getSubTrainConfList().size(); i++) {
            SubTrainConf subTrainConf = this.comboModelTrain.getSubTrainConfList().get(i);
            String evalModelName = null;
            if (i == this.comboModelTrain.getSubTrainConfList().size() - 1) {
                evalModelName = genAssembleModelName(modelConfig.getModelSetName());
            } else {
                evalModelName = genSubModelName(i, subTrainConf);
            }

            final String subModelName = evalModelName;
            tasks.add(new Callable<Integer>() {
                @Override
                public Integer call() {
                    try {
                        return ProcessManager.runShellProcess(subModelName,
                                new String[][]{new String[]{"shifu", "eval", "-run", evalName}});
                    } catch (IOException e) {
                        LOG.error("Fail to run commands.", e);
                        return 1;
                    }
                }
            });
        }

        return tasks;
    }

    /**
     * Shifu combo processor validation
     *
     * @return 0 - success
     * other - fail
     */
    private int validate() {
        if (ComboStep.NEW.equals(this.comboStep)) {
            return validate(this.algorithms);
        } else {
            File comboTrainFile = new File(Constants.COMBO_CONFIG_JSON_FILE_NAME);
            if (!comboTrainFile.exists()) {
                LOG.error("{} doesn't exist. Please run `shifu combo -new <algorithms>` firstly.",
                        Constants.COMBO_CONFIG_JSON_FILE_NAME);
                return 1;
            }
        }
        return 0;
    }

    /**
     * Validate the algorithms from user's input
     *
     * @param algorithms - algorithm list that user want to combo
     * @return 0 - success
     * other - fail
     */
    private int validate(String algorithms) {
        if (StringUtils.isBlank(algorithms)) {
            LOG.error("The combo algorithms should not be empty");
            return 1;
        }

        String[] algs = algorithms.split(ALG_DELIMITER);
        if (algs.length < 3) {
            LOG.error("At least, you should have 2 basic algorithms, and 1 assembling algorithm.");
            return 2;
        }

        this.comboAlgs = new ArrayList<ModelTrainConf.ALGORITHM>();
        for (String alg : algs) {
            try {
                ModelTrainConf.ALGORITHM algorithm = ModelTrainConf.ALGORITHM.valueOf(alg);
                if (algorithm == null) {
                    LOG.error("Unsupported algorithm - {}", alg);
                    return 3;
                }
                this.comboAlgs.add(algorithm);
            } catch (Throwable t) {
                LOG.error("Unsupported algorithm - {}", alg);
                return 3;
            }
        }
        return 0;
    }

    /**
     * Create @VarTrainConf according the @ModelTrainConf.ALGORITHM
     *
     * @param alg - the algorithm, see @ModelTrainConf.ALGORITHM
     * @return train config instance
     */
    private SubTrainConf createSubTrainConf(ModelTrainConf.ALGORITHM alg) {
        SubTrainConf subTrainConf = new SubTrainConf();
        subTrainConf.setModelStatsConf(createModelStatsConf(alg));
        subTrainConf.setModelNormalizeConf(createModelNormalizeConf(alg));
        subTrainConf.setModelVarSelectConf(createModelVarSelectConf(alg));
        subTrainConf.setModelTrainConf(createModelTrainConf(alg));
        return subTrainConf;
    }

    private ModelStatsConf createModelStatsConf(ModelTrainConf.ALGORITHM alg) {
        ModelStatsConf statsConf = new ModelStatsConf();
        if (ModelTrainConf.ALGORITHM.NN.equals(alg) || ModelTrainConf.ALGORITHM.LR.equals(alg)) {
            statsConf.setBinningAlgorithm(ModelStatsConf.BinningAlgorithm.DynamicBinning);
            statsConf.setBinningMethod(ModelStatsConf.BinningMethod.EqualTotal);
            statsConf.setMaxNumBin(20);
        } else if (ModelTrainConf.ALGORITHM.RF.equals(alg) || ModelTrainConf.ALGORITHM.GBT.equals(alg)) {
            statsConf.setBinningAlgorithm(ModelStatsConf.BinningAlgorithm.SPDTI);
            statsConf.setBinningMethod(ModelStatsConf.BinningMethod.EqualPositive);
            statsConf.setMaxNumBin(20);
        }
        return statsConf;
    }

    private ModelNormalizeConf createModelNormalizeConf(ModelTrainConf.ALGORITHM alg) {
        ModelNormalizeConf normalizeConf = new ModelNormalizeConf();
        normalizeConf.setNormType(ModelNormalizeConf.NormType.WOE);
        normalizeConf.setSampleNegOnly(false);
        normalizeConf.setSampleRate(1.0);
        return normalizeConf;
    }

    private ModelVarSelectConf createModelVarSelectConf(ModelTrainConf.ALGORITHM alg) {
        ModelVarSelectConf varSelectConf = new ModelVarSelectConf();
        varSelectConf.setFilterNum(20);
        if (ModelTrainConf.ALGORITHM.NN.equals(alg) || ModelTrainConf.ALGORITHM.LR.equals(alg)) {
            varSelectConf.setFilterBy("IV");
        } else if (ModelTrainConf.ALGORITHM.RF.equals(alg) || ModelTrainConf.ALGORITHM.GBT.equals(alg)) {
            varSelectConf.setFilterBy("KS");
        }
        return varSelectConf;
    }

    /**
     * Create @ModelTrainConf according the @ModelTrainConf.ALGORITHM
     *
     * @param alg - the algorithm, see @ModelTrainConf.ALGORITHM
     * @return train config instance
     */
    private ModelTrainConf createModelTrainConf(ModelTrainConf.ALGORITHM alg) {
        ModelTrainConf trainConf = new ModelTrainConf();

        trainConf.setAlgorithm(alg.name());
        trainConf.setEpochsPerIteration(1);
        trainConf.setParams(ModelTrainConf.createParamsByAlg(alg, trainConf));
        trainConf.setNumTrainEpochs(100);
        if (ModelTrainConf.ALGORITHM.NN.equals(alg)) {
            trainConf.setNumTrainEpochs(200);
        } else if (ModelTrainConf.ALGORITHM.SVM.equals(alg)) {
            trainConf.setNumTrainEpochs(100);
        } else if (ModelTrainConf.ALGORITHM.RF.equals(alg)) {
            trainConf.setNumTrainEpochs(40000);
        } else if (ModelTrainConf.ALGORITHM.GBT.equals(alg)) {
            trainConf.setNumTrainEpochs(40000);
        } else if (ModelTrainConf.ALGORITHM.LR.equals(alg)) {
            trainConf.setNumTrainEpochs(100);
        }
        trainConf.setBaggingWithReplacement(true);

        return trainConf;
    }

    /**
     * Get evaluation output file format
     *
     * @param runMode run mode
     * @return file type
     */
    private ColumnFile.FileType genEvalFileType(ModelBasicConf.RunMode runMode) {
        return (ModelBasicConf.RunMode.MAPRED.equals(runMode) ? ColumnFile.FileType.PIGSTORAGE
                : ColumnFile.FileType.CSV);
    }

    /**
     * Generate sub model name
     *
     * @param i            - sequence to keep unique
     * @param subTrainConf train config
     * @return sub model name
     */
    private String genSubModelName(int i, SubTrainConf subTrainConf) {
        return this.modelConfig.getBasic().getName() + "_" + subTrainConf.getModelTrainConf().getAlgorithm() + "_" + i;
    }

    /**
     * Generate assembel model name
     *
     * @param modelName model name
     * @return assemble model bane
     */
    private String genAssembleModelName(String modelName) {
        return modelName + "_" + Constants.COMBO_ASSEMBLE;
    }

    /**
     * Generate train data evaluation set name
     *
     * @return eval train name
     */
    private String genEvalTrainName() {
        return Constants.COMBO_EVAL_TRAIN;
    }

    /**
     * Save ComboTrain.json into local directory
     *
     * @param comboModelTrain combo model train instance
     * @return 0 success, otherwise failed
     */
    private int saveComboTrain(ComboModelTrain comboModelTrain) {
        try {
            JSONUtils.writeValue(new File(Constants.COMBO_CONFIG_JSON_FILE_NAME), comboModelTrain);
        } catch (Exception e) {
            LOG.error("Fail to save ComboModelTrain object to ComboTrain.json");
            return 1;
        }
        return 0;
    }

    /**
     * Load ComboModelTrain from ComboTrain.json
     *
     * @return combo model train instance, null if exception
     */
    private ComboModelTrain loadComboTrain() {
        try {
            return JSONUtils.readValue(new File(Constants.COMBO_CONFIG_JSON_FILE_NAME), ComboModelTrain.class);
        } catch (Exception e) {
            LOG.error("Fail to load ComboModelTrain object from ComboTrain.json");
            return null;
        }
    }

    /**
     * Clone @ColumnConfig list for sub-models
     *
     * @param columnConfigList column config list
     * @return cloned column config list
     */
    private List<ColumnConfig> cloneColumnConfigs(List<ColumnConfig> columnConfigList) {
        List<ColumnConfig> columnConfigs = new ArrayList<ColumnConfig>();
        for (ColumnConfig columnConfig : columnConfigList) {
            columnConfigs.add(columnConfig.clone());
        }
        return columnConfigs;
    }

    /**
     * Save ModelConfig into some folder
     *
     * @param folder      - folder to host ModelConfig.json
     * @param modelConfig model config instance
     * @throws IOException any io exception
     */
    private void saveModelConfig(String folder, ModelConfig modelConfig) throws IOException {
        JSONUtils.writeValue(new File(folder + File.separator + Constants.MODEL_CONFIG_JSON_FILE_NAME), modelConfig);
    }

    /**
     * Save ColumnConfig list into some folder
     *
     * @param folder        - folder to host ColumnConfig.json
     * @param columnConfigs column config list
     * @throws IOException any io exception
     */
    private void saveColumnConfigList(String folder, List<ColumnConfig> columnConfigs) throws IOException {
        JSONUtils.writeValue(new File(folder + File.separator + Constants.COLUMN_CONFIG_JSON_FILE_NAME), columnConfigs);
    }

    /**
     * Check whether there is any fail results in the list
     *
     * @param taskResults
     * @return true - there is any fail result
     * false - no fail task
     */
    private boolean hasFailTaskResults(List<Integer> taskResults) {
        if (CollectionUtils.isNotEmpty(taskResults)) {
            for (Integer result : taskResults) {
                if (result == null || result != 0) {
                    LOG.error("Found some abnormal result - {}", result);
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Create configuration file for sub-model
     * if the configuration for parent model exists, it will copy that parent configuration firstly
     * and append new content.
     *
     * @param subModelName    sub model name
     * @param namesPrefix     prefix of name
     * @param parentNamesFile parent names of files
     * @param varNames        var names
     * @return model file name
     * @throws IOException any io exception
     */
    private String createModelNamesFile(String subModelName, String namesPrefix, String parentNamesFile,
                                        String... varNames) throws IOException {
        String modelNamesCfg = namesPrefix + ".names";
        File mnFile = new File(subModelName + File.separator + modelNamesCfg);

        // copy existing meta file
        if (StringUtils.isNotBlank(parentNamesFile)) {
            FileUtils.copyFile(new File(parentNamesFile), mnFile);
        }

        // append uid column as meta
        FileWriter writer = new FileWriter(mnFile);
        try {
            for (String var : varNames) {
                writer.append(var + "\n");
            }
        } catch (IOException e) {
            // skip it
        } finally {
            writer.close();
        }

        return modelNamesCfg;
    }
}
