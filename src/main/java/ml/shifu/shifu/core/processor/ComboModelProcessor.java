package ml.shifu.shifu.core.processor;

import ml.shifu.shifu.combo.ColumnFile;
import ml.shifu.shifu.combo.DataMerger;
import ml.shifu.shifu.container.obj.*;
import ml.shifu.shifu.core.validator.ModelInspector;
import ml.shifu.shifu.executor.ExecutorManager;
import ml.shifu.shifu.executor.ProcessManager;
import ml.shifu.shifu.fs.PathFinder;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.JSONUtils;
import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
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

    private List<ModelTrainConf.ALGORITHM> comboAlgs;
    private ComboModelTrain comboModelTrain;

    private ExecutorManager<Integer> excutorManager = new ExecutorManager<Integer>();

    public ComboModelProcessor(ComboStep comboStep) {
        this.comboStep = comboStep;
    }

    public ComboModelProcessor(ComboStep comboStep, String algorithms) {
        this(comboStep);
        this.algorithms = algorithms;
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

    @Override
    public void clearUp(ModelInspector.ModelStep modelStep) {
        this.excutorManager.forceShutDown();
    }

    /**
     * Create ComboTrain.json according algorithm list.
     * ModelTrain configuration is set from template.
     *
     * @return
     */
    private int createNewCombo() {
        ComboModelTrain comboModelTrain = new ComboModelTrain();
        comboModelTrain.setUidColumnName("");

        List<VarTrainConf> varTrainConfList = new ArrayList<VarTrainConf>(this.comboAlgs.size() - 1);
        for (int i = 0; i < this.comboAlgs.size() - 1; i++) {
            varTrainConfList.add(createVarTrainConf(this.comboAlgs.get(i)));
        }
        comboModelTrain.setVarTrainConfList(varTrainConfList);

        comboModelTrain.setFusionModelTrainConf(createModelTrainConf(this.comboAlgs.get(this.comboAlgs.size() - 1)));

        return saveComboTrain(comboModelTrain);
    }


    /**
     * Create folder for sub-models, and create related files for sub-models.
     * All settings in sub-model will use parent model as reference.
     *
     * @return
     * @throws IOException
     */
    private int initComboModels() throws IOException {
        if (this.comboModelTrain == null) {
            LOG.error("ComboModelTrain doesn't exists.");
            return 1;
        }

        String[] subModelNames = new String[this.comboModelTrain.getVarTrainConfList().size()];

        for (int i = 0; i < this.comboModelTrain.getVarTrainConfList().size(); i++) {
            VarTrainConf varTrainConf = this.comboModelTrain.getVarTrainConfList().get(i);
            String subModelName = genSubModelName(i, varTrainConf);

            // 0) save all sub model names, it will used as variables for assemble model
            subModelNames[i] = subModelName;

            // 1) create folder for sub-model
            new File(subModelName).mkdirs();

            // 2) create ModelConfig
            ModelConfig subModelConfig = this.modelConfig.clone();
            subModelConfig.getBasic().setName(subModelName);
            subModelConfig.setTrain(varTrainConf.getModelTrainConf());
            subModelConfig.getTrain().setCustomPaths(new HashMap<String, String>());

            // 2.1) set normalized data path
            String normalizedPath = pathFinder.getNormalizedDataPath();
            if (RawSourceData.SourceType.LOCAL.equals(modelConfig.getDataSet().getSource())) {
                normalizedPath = new File(normalizedPath).getAbsolutePath();
            }
            subModelConfig.getTrain().getCustomPaths().put(Constants.KEY_NORMALIZED_DATA_PATH, normalizedPath);

            // 2.2) update selected variables, if user want to use different variables list for sub-models
            List<ColumnConfig> columnConfigs = this.columnConfigList;
            if (CollectionUtils.isNotEmpty(varTrainConf.getVariables())) {
                Set<String> selectedVars = new HashSet<String>(varTrainConf.getVariables());

                columnConfigs = cloneColumnConfigs(this.columnConfigList);
                for (ColumnConfig columnConfig : columnConfigs) {
                    columnConfig.setFinalSelect(false);
                    if (selectedVars.contains(columnConfig.getColumnName())) {
                        columnConfig.setFinalSelect(true);
                    }
                }
            }

            // 2.3) update evaluation set to add uid column as meta
            for (EvalConfig evalConfig : subModelConfig.getEvals()) {
                String scoreMetaFileName = createModelNamesFile(
                        subModelName,
                        evalConfig.getName() + ".score.meta",
                        evalConfig.getScoreMetaColumnNameFile(),
                        this.genEvalScoreMetaVars(null));
                evalConfig.setScoreMetaColumnNameFile(scoreMetaFileName);
            }

            // 2.4) create evaluation set for training data, it will be used as training data for assemble model
            EvalConfig trainEval = new EvalConfig();
            trainEval.setName(genEvalTrainName());
            trainEval.setDataSet(subModelConfig.getDataSet().cloneRawSourceData());
            String scoreMetaFileName = createModelNamesFile(
                    subModelName,
                    trainEval.getName() + ".score.meta",
                    null,
                    this.genEvalScoreMetaVars(this.modelConfig.getDataSet().getWeightColumnName()));
            trainEval.setScoreMetaColumnNameFile(scoreMetaFileName);
            // add train data as evaluation set, this is for assemble model
            subModelConfig.getEvals().add(trainEval);

            // 3) save to sub-model's ModelConfig.json and ColumnConfig.json
            saveModelConfig(subModelName, subModelConfig);
            saveColumnConfigList(subModelName, columnConfigs);
        }

        // create ModelConfig for assemble model
        ModelConfig assembleModelConfig = this.modelConfig.clone();
        String assembleModelName = genAssembleModelName(this.modelConfig.getModelSetName());
        new File(assembleModelName).mkdirs();

        assembleModelConfig.setModelSetName(assembleModelName);
        assembleModelConfig.getDataSet().setCategoricalColumnNameFile(null);
        assembleModelConfig.getVarSelect().setForceRemoveColumnNameFile(null);

        // create force selects file
        String forceSelectNames = createModelNamesFile(
                assembleModelName,
                assembleModelName + ".forceselect",
                null,
                CommonUtils.genPigFieldName(subModelNames));
        assembleModelConfig.getVarSelect().setForceSelectColumnNameFile(forceSelectNames);
        assembleModelConfig.getVarSelect().setForceEnable(true);
        assembleModelConfig.getVarSelect().setFilterNum(subModelNames.length);
        assembleModelConfig.getVarSelect().setFilterEnable(true);

        // create metas file
        String metaColumnNames = createModelNamesFile(
                assembleModelName,
                assembleModelName + ".meta",
                null,
                this.comboModelTrain.getUidColumnName());
        assembleModelConfig.getDataSet().setMetaColumnNameFile(metaColumnNames);
        assembleModelConfig.setTrain(this.comboModelTrain.getFusionModelTrainConf());

        // update evaluation set to add uid column as meta
        for (EvalConfig evalConfig : assembleModelConfig.getEvals()) {
            String scoreMetaFileName = createModelNamesFile(
                    assembleModelName,
                    evalConfig.getName() + ".score.meta",
                    evalConfig.getScoreMetaColumnNameFile(),
                    this.genEvalScoreMetaVars(null));
            evalConfig.setScoreMetaColumnNameFile(scoreMetaFileName);
        }

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
     */
    public int runComboModels() {
        // 1. train sub models and evaluate sub models using training data
        int status = 0;

        List<Callable<Integer>> tasks = new ArrayList<Callable<Integer>>();
        for (int i = 0; i < this.comboModelTrain.getVarTrainConfList().size(); i++) {
            VarTrainConf varTrainConf = this.comboModelTrain.getVarTrainConfList().get(i);
            String subModelName = genSubModelName(i, varTrainConf);
            tasks.add(createTrainAndEvalTasks(subModelName, genEvalTrainName()));
        }
        if (hasFailTaskResults(this.excutorManager.submitTasksAndWaitResults(tasks))) {
            LOG.error("There are errors when training and evaluating sub-models. Please check log.");
            return 1;
        }

        // 2. merge all train-evaluation data and prepare it as training data for assemble model

        // 2.1 prepare the data and information for data merge
        LOG.info("Start to merge train-evaluation data for assemble model.");
        String assembleTrainData = this.pathFinder.getSubModelsAssembleTrainData();
        DataMerger merger = new DataMerger(this.modelConfig.getBasic().getRunMode(),
                this.comboModelTrain.getUidColumnName(), assembleTrainData);

        String evalName = genEvalTrainName();
        try {
            merger.addColumnFileList(genSubModelEvalColumnFiles(evalName));
        } catch (IOException e) {
            LOG.error("Fatal error - fail to add column files to merge.", e);
            return 1;
        }

        // 2.2 run the data merge
        try {
            merger.doMerge();
        } catch (IOException e) {
            LOG.error("Fail to merge the data.", e);
            return 1;
        }

        // 3. run the assemble model
        // 3.1 set the train data and header for assemble model
        try {
            ModelConfig assembleModelConfig = CommonUtils.loadModelConfig(
                    genAssembleModelName(this.modelConfig.getModelSetName())
                            + File.separator + Constants.MODEL_CONFIG_JSON_FILE_NAME,
                    RawSourceData.SourceType.LOCAL);
            assembleModelConfig.getDataSet().setHeaderDelimiter("|");
            assembleModelConfig.getDataSet().setDataDelimiter("|");

            if (ModelBasicConf.RunMode.LOCAL.equals(this.modelConfig.getBasic().getRunMode())) {
                File file = new File(assembleTrainData + File.separator + ".pig_header");
                assembleModelConfig.getDataSet().setHeaderPath(file.getAbsolutePath());
                file = new File(assembleTrainData);
                assembleModelConfig.getDataSet().setDataPath(file.getAbsolutePath());
            } else {
                assembleModelConfig.getDataSet().setHeaderPath(assembleTrainData + File.separator + ".pig_header");
                assembleModelConfig.getDataSet().setDataPath(assembleTrainData);
            }

            saveModelConfig(genAssembleModelName(this.modelConfig.getModelSetName()), assembleModelConfig);
        } catch (IOException e) {
            LOG.error("Fail to ModelConfig for assemble model.", e);
            return 1;
        }

        // 4.2 run the whole process for assemble model
        try {
            status = ProcessManager.runShellProcess(genAssembleModelName(this.modelConfig.getModelSetName()),
                    new String[][]{
                            new String[]{"shifu", "init"},
                            new String[]{"shifu", "stats"},
                            new String[]{"shifu", "norm"},
                            new String[]{"shifu", "varsel"},
                            new String[]{"shifu", "train"}
                    });
        } catch (IOException e) {
            LOG.error("Fail to run assemble model.", e);
            status = 1;
        }

        LOG.info("Finish to run combo train.");
        return status;
    }

    /**
     * Evaluate the Combo model performance
     * 1. Evaluate all evaluation sets in sub models;
     * 2. Join the evaluation result data for assemble model;
     * 3. Run evaluation for assemble model
     *
     * @return
     * @throws IOException
     */
    private int evalComboModels() throws IOException {
        int status = 0;

        // 1. For all sub-models, evaluate all evaluation sets except training data set
        List<Callable<Integer>> tasks = new ArrayList<Callable<Integer>>();
        for (EvalConfig evalConfig : this.modelConfig.getEvals()) {
            tasks.addAll(createEvaluateTasks(evalConfig.getName()));
            if (hasFailTaskResults(this.excutorManager.submitTasksAndWaitResults(tasks))) {
                LOG.error("Error occurred when evaluate sub-models. Please check log!");
                return 1;
            }
            tasks.clear();
        }

        // 2. Join evaluation data sets for assemble model
        ModelConfig assembleModelConfig = CommonUtils.loadModelConfig(
                genAssembleModelName(this.modelConfig.getModelSetName())
                        + File.separator + Constants.MODEL_CONFIG_JSON_FILE_NAME,
                RawSourceData.SourceType.LOCAL);

        // clear old tasks before adding new tasks
        for (EvalConfig evalConfig : this.modelConfig.getEvals()) {
            String assembleEvalData = this.pathFinder.getSubModelsAssembleEvalData(
                    evalConfig.getName(), evalConfig.getDataSet().getSource());
            final DataMerger dataMerger = new DataMerger(this.modelConfig.getBasic().getRunMode(),
                    this.comboModelTrain.getUidColumnName(), assembleEvalData);

            dataMerger.addColumnFileList(genSubModelEvalColumnFiles(evalConfig.getName()));

            // 2.1 run the data merge
            tasks.add(new Callable<Integer>() {
                @Override
                public Integer call() {
                    try {
                        return (dataMerger.doMerge() ? 0 : 1);
                    } catch (IOException e) {
                        LOG.error("Fail to merge the data.", e);
                        return 1;
                    }
                }
            });

            // 2.2 set the train data and header for assemble model
            EvalConfig assembleEvalConfig = getEvalConfigByName(assembleModelConfig, evalConfig.getName());
            assembleEvalConfig.getDataSet().setHeaderDelimiter("|");
            assembleEvalConfig.getDataSet().setDataDelimiter("|");

            if (ModelBasicConf.RunMode.LOCAL.equals(this.modelConfig.getBasic().getRunMode())) {
                File file = new File(assembleEvalData + File.separator + ".pig_header");
                assembleEvalConfig.getDataSet().setHeaderPath(file.getAbsolutePath());
                file = new File(assembleEvalData);
                assembleEvalConfig.getDataSet().setDataPath(file.getAbsolutePath());
            } else {
                assembleEvalConfig.getDataSet().setHeaderPath(assembleEvalData + File.separator + ".pig_header");
                assembleEvalConfig.getDataSet().setDataPath(assembleEvalData);
            }
        }

        if (hasFailTaskResults(this.excutorManager.submitTasksAndWaitResults(tasks))) {
            LOG.error("Error occurred when joining evaluation result. Please check log!");
            return 1;
        }

        saveModelConfig(genAssembleModelName(this.modelConfig.getModelSetName()), assembleModelConfig);

        // 3. run assemble model evaluation
        try {
            status = ProcessManager.runShellProcess(genAssembleModelName(this.modelConfig.getModelSetName()),
                    new String[][]{
                            new String[]{"shifu", "eval"}
                    });
        } catch (IOException e) {
            LOG.error("Fail to run assemble model.", e);
            status = 1;
        }

        LOG.info("Finish to eval combo train.");
        return status;
    }

    /**
     * iterate all evaluation sets and generate ColumnFiles for data join
     *
     * @param evalName
     * @return - ColumnFiles for some @evalName in all sub-models
     * @throws IOException
     */
    private List<ColumnFile> genSubModelEvalColumnFiles(String evalName) throws IOException {
        List<ColumnFile> columnFiles = new ArrayList<ColumnFile>();

        for (int i = 0; i < this.comboModelTrain.getVarTrainConfList().size(); i++) {
            VarTrainConf varTrainConf = this.comboModelTrain.getVarTrainConfList().get(i);

            String subModelName = genSubModelName(i, varTrainConf);
            ModelConfig subModelConfig = CommonUtils.loadModelConfig(
                    subModelName + File.separator + Constants.MODEL_CONFIG_JSON_FILE_NAME,
                    RawSourceData.SourceType.LOCAL);
            EvalConfig evalConfig = getEvalConfigByName(subModelConfig, evalName);

            Map<String, String> varsMapping = new HashMap<String, String>();
            varsMapping.put(SCORE_FIELD, CommonUtils.genPigFieldName(subModelName));
            String[] selectedVars = null;
            if (i == 0) {
                if (StringUtils.isNotBlank(evalConfig.getDataSet().getWeightColumnName())) {
                    selectedVars = new String[]{this.comboModelTrain.getUidColumnName(),
                            this.modelConfig.getTargetColumnName(), SCORE_FIELD,
                            evalConfig.getDataSet().getWeightColumnName()};
                } else {
                    selectedVars = new String[]{this.comboModelTrain.getUidColumnName(),
                            this.modelConfig.getTargetColumnName(), SCORE_FIELD};
                }
            } else {
                selectedVars = new String[]{SCORE_FIELD};
            }

            String evalDataPath = getEvalDataPath(subModelConfig, evalConfig);
            LOG.info("Include data - {} ...", evalDataPath);
            columnFiles.add(new ColumnFile(evalDataPath,
                    genEvalFileType(this.modelConfig.getBasic().getRunMode()), "|",
                    selectedVars, varsMapping));
        }

        return columnFiles;
    }

    /**
     * get the eval data path for sub model
     *
     * @param subModelConfig - @ModelConfig for sub model
     * @param evalConfig
     * @return
     */
    private String getEvalDataPath(ModelConfig subModelConfig, EvalConfig evalConfig) {
        PathFinder pathFinder = new PathFinder(subModelConfig);
        String evalDataPath = pathFinder.getEvalScorePath(evalConfig);

        // if it is local, and the sub-model name as directory
        if (RawSourceData.SourceType.LOCAL.equals(evalConfig.getDataSet().getSource())) {
            evalDataPath = subModelConfig.getModelSetName()
                    + File.separator + evalDataPath;
        }

        return evalDataPath;
    }

    /**
     * get @EvalConfig from @ModelConfig by evalset name
     *
     * @param mconfig
     * @param name
     * @return
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
     * @param subModelName
     * @param evalSetName
     * @return
     */
    private Callable<Integer> createTrainAndEvalTasks(final String subModelName, final String evalSetName) {
        return new Callable() {
            @Override
            public Integer call() {
                try {
                    return ProcessManager.runShellProcess(subModelName, new String[][]{
                            new String[]{"shifu", "train"},
                            new String[]{"shifu", "eval", "-score", evalSetName}
                    });
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
     * @return
     * @throws IOException
     */
    private List<Callable<Integer>> createEvaluateTasks(final String evalName) throws IOException {
        List<Callable<Integer>> tasks = new ArrayList<Callable<Integer>>();

        for (int i = 0; i < this.comboModelTrain.getVarTrainConfList().size(); i++) {
            VarTrainConf varTrainConf = this.comboModelTrain.getVarTrainConfList().get(i);
            final String subModelName = genSubModelName(i, varTrainConf);

            tasks.add(new Callable<Integer>() {
                @Override
                public Integer call() {
                    try {
                        return ProcessManager.runShellProcess(subModelName, new String[][]{
                                new String[]{"shifu", "eval", "-run", evalName}
                        });
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
     * @return
     */
    private VarTrainConf createVarTrainConf(ModelTrainConf.ALGORITHM alg) {
        VarTrainConf varTrainConf = new VarTrainConf();
        varTrainConf.setVariables(new ArrayList<String>());
        varTrainConf.setModelTrainConf(createModelTrainConf(alg));
        return varTrainConf;
    }

    /**
     * Create @ModelTrainConf according the @ModelTrainConf.ALGORITHM
     *
     * @param alg - the algorithm, see @ModelTrainConf.ALGORITHM
     * @return
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
            trainConf.setNumTrainEpochs(20000);
        } else if (ModelTrainConf.ALGORITHM.GBT.equals(alg)) {
            trainConf.setNumTrainEpochs(20000);
        } else if (ModelTrainConf.ALGORITHM.LR.equals(alg)) {
            trainConf.setNumTrainEpochs(100);
        }
        trainConf.setBaggingWithReplacement(true);

        return trainConf;
    }

    /**
     * Get evaluation output file format
     *
     * @param runMode
     * @return
     */
    private ColumnFile.FileType genEvalFileType(ModelBasicConf.RunMode runMode) {
        return (ModelBasicConf.RunMode.MAPRED.equals(runMode) ?
                ColumnFile.FileType.PIGSTORAGE : ColumnFile.FileType.CSV);
    }

    /**
     * Append weight column, if not blank, with uid column as score meta data.
     *
     * @param weightColumnName
     * @return
     */
    private String[] genEvalScoreMetaVars(String weightColumnName) {
        if (StringUtils.isNotBlank(weightColumnName)) {
            return new String[]{this.comboModelTrain.getUidColumnName(), weightColumnName};
        } else {
            return new String[]{this.comboModelTrain.getUidColumnName()};
        }
    }

    /**
     * Generate sub model name
     *
     * @param i            - sequence to keep unique
     * @param varTrainConf
     * @return
     */
    private String genSubModelName(int i, VarTrainConf varTrainConf) {
        return this.modelConfig.getBasic().getName()
                + "_" + varTrainConf.getModelTrainConf().getAlgorithm()
                + "_" + i;
    }

    /**
     * Generate assembel model name
     *
     * @param modelName
     * @return
     */
    private String genAssembleModelName(String modelName) {
        return modelName + "_" + Constants.COMBO_ASSEMBLE;
    }

    /**
     * Generate train data evaluation set name
     *
     * @return
     */
    private String genEvalTrainName() {
        return Constants.COMBO_EVAL_TRAIN;
    }

    /**
     * Save ComboTrain.json into local directory
     *
     * @param comboModelTrain
     * @return
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
     * @return
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
     * @param columnConfigList
     * @return
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
     * @param modelConfig
     * @throws IOException
     */
    private void saveModelConfig(String folder, ModelConfig modelConfig) throws IOException {
        JSONUtils.writeValue(new File(folder + File.separator + Constants.MODEL_CONFIG_JSON_FILE_NAME), modelConfig);
    }

    /**
     * Save ColumnConfig list into some folder
     *
     * @param folder        - folder to host ColumnConfig.json
     * @param columnConfigs
     * @throws IOException
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
     * @param subModelName
     * @param namesPrefix
     * @param parentNamesFile
     * @param varNames
     * @return
     * @throws IOException
     */
    private String createModelNamesFile(
            String subModelName,
            String namesPrefix,
            String parentNamesFile,
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
