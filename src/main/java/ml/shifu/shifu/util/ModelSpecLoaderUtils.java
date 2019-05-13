package ml.shifu.shifu.util;

import ml.shifu.shifu.container.obj.*;
import ml.shifu.shifu.container.obj.GenericModelConfig.ComputeImplClass;
import ml.shifu.shifu.container.obj.ModelTrainConf.ALGORITHM;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.*;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.dataset.BasicFloatNetwork;
import ml.shifu.shifu.core.dtrain.dataset.PersistBasicFloatNetwork;
import ml.shifu.shifu.core.dtrain.gs.GridSearch;
import ml.shifu.shifu.core.dtrain.lr.LogisticRegressionContants;
import ml.shifu.shifu.core.model.ModelSpec;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.fs.PathFinder;
import ml.shifu.shifu.fs.ShifuFileUtils;
import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.encog.ml.BasicML;
import org.encog.neural.networks.BasicNetwork;
import org.encog.persist.EncogDirectoryPersistence;
import org.encog.persist.PersistorRegistry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.*;

/**
 * Copyright [2013-2018] PayPal Software Foundation
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License")
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 **/

public class ModelSpecLoaderUtils {

    private static final Logger log = LoggerFactory.getLogger(ModelSpecLoaderUtils.class);

    /**
     * Avoid using new for our utility class.
     */
    private ModelSpecLoaderUtils() {
    }

    /**
     * Load basic models by configuration
     *
     * @param modelConfig
     *            ModelConfig
     * @param evalConfig
     *            eval configuration
     * @return the list of models
     * @throws IOException
     *             if any IO exception in reading model file.
     * @throws IllegalArgumentException
     *             if {@code modelConfig} is, if invalid model algorithm .
     * @throws IllegalStateException
     *             if not HDFS or LOCAL source type or algorithm not supported.
     */
    public static List<BasicML> loadBasicModels(ModelConfig modelConfig, EvalConfig evalConfig) throws IOException {
        if(modelConfig == null || (!Constants.NN.equalsIgnoreCase(modelConfig.getAlgorithm()) // NN model
                && !Constants.SVM.equalsIgnoreCase(modelConfig.getAlgorithm()) // SVM model
                && !Constants.LR.equalsIgnoreCase(modelConfig.getAlgorithm()) // LR model
                && !CommonUtils.isTreeModel(modelConfig.getAlgorithm())
                && !CommonUtils.isTensorFlowModel(modelConfig.getAlgorithm()))) {
            throw new IllegalArgumentException(modelConfig == null ? "modelConfig is null."
                    : String.format(" invalid model algorithm %s.", modelConfig.getAlgorithm()));
        }

        return loadBasicModels(modelConfig, evalConfig, modelConfig.getDataSet().getSource());
    }

    /**
     * Load basic models by configuration
     *
     * @param modelConfig
     *            ModelConfig
     * @param evalConfig
     *            eval configuration
     * @param sourceType
     *            source type
     * @return the list of models
     * @throws IOException
     *             Exception when fail to locate or load models
     */
    public static List<BasicML> loadBasicModels(ModelConfig modelConfig, EvalConfig evalConfig, SourceType sourceType)
            throws IOException {
        FileSystem fs = ShifuFileUtils.getFileSystemBySourceType(sourceType);

        List<BasicML> models = new ArrayList<BasicML>();
        List<Path> modelFileStats = locateBasicModels(modelConfig, evalConfig, sourceType);
        if(CollectionUtils.isNotEmpty(modelFileStats)) {
            for(Path f: modelFileStats) {
                models.add(loadModel(modelConfig, f, fs));
            }
        }

        return models;
    }

    /**
     * Get BasicNetwork from model, if model is not {@link BasicNetwork or NNModel}, Exception will be thrown
     *
     * @param model
     *            basic model
     * @return Network of model
     */
    public static BasicNetwork getBasicNetwork(BasicML model) {
        if(model instanceof BasicFloatNetwork) {
            return (BasicFloatNetwork) model;
        } else if(model instanceof NNModel) {
            return ((NNModel) model).getIndependentNNModel().getBasicNetworks().get(0);
        }
        throw new IllegalArgumentException("Only nn model is supported");
    }

    /**
     * Load basic models by configuration
     *
     * @param modelConfig
     *            model config
     * @param evalConfig
     *            eval config
     * @param sourceType
     *            source type
     * @param gbtConvertToProb
     *            convert gbt score to prob or not
     * @return list of models
     * @throws IOException
     *             if any IO exception in reading model file.
     * @throws IllegalArgumentException
     *             if {@code modelConfig} is, if invalid model algorithm .
     * @throws IllegalStateException
     *             if not HDFS or LOCAL source type or algorithm not supported.
     */
    public static List<BasicML> loadBasicModels(ModelConfig modelConfig, EvalConfig evalConfig, SourceType sourceType,
            boolean gbtConvertToProb) throws IOException {
        return loadBasicModels(modelConfig, evalConfig, sourceType, gbtConvertToProb, null);
    }

    /**
     * Load basic models by configuration
     *
     * @param modelConfig
     *            model config
     * @param evalConfig
     *            eval config
     * @param sourceType
     *            source type
     * @param gbtConvertToProb
     *            convert gbt score to prob or not
     * @param gbtScoreConvertStrategy
     *            specify how to convert gbt raw score
     * @return list of models
     * @throws IOException
     *             if any IO exception in reading model file.
     * @throws IllegalArgumentException
     *             if {@code modelConfig} is, if invalid model algorithm .
     * @throws IllegalStateException
     *             if not HDFS or LOCAL source type or algorithm not supported.
     */
    public static List<BasicML> loadBasicModels(ModelConfig modelConfig, EvalConfig evalConfig, SourceType sourceType,
            boolean gbtConvertToProb, String gbtScoreConvertStrategy) throws IOException {
        List<BasicML> models = new ArrayList<BasicML>();
        FileSystem fs = ShifuFileUtils.getFileSystemBySourceType(sourceType);
        // check if eval generic model, if so bypass the Shifu model loader procedure
        if(Constants.GENERIC.equalsIgnoreCase(modelConfig.getAlgorithm()) // generic or TensorFlow algorithm
                || Constants.TENSORFLOW.equalsIgnoreCase(modelConfig.getAlgorithm())) {
            List<Path> genericModelConfigs = findGenericModels(modelConfig, evalConfig, sourceType);
            if(genericModelConfigs.isEmpty()) {
                throw new RuntimeException("Load generic model failed.");
            }
            models = loadGenericModels(modelConfig, genericModelConfigs, sourceType);
            log.debug("return generic model {}", models.size());
            return models;
        }

        List<Path> modelFileStats = locateBasicModels(modelConfig, evalConfig, sourceType);
        if(CollectionUtils.isNotEmpty(modelFileStats)) {
            for(Path fst: modelFileStats) {
                models.add(loadModel(modelConfig, fst, fs, gbtConvertToProb, gbtScoreConvertStrategy));
            }
        }

        return models;
    }

    /**
     * Load generic model from local or HDFS storage and initialize.
     *
     * @param modelConfig
     *            model config
     * @param genericModelConfigs
     *            generic model files
     * @param sourceType
     *            source type
     * @return list of model object after loading from path list
     * @throws IOException
     *             Exception when fail to load generic models
     */
    public static List<BasicML> loadGenericModels(ModelConfig modelConfig, List<Path> genericModelConfigs,
            SourceType sourceType) throws IOException {
        FileSystem hdfs = HDFSUtils.getFS();
        PathFinder pathFinder = new PathFinder(modelConfig);

        String src = pathFinder.getModelsPath(sourceType);
        File f = new File(System.getProperty(Constants.USER_DIR) + "/models");
        // check if model dir is exist
        if(!f.exists()) {
            hdfs.copyToLocalFile(false, new Path(src), // source
                    new Path(System.getProperty(Constants.USER_DIR)), true);
        }

        List<BasicML> results = new ArrayList<>();
        for(Path fst: genericModelConfigs) {
            GenericModelConfig gmc = CommonUtils.loadJSON( // loading as GenericModelConfig
                    fst.toString(), sourceType, GenericModelConfig.class);
            String alg = (String) gmc.getProperties().get(Constants.GENERIC_ALGORITHM);
            String genericModelPath = System.getProperty(Constants.USER_DIR) // <usr.dir>
                    + File.separator + Constants.MODELS; // + /models
            // + File.separator + modelConfig.getBasic().getName(); // + /ModelName
            gmc.getProperties().put(Constants.GENERIC_MODEL_PATH, genericModelPath);
            log.info("Generic model path is : {}.", gmc.getProperties().get(Constants.GENERIC_MODEL_PATH));
            if(Constants.TENSORFLOW.equals(alg)) {
                try {
                    // Initiate a evaluator class instance which used for evaluation
                    Class<?> clazz = Class.forName(ComputeImplClass.Tensorflow.getClassName());
                    Computable computable = (Computable) clazz.newInstance();
                    computable.init(gmc);
                    results.add(new GenericModel(computable, gmc.getProperties()));
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            } else {
                throw new RuntimeException("Algorithm: " + alg + " is not supported in generic model yet.");
            }
        }
        return results;
    }

    /**
     * Get how many models under current workspace
     *
     * @param modelConfig
     *            model config
     * @param evalConfig
     *            eval configuration
     * @param sourceType
     *            {@link SourceType} LOCAL or HDFS?
     * @return the number of models
     * @throws IOException
     *             Exception when fail to load basic models
     */
    public static int getBasicModelsCnt(ModelConfig modelConfig, EvalConfig evalConfig,
            RawSourceData.SourceType sourceType) throws IOException {
        List<Path> modelFileStats = locateBasicModels(modelConfig, evalConfig, sourceType);
        return (CollectionUtils.isEmpty(modelFileStats) ? 0 : modelFileStats.size());
    }

    /**
     * Find model spec files
     *
     * @param modelConfig
     *            model config
     * @param evalConfig
     *            eval configuration
     * @param sourceType
     *            {@link SourceType} LOCAL or HDFS?
     * @return The basic model file list
     * @throws IOException
     *             Exception when fail to load basic models
     */
    public static List<Path> locateBasicModels(ModelConfig modelConfig, EvalConfig evalConfig, SourceType sourceType)
            throws IOException {
        // we have to register PersistBasicFloatNetwork for loading such models
        PersistorRegistry.getInstance().add(new PersistBasicFloatNetwork());

        List<Path> listStatus = findModels(modelConfig, evalConfig, sourceType);
        if(CollectionUtils.isEmpty(listStatus)) {
            // throw new ShifuException(ShifuErrorCode.ERROR_MODEL_FILE_NOT_FOUND);
            // disable exception, since we there maybe sub-models
            listStatus = findGenericModels(modelConfig, evalConfig, sourceType);
            // if models not found, continue which makes eval works when training is in progress.
            if(CollectionUtils.isNotEmpty(listStatus)) {
                return listStatus;
            }
        }

        // to avoid the *unix and windows file list order
        Collections.sort(listStatus, new Comparator<Path>() {
            @Override
            public int compare(Path f1, Path f2) {
                return f1.getName().compareToIgnoreCase(f2.getName());
            }
        });

        // added in shifu 0.2.5 to slice models not belonging to last training
        int baggingModelSize = modelConfig.getTrain().getBaggingNum();
        if(modelConfig.isClassification() && modelConfig.getTrain().isOneVsAll()) {
            baggingModelSize = modelConfig.getTags().size();
        }

        Integer kCrossValidation = modelConfig.getTrain().getNumKFold();
        if(kCrossValidation != null && kCrossValidation > 0) {
            // if kfold is enabled , bagging set it to bagging model size
            baggingModelSize = kCrossValidation;
        }

        GridSearch gs = new GridSearch(modelConfig.getTrain().getParams(),
                modelConfig.getTrain().getGridConfigFileContent());
        if(gs.hasHyperParam()) {
            // if it is grid search, set model size to all flatten params
            baggingModelSize = gs.getFlattenParams().size();
        }

        listStatus = listStatus.size() <= baggingModelSize ? listStatus : listStatus.subList(0, baggingModelSize);
        return listStatus;
    }

    /**
     * Loading model spec according to existing model path.
     *
     * @param modelConfig
     *            model config
     * @param modelPath
     *            the path of model file
     * @param fs
     *            {@link FileSystem}
     * @return model object or null if no modelPath file
     * @throws IOException
     *             Exception when fail to load model
     */
    public static BasicML loadModel(ModelConfig modelConfig, Path modelPath, FileSystem fs) throws IOException {
        return loadModel(modelConfig, modelPath, fs, false, Constants.GBT_SCORE_RAW_CONVETER);
    }

    /**
     * Loading model according to existing model path.
     *
     * @param modelConfig
     *            model config
     * @param modelPath
     *            the path to store model
     * @param fs
     *            file system used to store model
     * @param gbtConvertToProb
     *            convert gbt score to prob or not
     * @return model object or null if no modelPath file
     * @throws IOException
     *             if loading file for any IOException
     */
    public static BasicML loadModel(ModelConfig modelConfig, Path modelPath, FileSystem fs, boolean gbtConvertToProb)
            throws IOException {
        return loadModel(modelConfig, modelPath, fs, gbtConvertToProb, null);
    }

    /**
     * Loading model according to existing model path.
     *
     * @param modelConfig
     *            model config
     * @param modelPath
     *            the path to store model
     * @param fs
     *            file system used to store model
     * @param gbtConvertToProb
     *            convert gbt score to prob or not
     * @param gbtScoreConvertStrategy
     *            specify how to convert gbt raw score
     * @return model object or null if no modelPath file,
     * @throws IOException
     *             if loading file for any IOException
     */
    public static BasicML loadModel(ModelConfig modelConfig, Path modelPath, FileSystem fs, boolean gbtConvertToProb,
            String gbtScoreConvertStrategy) throws IOException {
        if(!fs.exists(modelPath)) {
            // no such existing model, return null.
            return null;
        }

        // TF for dedicated model loader
        if(Constants.GENERIC.equalsIgnoreCase(modelConfig.getAlgorithm()) // generic or TensorFlow algorithm
                || Constants.TENSORFLOW.equalsIgnoreCase(modelConfig.getAlgorithm())) {
            // only 1 model
            return loadGenericModels(modelConfig, Arrays.asList(modelPath), modelConfig.getDataSet().getSource())
                    .get(0);
        }

        // we have to register PersistBasicFloatNetwork for loading such models
        PersistorRegistry.getInstance().add(new PersistBasicFloatNetwork());
        FSDataInputStream stream = null;
        BufferedReader br = null;
        try {
            stream = fs.open(modelPath);
            if(modelPath.getName().endsWith(LogisticRegressionContants.LR_ALG_NAME.toLowerCase())) { // LR model
                br = new BufferedReader(new InputStreamReader(stream));
                try {
                    return LR.loadFromString(br.readLine());
                } catch (Exception e) { // local LR model?
                    IOUtils.closeQuietly(br); // close and reopen
                    stream = fs.open(modelPath);
                    return BasicML.class.cast(EncogDirectoryPersistence.loadObject(stream));
                }
            } else if(modelPath.getName().endsWith(CommonConstants.RF_ALG_NAME.toLowerCase()) // RF or GBT
                    || modelPath.getName().endsWith(CommonConstants.GBT_ALG_NAME.toLowerCase())) {
                return TreeModel.loadFromStream(stream, gbtConvertToProb, gbtScoreConvertStrategy);
            } else if(modelPath.getName().endsWith(Constants.WDL_ALG_NAME.toLowerCase())) {
                return WDLModel.loadFromStream(stream);
            } else {
                GzipStreamPair pair = GzipStreamPair.isGZipFormat(stream);
                if(pair.isGzip()) {
                    return BasicML.class.cast(NNModel.loadFromStream(pair.getInput()));
                } else {
                    return BasicML.class.cast(EncogDirectoryPersistence.loadObject(pair.getInput()));
                }
            }
        } catch (Exception e) {
            String msg = "the expecting model file is: " + modelPath;
            throw new ShifuException(ShifuErrorCode.ERROR_FAIL_TO_LOAD_MODEL_FILE, e, msg);
        } finally {
            IOUtils.closeQuietly(br);
            IOUtils.closeQuietly(stream);
        }
    }

    /**
     * Find the model files for some @ModelConfig. There is a little tricky about this function.
     * If @EvalConfig is specified, try to load the models according setting in @EvalConfig,
     * or if {@link EvalConfig} is null or modelsPath is blank, Shifu will try to load models under `models`
     * directory
     *
     * @param modelConfig
     *            - {@link ModelConfig}, need this, since the model file may exist in HDFS
     * @param evalConfig
     *            - {@link EvalConfig}, maybe null
     * @param sourceType
     *            - Where is file system
     * @return - {@link FileStatus} array for all found models
     * @throws IOException
     *             io exception to load files
     */
    public static List<Path> findModels(ModelConfig modelConfig, EvalConfig evalConfig, SourceType sourceType)
            throws IOException {
        FileSystem fs = ShifuFileUtils.getFileSystemBySourceType(sourceType);
        PathFinder pathFinder = new PathFinder(modelConfig);

        // If the algorithm in ModelConfig is NN, we only load NN models
        // the same as SVM, LR
        String modelSuffix = "." + modelConfig.getAlgorithm().toLowerCase();

        List<FileStatus> fileList = new ArrayList<>();
        if(null == evalConfig || StringUtils.isBlank(evalConfig.getModelsPath())) {
            Path path = new Path(pathFinder.getModelsPath(sourceType));
            fileList.addAll(Arrays.asList(fs.listStatus(path, new FileSuffixPathFilter(modelSuffix))));
        } else {
            String modelsPath = evalConfig.getModelsPath();
            FileStatus[] expandedPaths = fs.globStatus(new Path(modelsPath));
            if(ArrayUtils.isNotEmpty(expandedPaths)) {
                for(FileStatus fileStatus: expandedPaths) {
                    fileList.addAll(Arrays.asList(fs.listStatus(fileStatus.getPath(), // list all files
                            new FileSuffixPathFilter(modelSuffix))));
                }
            }
        }

        List<Path> paths = new ArrayList<>();
        for(FileStatus fileStatus: fileList) {
            paths.add(fileStatus.getPath());
        }

        return paths;
    }

    /**
     * Load the generic model config and parse it to java object.
     * Similar as {@link #findModels(ModelConfig, EvalConfig, RawSourceData.SourceType)}
     * 
     * @param modelConfig
     *            - {@link ModelConfig}, need this, since the model file may exist in HDFS
     * @param evalConfig
     *            - {@link EvalConfig}, maybe null
     * @param sourceType
     *            - {@link SourceType}, HDFS or Local?
     * @return the file status list for generic models
     * @throws IOException
     *             Exception occurred when finding generic models
     */
    public static List<Path> findGenericModels(ModelConfig modelConfig, EvalConfig evalConfig,
            RawSourceData.SourceType sourceType) throws IOException {
        FileSystem fs = ShifuFileUtils.getFileSystemBySourceType(sourceType);
        PathFinder pathFinder = new PathFinder(modelConfig);

        // Find generic model config file with suffix .json
        String modelSuffix = ".json";

        List<FileStatus> fileList = new ArrayList<>();
        if(null == evalConfig || StringUtils.isBlank(evalConfig.getModelsPath())) {
            Path path = new Path(pathFinder.getModelsPath(sourceType)); // modelsPath / <ModelName>
            // + File.separator + modelConfig.getBasic().getName());
            fileList.addAll(Arrays.asList(fs.listStatus(path, new FileSuffixPathFilter(modelSuffix))));
        } else {
            String modelsPath = evalConfig.getModelsPath();
            FileStatus[] expandedPaths = fs.globStatus(new Path(modelsPath)); // models / <ModelName>
            // + File.separator + modelConfig.getBasic().getName()));
            if(ArrayUtils.isNotEmpty(expandedPaths)) {
                for(FileStatus epath: expandedPaths) {
                    fileList.addAll(Arrays.asList(fs.listStatus(epath.getPath(), // list all files with suffix
                            new FileSuffixPathFilter(modelSuffix))));
                }
            }
        }

        List<Path> paths = new ArrayList<>();
        for(FileStatus fileStatus: fileList) {
            paths.add(fileStatus.getPath());
        }

        return paths;
    }

    /**
     * Load sub-models under current model space
     * 
     * @param modelConfig
     *            - {@link ModelConfig}, need this, since the model file may exist in HDFS
     * @param columnConfigList
     *            - List of {@link ColumnConfig}
     * @param evalConfig
     *            - {@link EvalConfig}, maybe null
     * @param sourceType
     *            - {@link SourceType}, HDFS or Local?
     * @param gbtConvertToProb
     *            - convert to probability or not for gbt model
     * @return list of {@link ModelSpec} for sub models
     */
    public static List<ModelSpec> loadSubModels(ModelConfig modelConfig, List<ColumnConfig> columnConfigList,
            EvalConfig evalConfig, RawSourceData.SourceType sourceType, Boolean gbtConvertToProb) {
        return loadSubModels(modelConfig, columnConfigList, evalConfig, sourceType, gbtConvertToProb, null);
    }

    /**
     * Load sub-models under current model space
     * 
     * @param modelConfig
     *            - {@link ModelConfig}, need this, since the model file may exist in HDFS
     * @param columnConfigList
     *            - List of {@link ColumnConfig}
     * @param evalConfig
     *            - {@link EvalConfig}, maybe null
     * @param sourceType
     *            - {@link SourceType}, HDFS or Local?
     * @param gbtConvertToProb
     *            - convert to probability or not for gbt model
     * @param gbtScoreConvertStrategy
     *            - gbt score conversion strategy
     * @return list of {@link ModelSpec} for sub models
     */
    @SuppressWarnings("deprecation")
    public static List<ModelSpec> loadSubModels(ModelConfig modelConfig, List<ColumnConfig> columnConfigList,
            EvalConfig evalConfig, RawSourceData.SourceType sourceType, Boolean gbtConvertToProb,
            String gbtScoreConvertStrategy) {
        List<ModelSpec> modelSpecs = new ArrayList<ModelSpec>();
        FileSystem fs = ShifuFileUtils.getFileSystemBySourceType(sourceType);

        // we have to register PersistBasicFloatNetwork for loading such models
        PersistorRegistry.getInstance().add(new PersistBasicFloatNetwork());
        PathFinder pathFinder = new PathFinder(modelConfig);
        String modelsPath = null;

        if(evalConfig == null || StringUtils.isEmpty(evalConfig.getModelsPath())) {
            modelsPath = pathFinder.getModelsPath(sourceType);
        } else {
            modelsPath = evalConfig.getModelsPath();
        }

        try {
            FileStatus[] fsArr = fs.listStatus(new Path(modelsPath));
            for(FileStatus fileStatus: fsArr) {
                if(fileStatus.isDir()) {
                    ModelSpec modelSpec = loadSubModelSpec(modelConfig, columnConfigList, fileStatus, sourceType,
                            gbtConvertToProb, gbtScoreConvertStrategy);
                    if(modelSpec != null) {
                        modelSpecs.add(modelSpec);
                    }
                }
            }
        } catch (IOException e) {
            log.error("Error occurred when loading sub-models.", e);
        }

        return modelSpecs;
    }

    /**
     * Load sub-model with FileStatus
     * 
     * @param modelConfig
     *            - {@link ModelConfig}, need this, since the model file may exist in HDFS
     * @param columnConfigList
     *            - List of {@link ColumnConfig}
     * @param fileStatus
     *            - {@link EvalConfig}, maybe null
     * @param sourceType
     *            - {@link SourceType}, HDFS or Local?
     * @param gbtConvertToProb
     *            - convert to probability or not for gbt model
     * @param gbtScoreConvertStrategy
     *            - gbt score conversion strategy
     * @return {@link ModelSpec} for sub-model
     */
    private static ModelSpec loadSubModelSpec(ModelConfig modelConfig, List<ColumnConfig> columnConfigList,
            FileStatus fileStatus, RawSourceData.SourceType sourceType, Boolean gbtConvertToProb,
            String gbtScoreConvertStrategy) throws IOException {
        FileSystem fs = ShifuFileUtils.getFileSystemBySourceType(sourceType);

        String subModelName = fileStatus.getPath().getName();
        List<FileStatus> modelFileStats = new ArrayList<FileStatus>();
        FileStatus[] subConfigs = new FileStatus[2];
        ALGORITHM algorithm = getModelsAlgAndSpecFiles(fileStatus, sourceType, modelFileStats, subConfigs);

        ModelSpec modelSpec = null;
        if(CollectionUtils.isNotEmpty(modelFileStats)) {
            Collections.sort(modelFileStats, new Comparator<FileStatus>() {
                @Override
                public int compare(FileStatus fa, FileStatus fb) {
                    return fa.getPath().getName().compareTo(fb.getPath().getName());
                }
            });
            List<BasicML> models = new ArrayList<BasicML>();
            for(FileStatus f: modelFileStats) {
                models.add(loadModel(modelConfig, f.getPath(), fs, gbtConvertToProb, gbtScoreConvertStrategy));
            }

            ModelConfig subModelConfig = modelConfig;
            if(subConfigs[0] != null) {
                subModelConfig = CommonUtils.loadModelConfig(subConfigs[0].getPath().toString(), sourceType);
            }
            List<ColumnConfig> subColumnConfigList = columnConfigList;
            if(subConfigs[1] != null) {
                subColumnConfigList = CommonUtils.loadColumnConfigList(subConfigs[1].getPath().toString(), sourceType);
            }

            modelSpec = new ModelSpec(subModelName, subModelConfig, subColumnConfigList, algorithm, models);
        }

        return modelSpec;
    }

    /**
     * Get the model spec file stats and return the ALGORITHM for the model spc
     * 
     * @param fileStatus
     *            directory to detect
     * @param sourceType
     *            {@link SourceType}
     * @param modelFileStats
     *            model spec file list to return
     * @param subConfigs
     *            configurations for the sub model
     * @return {@link ALGORITHM}
     * @throws IOException
     *             Exception occurred when finding model spec files
     */
    @SuppressWarnings("deprecation")
    public static ALGORITHM getModelsAlgAndSpecFiles(FileStatus fileStatus, RawSourceData.SourceType sourceType,
            List<FileStatus> modelFileStats, FileStatus[] subConfigs) throws IOException {
        assert modelFileStats != null;

        FileSystem fs = ShifuFileUtils.getFileSystemBySourceType(sourceType);
        ALGORITHM algorithm = null;

        FileStatus[] fileStatsArr = fs.listStatus(fileStatus.getPath());
        if(fileStatsArr != null) {
            for(FileStatus fls: fileStatsArr) {
                if(!fls.isDir()) {
                    String fileName = fls.getPath().getName();

                    if(algorithm == null) {
                        if(fileName.endsWith("." + ALGORITHM.NN.name().toLowerCase())) {
                            algorithm = ALGORITHM.NN;
                        } else if(fileName.endsWith("." + ALGORITHM.LR.name().toLowerCase())) {
                            algorithm = ALGORITHM.LR;
                        } else if(fileName.endsWith("." + ALGORITHM.GBT.name().toLowerCase())) {
                            algorithm = ALGORITHM.GBT;
                        }
                    }

                    if(algorithm != null && fileName.endsWith("." + algorithm.name().toLowerCase())) {
                        modelFileStats.add(fls);
                    }

                    if(fileName.equalsIgnoreCase(Constants.MODEL_CONFIG_JSON_FILE_NAME)) {
                        subConfigs[0] = fls;
                    } else if(fileName.equalsIgnoreCase(Constants.COLUMN_CONFIG_JSON_FILE_NAME)) {
                        subConfigs[1] = fls;
                    }
                }
            }
        }

        return algorithm;
    }

    /**
     * Get how many models for each sub models
     *
     * @param modelConfig
     *            model config
     * @param columnConfigList
     *            list of {@link ColumnConfig}
     * @param evalConfig
     *            eval configuration
     * @param sourceType
     *            {@link SourceType} LOCAL or HDFS?
     * @return the number of models
     */
    @SuppressWarnings("deprecation")
    public static Map<String, Integer> getSubModelsCnt(ModelConfig modelConfig, List<ColumnConfig> columnConfigList,
            EvalConfig evalConfig, RawSourceData.SourceType sourceType) {
        FileSystem fs = ShifuFileUtils.getFileSystemBySourceType(sourceType);
        PathFinder pathFinder = new PathFinder(modelConfig);

        String modelsPath = null;

        if(evalConfig == null || StringUtils.isEmpty(evalConfig.getModelsPath())) {
            modelsPath = pathFinder.getModelsPath(sourceType);
        } else {
            modelsPath = evalConfig.getModelsPath();
        }

        Map<String, Integer> subModelsCnt = new TreeMap<String, Integer>();

        try {
            FileStatus[] fsArr = fs.listStatus(new Path(modelsPath));
            for(FileStatus fileStatus: fsArr) {
                if(fileStatus.isDir()) {
                    List<FileStatus> subModelSpecFiles = new ArrayList<FileStatus>();
                    getModelsAlgAndSpecFiles(fileStatus, sourceType, subModelSpecFiles, new FileStatus[2]);
                    if(CollectionUtils.isNotEmpty(subModelSpecFiles)) {
                        subModelsCnt.put(fileStatus.getPath().getName(), subModelSpecFiles.size());
                    }
                }
            }
        } catch (IOException e) {
            log.error("Error occurred when finnding sub-models.", e);
        }

        return subModelsCnt;
    }

    /**
     * Load models with specified models path and algorithm
     * 
     * @param modelsPath
     *            - model spec path
     * @param alg
     *            - algorithm
     * @return - list of {@link BasicML}
     * @throws IOException
     *             Exception occurred when loading model specsModelSpecLoaderUtils
     */
    public static List<BasicML> loadBasicModels(final String modelsPath, final ModelTrainConf.ALGORITHM alg)
            throws IOException {
        return loadBasicModels(modelsPath, alg, false, Constants.GBT_SCORE_RAW_CONVETER);
    }

    /**
     * Load neural network models from specified file path
     *
     * @param modelsPath
     *            - a file or directory that contains .nn files
     * @param alg
     *            the algorithm
     * @param isConvertToProb
     *            if convert to prob for gbt model
     * @param gbtScoreConvertStrategy
     *            specify how to convert gbt raw score
     * @return - a list of @BasicML
     * @throws IOException
     *             - throw exception when loading model files
     */
    public static List<BasicML> loadBasicModels(final String modelsPath, final ALGORITHM alg, boolean isConvertToProb,
            String gbtScoreConvertStrategy) throws IOException {
        if(modelsPath == null || alg == null || ALGORITHM.DT.equals(alg)) {
            throw new IllegalArgumentException("The model path shouldn't be null");
        }
        // we have to register PersistBasicFloatNetwork for loading such models
        if(ALGORITHM.NN.equals(alg)) {
            PersistorRegistry.getInstance().add(new PersistBasicFloatNetwork());
        }

        File modelsPathDir = new File(modelsPath);
        File[] modelFiles = null;
        if(modelsPathDir.isDirectory()) { // user provide a directory
            modelFiles = modelsPathDir.listFiles(new FilenameFilter() {
                @Override
                public boolean accept(File dir, String name) {
                    return name.endsWith("." + alg.name().toLowerCase());
                }
            });
        } else { // user provide a single model spec
            if(modelsPath.endsWith("." + alg.name().toLowerCase())) {
                modelFiles = new File[] { modelsPathDir };
            }
        }

        if(modelFiles != null) {
            // sort file names
            Arrays.sort(modelFiles, new Comparator<File>() {
                @Override
                public int compare(File from, File to) {
                    return from.getName().compareTo(to.getName());
                }
            });

            List<BasicML> models = new ArrayList<BasicML>(modelFiles.length);
            for(File nnf: modelFiles) {
                InputStream is = null;
                try {
                    is = new FileInputStream(nnf);
                    if(ALGORITHM.NN.equals(alg)) {
                        GzipStreamPair pair = GzipStreamPair.isGZipFormat(is);
                        if(pair.isGzip()) {
                            models.add(BasicML.class.cast(NNModel.loadFromStream(pair.getInput())));
                        } else {
                            models.add(BasicML.class.cast(EncogDirectoryPersistence.loadObject(pair.getInput())));
                        }
                    } else if(ALGORITHM.LR.equals(alg)) {
                        models.add(LR.loadFromStream(is));
                    } else if(ALGORITHM.GBT.equals(alg) || ALGORITHM.RF.equals(alg)) {
                        models.add(TreeModel.loadFromStream(is, isConvertToProb, gbtScoreConvertStrategy));
                    }
                } finally {
                    IOUtils.closeQuietly(is);
                }
            }

            return models;
        } else {
            throw new IOException(String.format("Failed to list files in %s", modelsPathDir.getAbsolutePath()));
        }
    }
}
