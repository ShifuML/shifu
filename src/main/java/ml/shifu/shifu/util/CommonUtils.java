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
package ml.shifu.shifu.util;

import com.google.common.base.Function;
import com.google.common.base.Splitter;
import com.google.common.collect.Collections2;
import com.google.common.collect.Lists;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.collections.Predicate;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.pig.backend.executionengine.ExecException;
import org.apache.pig.data.Tuple;
import org.encog.ml.BasicML;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.neural.networks.BasicNetwork;
import org.encog.persist.EncogDirectoryPersistence;
import org.encog.persist.PersistorRegistry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import ml.shifu.guagua.GuaguaRuntimeException;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ColumnConfig.ColumnFlag;
import ml.shifu.shifu.container.obj.ColumnConfig.ColumnType;
import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelTrainConf.ALGORITHM;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.LR;
import ml.shifu.shifu.core.Normalizer;
import ml.shifu.shifu.core.TreeModel;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.dataset.PersistBasicFloatNetwork;
import ml.shifu.shifu.core.dtrain.lr.LogisticRegressionContants;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.fs.PathFinder;
import ml.shifu.shifu.fs.ShifuFileUtils;

/**
 * {@link CommonUtils} is used to for almost all kinds of utility function in this framework.
 */
public final class CommonUtils {

    /**
     * Avoid using new for our utility class.
     */
    private CommonUtils() {
    }

    private static final Logger log = LoggerFactory.getLogger(CommonUtils.class);

    /**
     * Sync up all local configuration files to HDFS.
     * 
     * @throws IOException
     *             If any exception on HDFS IO or local IO.
     * @throws NullPointerException
     *             If parameter {@code modelConfig} is null
     */
    public static boolean copyConfFromLocalToHDFS(ModelConfig modelConfig) throws IOException {
        FileSystem hdfs = HDFSUtils.getFS();
        FileSystem localFs = HDFSUtils.getLocalFS();

        PathFinder pathFinder = new PathFinder(modelConfig);

        Path pathModelSet = new Path(pathFinder.getModelSetPath(SourceType.HDFS));
        // don't check whether pathModelSet is exists, should be remove by user.
        hdfs.mkdirs(pathModelSet);

        // Copy ModelConfig
        Path srcModelConfig = new Path(pathFinder.getModelConfigPath(SourceType.LOCAL));
        Path dstModelConfig = new Path(pathFinder.getModelSetPath(SourceType.HDFS));
        hdfs.copyFromLocalFile(srcModelConfig, dstModelConfig);

        // Copy ColumnConfig
        Path srcColumnConfig = new Path(pathFinder.getColumnConfigPath(SourceType.LOCAL));
        Path dstColumnConfig = new Path(pathFinder.getColumnConfigPath(SourceType.HDFS));
        if(ShifuFileUtils.isFileExists(srcColumnConfig.toString(), SourceType.LOCAL)) {
            hdfs.copyFromLocalFile(srcColumnConfig, dstColumnConfig);
        }

        // copy others
        Path srcVersion = new Path(pathFinder.getModelVersion(SourceType.LOCAL));
        if(localFs.exists(srcVersion)) {
            Path dstVersion = new Path(pathFinder.getModelVersion(SourceType.HDFS));
            hdfs.delete(dstVersion, true);
            hdfs.copyFromLocalFile(srcVersion, pathModelSet);
        }

        // Copy Models
        Path srcModels = new Path(pathFinder.getModelsPath(SourceType.LOCAL));
        if(localFs.exists(srcModels)) {
            Path dstModels = new Path(pathFinder.getModelsPath(SourceType.HDFS));
            hdfs.delete(dstModels, true);
            hdfs.copyFromLocalFile(srcModels, pathModelSet);
        }

        // Copy EvalSets
        Path evalsPath = new Path(pathFinder.getEvalsPath(SourceType.LOCAL));
        if(localFs.exists(evalsPath)) {
            for(FileStatus evalset: localFs.listStatus(evalsPath)) {
                EvalConfig evalConfig = modelConfig.getEvalConfigByName(evalset.getPath().getName());
                if(evalConfig != null) {
                    copyEvalDataFromLocalToHDFS(modelConfig, evalConfig.getName());
                }
            }
        }

        return true;
    }

    /**
     * Sync-up the evalulation data into HDFS
     * 
     * @param modelConfig
     * @param evalName
     * @throws IOException
     */
    @SuppressWarnings("deprecation")
    public static void copyEvalDataFromLocalToHDFS(ModelConfig modelConfig, String evalName) throws IOException {
        EvalConfig evalConfig = modelConfig.getEvalConfigByName(evalName);
        if(evalConfig != null) {
            FileSystem hdfs = HDFSUtils.getFS();
            FileSystem localFs = HDFSUtils.getLocalFS();
            PathFinder pathFinder = new PathFinder(modelConfig);

            Path evalDir = new Path(pathFinder.getEvalSetPath(evalConfig, SourceType.LOCAL));
            Path dst = new Path(pathFinder.getEvalSetPath(evalConfig, SourceType.HDFS));
            if(localFs.exists(evalDir) // local evaluation folder exists
                    && localFs.getFileStatus(evalDir).isDir() // is directory
                    && !hdfs.exists(dst)) {
                hdfs.copyFromLocalFile(evalDir, dst);
            }

            if(StringUtils.isNotBlank(evalConfig.getScoreMetaColumnNameFile())) {
                hdfs.copyFromLocalFile(new Path(evalConfig.getScoreMetaColumnNameFile()),
                        new Path(pathFinder.getEvalSetPath(evalConfig)));
            }
        }
    }

    /**
     * Load ModelConfig from local json ModelConfig.json file.
     */
    public static ModelConfig loadModelConfig() throws IOException {
        return loadModelConfig(Constants.LOCAL_MODEL_CONFIG_JSON, SourceType.LOCAL);
    }

    /**
     * Load model configuration from the path and the source type.
     * 
     * @throws IOException
     *             if any IO exception in parsing json.
     * @throws IllegalArgumentException
     *             if {@code path} is null or empty, if sourceType is null.
     */
    public static ModelConfig loadModelConfig(String path, SourceType sourceType) throws IOException {
        return loadJSON(path, sourceType, ModelConfig.class);
    }

    private static void checkPathAndMode(String path, SourceType sourceType) {
        if(StringUtils.isEmpty(path) || sourceType == null) {
            throw new IllegalArgumentException(String.format(
                    "path should not be null or empty, sourceType should not be null, path:%s, sourceType:%s", path,
                    sourceType));
        }
    }

    /**
     * Load reason code map and change it to column->resonCode map.
     * 
     * @throws IOException
     *             if any IO exception in parsing json.
     * @throws IllegalArgumentException
     *             if {@code path} is null or empty, if sourceType is null.
     */
    public static Map<String, String> loadAndFlattenReasonCodeMap(String path, SourceType sourceType)
            throws IOException {
        @SuppressWarnings("unchecked")
        Map<String, List<String>> rawMap = loadJSON(path, sourceType, Map.class);

        Map<String, String> reasonCodeMap = new HashMap<String, String>();

        for(Map.Entry<String, List<String>> entry: rawMap.entrySet()) {
            for(String str: entry.getValue()) {
                reasonCodeMap.put(getRelativePigHeaderColumnName(str), entry.getKey());
            }
        }
        return reasonCodeMap;
    }

    /**
     * Load JSON instance
     * 
     * @throws IOException
     *             if any IO exception in parsing json.
     * @throws IllegalArgumentException
     *             if {@code path} is null or empty, if sourceType is null.
     */
    public static <T> T loadJSON(String path, SourceType sourceType, Class<T> clazz) throws IOException {
        checkPathAndMode(path, sourceType);
        log.debug("loading {} with sourceType {}", path, sourceType);
        BufferedReader reader = null;
        try {
            reader = ShifuFileUtils.getReader(path, sourceType);
            return JSONUtils.readValue(reader, clazz);
        } finally {
            IOUtils.closeQuietly(reader);
        }
    }

    /**
     * Load column configuration list.
     * 
     * @throws IOException
     *             if any IO exception in parsing json.
     */
    public static List<ColumnConfig> loadColumnConfigList() throws IOException {
        return loadColumnConfigList(Constants.LOCAL_COLUMN_CONFIG_JSON, SourceType.LOCAL);
    }

    /**
     * Load column configuration list.
     * 
     * @throws IOException
     *             if any IO exception in parsing json.
     * @throws IllegalArgumentException
     *             if {@code path} is null or empty, if sourceType is null.
     */
    public static List<ColumnConfig> loadColumnConfigList(String path, SourceType sourceType) throws IOException {
        return Arrays.asList(loadJSON(path, sourceType, ColumnConfig[].class));
    }

    /**
     * Return final selected column collection.
     */
    public static Collection<ColumnConfig> getFinalSelectColumnConfigList(Collection<ColumnConfig> columnConfigList) {
        return Collections2.filter(columnConfigList, new com.google.common.base.Predicate<ColumnConfig>() {
            @Override
            public boolean apply(ColumnConfig input) {
                return input.isFinalSelect();
            }
        });
    }

    public static String[] getFinalHeaders(ModelConfig modelConfig) throws IOException {
        String[] fields = null;
        boolean isSchemaProvided = true;
        if(StringUtils.isNotBlank(modelConfig.getHeaderPath())) {
            fields = CommonUtils.getHeaders(modelConfig.getHeaderPath(), modelConfig.getHeaderDelimiter(), modelConfig
                    .getDataSet().getSource());
        } else {
            fields = CommonUtils.takeFirstLine(modelConfig.getDataSetRawPath(), StringUtils.isBlank(modelConfig
                    .getHeaderDelimiter()) ? modelConfig.getDataSetDelimiter() : modelConfig.getHeaderDelimiter(),
                    modelConfig.getDataSet().getSource());
            if(StringUtils.join(fields, "").contains(modelConfig.getTargetColumnName())) {
                // if first line contains target column name, we guess it is csv format and first line is header.
                isSchemaProvided = true;
                log.warn("No header path is provided, we will try to read first line and detect schema.");
                log.warn("Schema in ColumnConfig.json are named as first line of data set path.");
            } else {
                isSchemaProvided = false;
                log.warn("No header path is provided, we will try to read first line and detect schema.");
                log.warn("Schema in ColumnConfig.json are named as  index 0, 1, 2, 3 ...");
                log.warn("Please make sure weight column and tag column are also taking index as name.");
            }
        }

        for(int i = 0; i < fields.length; i++) {
            if(!isSchemaProvided) {
                fields[i] = i + "";
            } else {
                fields[i] = getRelativePigHeaderColumnName(fields[i]);
            }
        }
        return fields;
    }

    public static String[] getFinalHeaders(EvalConfig evalConfig) throws IOException {
        String[] fields = null;
        boolean isSchemaProvided = true;
        if(StringUtils.isNotBlank(evalConfig.getDataSet().getHeaderPath())) {
            fields = CommonUtils.getHeaders(evalConfig.getDataSet().getHeaderPath(), evalConfig.getDataSet()
                    .getHeaderDelimiter(), evalConfig.getDataSet().getSource());
        } else {
            fields = CommonUtils.takeFirstLine(evalConfig.getDataSet().getDataPath(), StringUtils.isBlank(evalConfig
                    .getDataSet().getHeaderDelimiter()) ? evalConfig.getDataSet().getDataDelimiter() : evalConfig
                    .getDataSet().getHeaderDelimiter(), evalConfig.getDataSet().getSource());
            if(StringUtils.join(fields, "").contains(evalConfig.getDataSet().getTargetColumnName())) {
                // if first line contains target column name, we guess it is csv format and first line is header.
                isSchemaProvided = true;
                log.warn("No header path is provided, we will try to read first line and detect schema.");
                log.warn("Schema in ColumnConfig.json are named as first line of data set path.");
            } else {
                isSchemaProvided = false;
                log.warn("No header path is provided, we will try to read first line and detect schema.");
                log.warn("Schema in ColumnConfig.json are named as  index 0, 1, 2, 3 ...");
                log.warn("Please make sure weight column and tag column are also taking index as name.");
            }
        }

        for(int i = 0; i < fields.length; i++) {
            if(!isSchemaProvided) {
                fields[i] = i + "";
            } else {
                fields[i] = getRelativePigHeaderColumnName(fields[i]);
            }
        }
        return fields;
    }

    /**
     * Return header column list from header file.
     * 
     * @throws IOException
     *             if any IO exception in reading file.
     * @throws IllegalArgumentException
     *             if sourceType is null, if pathHeader is null or empty, if delimiter is null or empty.
     * @throws RuntimeException
     *             if first line of pathHeader is null or empty.
     */
    public static String[] getHeaders(String pathHeader, String delimiter, SourceType sourceType) throws IOException {
        return getHeaders(pathHeader, delimiter, sourceType, false);
    }

    /**
     * Return header column array from header file.
     * 
     * @throws IOException
     *             if any IO exception in reading file.
     * @throws IllegalArgumentException
     *             if sourceType is null, if pathHeader is null or empty, if delimiter is null or empty.
     * @throws RuntimeException
     *             if first line of pathHeader is null or empty.
     */
    public static String[] getHeaders(String pathHeader, String delimiter, SourceType sourceType, boolean isFull)
            throws IOException {
        if(StringUtils.isEmpty(pathHeader) || StringUtils.isEmpty(delimiter) || sourceType == null) {
            throw new IllegalArgumentException(String.format(
                    "Null or empty parameters srcDataPath:%s, dstDataPath:%s, sourceType:%s", pathHeader, delimiter,
                    sourceType));
        }
        BufferedReader reader = null;
        String pigHeaderStr = null;

        try {
            reader = ShifuFileUtils.getReader(pathHeader, sourceType);
            pigHeaderStr = reader.readLine();
            if(StringUtils.isEmpty(pigHeaderStr)) {
                throw new RuntimeException(String.format("Cannot reade header info from the first line of file: %s",
                        pathHeader));
            }
        } catch (Exception e) {
            log.error(
                    "Error in getReader, this must be catched in this method to make sure the next reader can be returned.",
                    e);
            throw new ShifuException(ShifuErrorCode.ERROR_HEADER_NOT_FOUND);
        } finally {
            IOUtils.closeQuietly(reader);
        }

        List<String> headerList = new ArrayList<String>();
        Set<String> headerSet = new HashSet<String>();
        int index = 0;
        for(String str: Splitter.on(delimiter).split(pigHeaderStr)) {
            String columnName;
            if(isFull) {
                columnName = getFullPigHeaderColumnName(str);
            } else {
                columnName = getRelativePigHeaderColumnName(str);
            }

            if(headerSet.contains(columnName)) {
                columnName = columnName + "_" + index;
            }
            headerSet.add(columnName);
            index++;
            headerList.add(columnName);
        }
        return headerList.toArray(new String[0]);
    }

    /**
     * Get full column name from pig header. For example, one column is a::b, return a_b. If b, return b.
     */
    public static String getFullPigHeaderColumnName(String raw) {
        return raw == null ? raw : raw.replaceAll(Constants.PIG_COLUMN_SEPARATOR, Constants.PIG_FULL_COLUMN_SEPARATOR);
        // return raw;
    }

    /**
     * Get relative column name from pig header. For example, one column is a::b, return b. If b, return b.
     * 
     * @throws NullPointerException
     *             if parameter raw is null.
     */
    public static String getRelativePigHeaderColumnName(String raw) {
        int position = raw.lastIndexOf(Constants.PIG_COLUMN_SEPARATOR);
        return position >= 0 ? raw.substring(position + Constants.PIG_COLUMN_SEPARATOR.length()) : raw;
    }

    /**
     * Given a column value, return bin list index. Return 0 for Category because of index 0 is started from
     * NEGATIVE_INFINITY.
     * 
     * @throws IllegalArgumentException
     *             if input is null or empty.
     * @throws NumberFormatException
     *             if columnVal does not contain a parsable number.
     */
    public static int getBinNum(ColumnConfig columnConfig, String columnVal) {
        if(columnConfig.isCategorical()) {
            List<String> binCategories = columnConfig.getBinCategory();
            for(int i = 0; i < binCategories.size(); i++) {
                if(binCategories.get(i).equals(columnVal)) {
                    return i;
                }
            }
            return -1;
        } else {
            if(StringUtils.isBlank(columnVal)) {
                return -1;
            }
            double dval = 0.0;
            try {
                dval = Double.parseDouble(columnVal);
            } catch (Exception e) {
                return -1;
            }
            return getBinIndex(columnConfig.getBinBoundary(), dval);
        }
    }

    /**
     * Return the real bin number for one value. As the first bin value is NEGATIVE_INFINITY, invalid index is 0, not
     * -1.
     * 
     * @param binBoundary
     *            bin boundary list which should be sorted.
     * @throws IllegalArgumentException
     *             if binBoundary is null or empty.
     */
    @SuppressWarnings("unused")
    private static int getNumericBinNum(List<Double> binBoundary, double value) {
        if(CollectionUtils.isEmpty(binBoundary)) {
            throw new IllegalArgumentException("binBoundary should not be null or empty.");
        }

        int n = binBoundary.size() - 1;
        while(n > 0 && value < binBoundary.get(n)) {
            n--;
        }
        return n;
    }

    /**
     * Common split function to ignore special character like '|'. It's better to return a list while many calls in our
     * framework using string[].
     * 
     * @throws IllegalArgumentException
     *             {@code raw} and {@code delimiter} is null or empty.
     */
    public static String[] split(String raw, String delimiter) {
        return splitAndReturnList(raw, delimiter).toArray(new String[0]);
    }

    /**
     * Common split function to ignore special character like '|'.
     * 
     * @throws IllegalArgumentException
     *             {@code raw} and {@code delimiter} is null or empty.
     */
    public static List<String> splitAndReturnList(String raw, String delimiter) {
        if(StringUtils.isEmpty(raw) || StringUtils.isEmpty(delimiter)) {
            throw new IllegalArgumentException(String.format(
                    "raw and delimeter should not be null or empty, raw:%s, delimeter:%s", raw, delimiter));
        }
        List<String> headerList = new ArrayList<String>();
        for(String str: Splitter.on(delimiter).split(raw)) {
            headerList.add(str);
        }
        return headerList;
    }

    /**
     * Get target column.
     * 
     * @throws IllegalArgumentException
     *             if columnConfigList is null or empty.
     * @throws IllegalStateException
     *             if no target column can be found.
     */
    public static Integer getTargetColumnNum(List<ColumnConfig> columnConfigList) {
        if(CollectionUtils.isEmpty(columnConfigList)) {
            throw new IllegalArgumentException("columnConfigList should not be null or empty.");
        }
        // I need cast operation because of common-collections dosen't support generic.
        ColumnConfig cc = (ColumnConfig) CollectionUtils.find(columnConfigList, new Predicate() {
            @Override
            public boolean evaluate(Object object) {
                return ((ColumnConfig) object).isTarget();
            }
        });
        if(cc == null) {
            throw new IllegalStateException("No target column can be found, please check your column configurations");
        }
        return cc.getColumnNum();
    }

    /**
     * Load basic models from files.
     * 
     * @throws IOException
     *             if any IO exception in reading model file.
     * @throws IllegalArgumentException
     *             if {@code modelConfig} is, if invalid model algorithm .
     * @throws IllegalStateException
     *             if not HDFS or LOCAL source type or algorithm not supported.
     */
    public static List<BasicML> loadBasicModels(ModelConfig modelConfig, List<ColumnConfig> columnConfigList,
            EvalConfig evalConfig) throws IOException {
        if(modelConfig == null
                || (!Constants.NN.equalsIgnoreCase(modelConfig.getAlgorithm())
                        && !Constants.SVM.equalsIgnoreCase(modelConfig.getAlgorithm())
                        && !Constants.LR.equalsIgnoreCase(modelConfig.getAlgorithm()) && !CommonUtils
                            .isDesicionTreeAlgorithm(modelConfig.getAlgorithm()))) {
            throw new IllegalArgumentException(modelConfig == null ? "modelConfig is null." : String.format(
                    " invalid model algorithm %s.", modelConfig.getAlgorithm()));
        }

        return loadBasicModels(modelConfig, columnConfigList, evalConfig, modelConfig.getDataSet().getSource());
    }

    /**
     * Get bin index by binary search. The last bin in <code>binBoundary</code> is missing value bin.
     */
    public static int getBinIndex(List<Double> binBoundary, Double dVal) {
        assert binBoundary != null && binBoundary.size() > 0;
        assert dVal != null;
        int binSize = binBoundary.size();

        int low = 0;
        int high = binSize - 1;

        while(low <= high) {
            int mid = (low + high) >>> 1;
            Double midVal = binBoundary.get(mid);
            int cmp = midVal.compareTo(dVal);

            if(cmp < 0) {
                low = mid + 1;
            } else if(cmp > 0) {
                high = mid - 1;
            } else {
                return mid; // key found
            }
        }

        return low == 0 ? 0 : low - 1;
    }

    public static List<BasicML> loadBasicModels(ModelConfig modelConfig, List<ColumnConfig> columnConfigList,
            EvalConfig evalConfig, SourceType sourceType) throws IOException {
        List<BasicML> models = new ArrayList<BasicML>();
        FileSystem fs = ShifuFileUtils.getFileSystemBySourceType(sourceType);

        List<FileStatus> modelFileStats = locateBasicModels(modelConfig, columnConfigList, evalConfig, sourceType);
        if(CollectionUtils.isNotEmpty(modelFileStats)) {
            for(FileStatus f: modelFileStats) {
                models.add(loadModel(modelConfig, columnConfigList, f.getPath(), fs));
            }
        }

        return models;
    }

    /**
     * Load basic models from files.
     * 
     * @throws IOException
     *             if any IO exception in reading model file.
     * @throws IllegalArgumentException
     *             if {@code modelConfig} is, if invalid model algorithm .
     * @throws IllegalStateException
     *             if not HDFS or LOCAL source type or algorithm not supported.
     */
    public static List<BasicML> loadBasicModels(ModelConfig modelConfig, List<ColumnConfig> columnConfigList,
            EvalConfig evalConfig, SourceType sourceType, boolean gbtConvertToProb) throws IOException {
        List<BasicML> models = new ArrayList<BasicML>();
        FileSystem fs = ShifuFileUtils.getFileSystemBySourceType(sourceType);

        List<FileStatus> modelFileStats = locateBasicModels(modelConfig, columnConfigList, evalConfig, sourceType);
        if(CollectionUtils.isNotEmpty(modelFileStats)) {
            for(FileStatus f: modelFileStats) {
                models.add(loadModel(modelConfig, columnConfigList, f.getPath(), fs, gbtConvertToProb));
            }
        }

        return models;
    }

    public static int getBasicModelsCnt(ModelConfig modelConfig, List<ColumnConfig> columnConfigList,
            EvalConfig evalConfig, SourceType sourceType) throws IOException {
        List<FileStatus> modelFileStats = locateBasicModels(modelConfig, columnConfigList, evalConfig, sourceType);
        return (CollectionUtils.isEmpty(modelFileStats) ? 0 : modelFileStats.size());
    }

    public static List<FileStatus> locateBasicModels(ModelConfig modelConfig, List<ColumnConfig> columnConfigList,
            EvalConfig evalConfig, SourceType sourceType) throws IOException {
        // we have to register PersistBasicFloatNetwork for loading such models
        PersistorRegistry.getInstance().add(new PersistBasicFloatNetwork());

        List<FileStatus> listStatus = findModels(modelConfig, evalConfig, sourceType);
        if(CollectionUtils.isEmpty(listStatus)) {
            throw new ShifuException(ShifuErrorCode.ERROR_MODEL_FILE_NOT_FOUND);
        }

        // to avoid the *unix and windows file list order
        Collections.sort(listStatus, new Comparator<FileStatus>() {
            @Override
            public int compare(FileStatus f1, FileStatus f2) {
                return f1.getPath().getName().compareToIgnoreCase(f2.getPath().getName());
            }

        });

        // added in shifu 0.2.5 to slice models not belonging to last training
        int baggingModelSize = modelConfig.getTrain().getBaggingNum();
        if(modelConfig.isClassification() && modelConfig.getTrain().isOneVsAll()) {
            baggingModelSize = modelConfig.getTags().size();
        }
        listStatus = listStatus.size() <= baggingModelSize ? listStatus : listStatus.subList(0, baggingModelSize);
        return listStatus;
    }

    public static BasicML loadModel(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, Path modelPath,
            FileSystem fs) throws IOException {
        return loadModel(modelConfig, columnConfigList, modelPath, fs, false);
    }

    /**
     * Loading model according to existing model path.
     * 
     * @param modelPath
     *            the path to store model
     * @param fs
     *            file system used to store model
     * @return model object or null if no modelPath file,
     * @throws IOException
     *             if loading file for any IOException
     * @throws GuaguaRuntimeException
     *             if any exception to load model object and cast to {@link BasicNetwork}
     */
    public static BasicML loadModel(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, Path modelPath,
            FileSystem fs, boolean gbtConvertToProb) throws IOException {
        if(!fs.exists(modelPath)) {
            // no such existing model, return null.
            return null;
        }
        // we have to register PersistBasicFloatNetwork for loading such models
        PersistorRegistry.getInstance().add(new PersistBasicFloatNetwork());
        FSDataInputStream stream = null;
        BufferedReader br = null;
        try {
            stream = fs.open(modelPath);
            if(modelPath.getName().endsWith(LogisticRegressionContants.LR_ALG_NAME.toLowerCase())) {
                br = new BufferedReader(new InputStreamReader(stream));
                return LR.loadFromString(br.readLine());
            } else if(modelPath.getName().endsWith(CommonConstants.RF_ALG_NAME.toLowerCase())
                    || modelPath.getName().endsWith(CommonConstants.GBT_ALG_NAME.toLowerCase())) {
                return TreeModel.loadFromStream(stream, gbtConvertToProb);
            } else {
                return BasicML.class.cast(EncogDirectoryPersistence.loadObject(stream));
            }
        } catch (Exception e) {
            String msg = "the expecting model file is: " + modelPath;
            throw new ShifuException(ShifuErrorCode.ERROR_FAIL_TO_LOAD_MODEL_FILE, e, msg);
        } finally {
            if(br != null) {
                IOUtils.closeQuietly(br);
            }
            if(stream != null) {
                IOUtils.closeQuietly(stream);
            }
        }
    }

    /**
     * Find the model files for some @ModelConfig. There is a little tricky about this function.
     * If @EvalConfig is specified, try to load the models according setting in @EvalConfig,
     * or if @EvalConfig is null or ModelsPath is blank, Shifu will try to load models under `models`
     * directory
     * 
     * @param modelConfig
     *            - @ModelConfig, need this, since the model file may exist in HDFS
     * @param evalConfig
     *            - @EvalConfig, maybe null
     * @param sourceType
     *            - Where is file system
     * @return - @FileStatus array for all found models
     * @throws IOException
     */
    public static List<FileStatus> findModels(ModelConfig modelConfig, EvalConfig evalConfig, SourceType sourceType)
            throws IOException {
        FileSystem fs = ShifuFileUtils.getFileSystemBySourceType(sourceType);
        PathFinder pathFinder = new PathFinder(modelConfig);

        // If the algorithm in ModelConfig is NN, we only load NN models
        // the same as SVM, LR
        String modelSuffix = "." + modelConfig.getAlgorithm().toLowerCase();

        List<FileStatus> fileList = new ArrayList<FileStatus>();
        if(null == evalConfig || StringUtils.isBlank(evalConfig.getModelsPath())) {
            Path path = new Path(pathFinder.getModelsPath(sourceType));
            fileList.addAll(Arrays.asList(fs.listStatus(path, new FileSuffixPathFilter(modelSuffix))));
        } else {
            String modelsPath = evalConfig.getModelsPath();
            FileStatus[] expandedPaths = fs.globStatus(new Path(modelsPath));
            if(ArrayUtils.isNotEmpty(expandedPaths)) {
                for(FileStatus epath: expandedPaths) {
                    fileList.addAll(Arrays.asList(fs.listStatus(epath.getPath(), new FileSuffixPathFilter(modelSuffix))));
                }
            }
        }

        return fileList;
    }

    public static class FileSuffixPathFilter implements PathFilter {
        private String fileSuffix;

        public FileSuffixPathFilter(String fileSuffix) {
            this.fileSuffix = fileSuffix;
        }

        @Override
        public boolean accept(Path path) {
            return path.getName().endsWith(fileSuffix);
        }
    }

    public static List<BasicML> loadBasicModels(final String modelsPath, final ALGORITHM alg) throws IOException {
        return loadBasicModels(modelsPath, alg, false);
    }

    /**
     * Load neural network models from specified file path
     * 
     * @param modelsPath
     *            - a file or directory that contains .nn files
     * @return - a list of @BasicML
     * @throws IOException
     *             - throw exception when loading model files
     */
    public static List<BasicML> loadBasicModels(final String modelsPath, final ALGORITHM alg, boolean isConvertToProb)
            throws IOException {
        if(modelsPath == null || alg == null || ALGORITHM.DT.equals(alg)) {
            throw new IllegalArgumentException("The model path shouldn't be null");
        }
        // we have to register PersistBasicFloatNetwork for loading such models
        if(ALGORITHM.NN.equals(alg)) {
            PersistorRegistry.getInstance().add(new PersistBasicFloatNetwork());
        }

        File modelsPathDir = new File(modelsPath);

        File[] modelFiles = modelsPathDir.listFiles(new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return name.endsWith("." + alg.name().toLowerCase());
            }
        });

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
                        models.add(BasicML.class.cast(EncogDirectoryPersistence.loadObject(is)));
                    } else if(ALGORITHM.LR.equals(alg)) {
                        models.add(LR.loadFromStream(is));
                    } else if(ALGORITHM.GBT.equals(alg) || ALGORITHM.RF.equals(alg)) {
                        models.add(TreeModel.loadFromStream(is, isConvertToProb));
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

    /**
     * Return one HashMap Object contains keys in the first parameter, values in the second parameter. Before calling
     * this method, you should be aware that headers should be unique.
     * 
     * @throws IllegalArgumentException
     *             if lengths of two arrays are not the same.
     * @throws NullPointerException
     *             if header or data is null.
     */
    public static Map<String, String> getRawDataMap(String[] header, String[] data) {
        if(header.length != data.length) {
            throw new IllegalArgumentException(String.format("Header/Data mismatch: Header length %s, Data length %s",
                    header.length, data.length));
        }

        Map<String, String> rawDataMap = new HashMap<String, String>(header.length);
        for(int i = 0; i < header.length; i++) {
            rawDataMap.put(header[i], data[i]);
        }
        return rawDataMap;
    }

    /**
     * Return all parameters for pig execution.
     * 
     * @throws IllegalArgumentException
     *             if modelConfig is null.
     */
    public static Map<String, String> getPigParamMap(ModelConfig modelConfig, SourceType sourceType) throws IOException {
        if(modelConfig == null) {
            throw new IllegalArgumentException("modelConfig should not be null.");
        }
        PathFinder pathFinder = new PathFinder(modelConfig);

        Map<String, String> pigParamMap = new HashMap<String, String>();
        pigParamMap.put(Constants.NUM_PARALLEL, Environment.getInt(Environment.HADOOP_NUM_PARALLEL, 400).toString());
        log.info("jar path is {}", pathFinder.getJarPath());
        pigParamMap.put(Constants.PATH_JAR, pathFinder.getJarPath());

        pigParamMap.put(Constants.PATH_RAW_DATA, modelConfig.getDataSetRawPath());
        pigParamMap.put(Constants.PATH_NORMALIZED_DATA, pathFinder.getNormalizedDataPath(sourceType));
        // default norm is not for clean, so set it to false, this will be overrided in Train#Norm for tree models
        pigParamMap.put(Constants.IS_NORM_FOR_CLEAN, Boolean.FALSE.toString());
        pigParamMap.put(Constants.PATH_PRE_TRAINING_STATS, pathFinder.getPreTrainingStatsPath(sourceType));
        pigParamMap.put(Constants.PATH_STATS_BINNING_INFO, pathFinder.getUpdatedBinningInfoPath(sourceType));
        pigParamMap.put(Constants.PATH_STATS_PSI_INFO, pathFinder.getPSIInfoPath(sourceType));

        pigParamMap.put(Constants.WITH_SCORE, Boolean.FALSE.toString());
        pigParamMap.put(Constants.STATS_SAMPLE_RATE, modelConfig.getBinningSampleRate().toString());
        pigParamMap.put(Constants.PATH_MODEL_CONFIG, pathFinder.getModelConfigPath(sourceType));
        pigParamMap.put(Constants.PATH_COLUMN_CONFIG, pathFinder.getColumnConfigPath(sourceType));
        pigParamMap.put(Constants.PATH_SELECTED_RAW_DATA, pathFinder.getSelectedRawDataPath(sourceType));
        pigParamMap.put(Constants.PATH_BIN_AVG_SCORE, pathFinder.getBinAvgScorePath(sourceType));
        pigParamMap.put(Constants.PATH_TRAIN_SCORE, pathFinder.getTrainScoresPath(sourceType));

        pigParamMap.put(Constants.SOURCE_TYPE, sourceType.toString());
        pigParamMap.put(Constants.JOB_QUEUE,
                Environment.getProperty(Environment.HADOOP_JOB_QUEUE, Constants.DEFAULT_JOB_QUEUE));
        return pigParamMap;
    }

    /**
     * Return all parameters for pig execution.
     * 
     * @throws IllegalArgumentException
     *             if modelConfig is null.
     */
    public static Map<String, String> getPigParamMap(ModelConfig modelConfig, SourceType sourceType,
            PathFinder pathFinder) throws IOException {
        if(modelConfig == null) {
            throw new IllegalArgumentException("modelConfig should not be null.");
        }
        if(pathFinder == null) {
            pathFinder = new PathFinder(modelConfig);
        }
        Map<String, String> pigParamMap = new HashMap<String, String>();
        pigParamMap.put(Constants.NUM_PARALLEL, Environment.getInt(Environment.HADOOP_NUM_PARALLEL, 400).toString());
        log.info("jar path is {}", pathFinder.getJarPath());
        pigParamMap.put(Constants.PATH_JAR, pathFinder.getJarPath());

        pigParamMap.put(Constants.PATH_RAW_DATA, modelConfig.getDataSetRawPath());
        pigParamMap.put(Constants.PATH_NORMALIZED_DATA, pathFinder.getNormalizedDataPath(sourceType));
        pigParamMap.put(Constants.PATH_PRE_TRAINING_STATS, pathFinder.getPreTrainingStatsPath(sourceType));
        pigParamMap.put(Constants.PATH_STATS_BINNING_INFO, pathFinder.getUpdatedBinningInfoPath(sourceType));
        pigParamMap.put(Constants.PATH_STATS_PSI_INFO, pathFinder.getPSIInfoPath(sourceType));

        pigParamMap.put(Constants.WITH_SCORE, Boolean.FALSE.toString());
        pigParamMap.put(Constants.STATS_SAMPLE_RATE, modelConfig.getBinningSampleRate().toString());
        pigParamMap.put(Constants.PATH_MODEL_CONFIG, pathFinder.getModelConfigPath(sourceType));
        pigParamMap.put(Constants.PATH_COLUMN_CONFIG, pathFinder.getColumnConfigPath(sourceType));
        pigParamMap.put(Constants.PATH_SELECTED_RAW_DATA, pathFinder.getSelectedRawDataPath(sourceType));
        pigParamMap.put(Constants.PATH_BIN_AVG_SCORE, pathFinder.getBinAvgScorePath(sourceType));
        pigParamMap.put(Constants.PATH_TRAIN_SCORE, pathFinder.getTrainScoresPath(sourceType));

        pigParamMap.put(Constants.SOURCE_TYPE, sourceType.toString());
        pigParamMap.put(Constants.JOB_QUEUE,
                Environment.getProperty(Environment.HADOOP_JOB_QUEUE, Constants.DEFAULT_JOB_QUEUE));
        pigParamMap.put(Constants.DATASET_NAME, modelConfig.getBasic().getName());
        return pigParamMap;
    }

    /**
     * Change list str to List object with double type.
     * 
     * @throws IllegalArgumentException
     *             if str is not a valid list str: [1,2].
     */
    public static List<Double> stringToDoubleList(String str) {
        List<String> list = checkAndReturnSplitCollections(str);

        return Lists.transform(list, new Function<String, Double>() {
            @Override
            public Double apply(String input) {
                return Double.valueOf(input.trim());
            }
        });
    }

    private static List<String> checkAndReturnSplitCollections(String str) {
        checkListStr(str);
        return Arrays.asList(str.trim().substring(1, str.length() - 1).split(Constants.COMMA));
    }

    private static List<String> checkAndReturnSplitCollections(String str, char separator) {
        checkListStr(str);
        return Arrays.asList(StringUtils.split(str.trim().substring(1, str.length() - 1), separator));
    }

    private static void checkListStr(String str) {
        if(StringUtils.isEmpty(str)) {
            throw new IllegalArgumentException("str should not be null or empty");
        }
        if(!str.startsWith("[") || !str.endsWith("]")) {
            throw new IllegalArgumentException("Invalid list string format, should be like '[1,2,3]'");
        }
    }

    /**
     * Change list str to List object with integer type.
     * 
     * @throws IllegalArgumentException
     *             if str is not a valid list str.
     */
    public static List<Integer> stringToIntegerList(String str) {
        List<String> list = checkAndReturnSplitCollections(str);
        return Lists.transform(list, new Function<String, Integer>() {
            @Override
            public Integer apply(String input) {
                return Integer.valueOf(input.trim());
            }
        });
    }

    /**
     * Change list str to List object with string type.
     * 
     * @throws IllegalArgumentException
     *             if str is not a valid list str.
     */
    public static List<String> stringToStringList(String str) {
        List<String> list = checkAndReturnSplitCollections(str);
        return Lists.transform(list, new Function<String, String>() {
            @Override
            public String apply(String input) {
                return input.trim();
            }
        });
    }

    /**
     * Change list str to List object with string type.
     * 
     * @throws IllegalArgumentException
     *             if str is not a valid list str.
     */
    public static List<String> stringToStringList(String str, char separator) {
        List<String> list = checkAndReturnSplitCollections(str, separator);
        return Lists.transform(list, new Function<String, String>() {
            @Override
            public String apply(String input) {
                return input.trim();
            }
        });
    }

    /**
     * Return map entries sorted by value.
     */
    public static <K, V extends Comparable<V>> List<Map.Entry<K, V>> getEntriesSortedByValues(Map<K, V> map) {
        List<Map.Entry<K, V>> entries = new LinkedList<Map.Entry<K, V>>(map.entrySet());

        Collections.sort(entries, new Comparator<Map.Entry<K, V>>() {
            @Override
            public int compare(Entry<K, V> o1, Entry<K, V> o2) {
                return o1.getValue().compareTo(o2.getValue());
            }
        });

        return entries;
    }

    /**
     * Assemble map data to Encog standard input format with default cut off value.
     * 
     * @throws NullPointerException
     *             if input is null
     * @throws NumberFormatException
     *             if column value is not number format.
     */
    public static MLDataPair assembleDataPair(ModelConfig modelConfig, List<ColumnConfig> columnConfigList,
            Map<String, ? extends Object> rawDataMap) {
        return assembleDataPair(modelConfig, columnConfigList, rawDataMap, Constants.DEFAULT_CUT_OFF);
    }

    /**
     * Assemble map data to Encog standard input format. If no variable selected(noVarSel = true), all candidate
     * variables will be selected.
     * 
     * @throws NullPointerException
     *             if input is null
     * @throws NumberFormatException
     *             if column value is not number format.
     */
    public static MLDataPair assembleDataPair(Map<Integer, Map<String, Integer>> binCategoryMap, boolean noVarSel,
            ModelConfig modelConfig, List<ColumnConfig> columnConfigList, Map<String, ? extends Object> rawDataMap,
            double cutoff) {
        double[] ideal = { Constants.DEFAULT_IDEAL_VALUE };

        List<Double> inputList = new ArrayList<Double>();
        for(ColumnConfig config: columnConfigList) {
            if(config == null) {
                continue;
            }
            String key = config.getColumnName();
            if(config.isFinalSelect() && !rawDataMap.containsKey(key)) {
                throw new IllegalStateException(String.format("Variable Missing in Test Data: %s", key));
            }

            if(config.isTarget()) {
                continue;
            } else {
                if(!noVarSel) {
                    if(config != null && !config.isMeta() && !config.isTarget() && config.isFinalSelect()) {
                        String val = rawDataMap.get(key) == null ? null : rawDataMap.get(key).toString();
                        if(CommonUtils.isDesicionTreeAlgorithm(modelConfig.getAlgorithm()) && config.isCategorical()) {
                            Integer index = binCategoryMap.get(config.getColumnNum()).get(val == null ? "" : val);
                            if(index == null) {
                                // not in binCategories, should be missing value
                                // -1 as missing value
                                inputList.add(-1d);
                            } else {
                                inputList.add(index * 1d);
                            }
                        } else {
                            inputList.add(computeNumericNormResult(modelConfig, cutoff, config, val));
                        }
                    }
                } else {
                    if(!config.isMeta() && !config.isTarget() && CommonUtils.isGoodCandidate(config)) {
                        String val = rawDataMap.get(key) == null ? null : rawDataMap.get(key).toString();
                        if(CommonUtils.isDesicionTreeAlgorithm(modelConfig.getAlgorithm()) && config.isCategorical()) {
                            Integer index = binCategoryMap.get(config.getColumnNum()).get(val == null ? "" : val);
                            if(index == null) {
                                // not in binCategories, should be missing value
                                // -1 as missing value
                                inputList.add(-1d);
                            } else {
                                inputList.add(index * 1d);
                            }
                        } else {
                            inputList.add(computeNumericNormResult(modelConfig, cutoff, config, val));
                        }
                    }
                }
            }
        }

        // god, Double [] cannot be casted to double[], toArray doesn't work
        int size = inputList.size();
        double[] input = new double[size];
        for(int i = 0; i < size; i++) {
            input[i] = inputList.get(i);
        }

        return new BasicMLDataPair(new BasicMLData(input), new BasicMLData(ideal));
    }

    private static double computeNumericNormResult(ModelConfig modelConfig, double cutoff, ColumnConfig config,
            String val) {
        Double normalizeValue = null;
        if(CommonUtils.isDesicionTreeAlgorithm(modelConfig.getAlgorithm())) {
            try {
                normalizeValue = Double.parseDouble(val);
            } catch (Exception e) {
                normalizeValue = Normalizer.defaultMissingValue(config);
            }
        } else {
            normalizeValue = Normalizer.normalize(config, val, cutoff, modelConfig.getNormalizeType());
        }
        return normalizeValue;
    }

    public static boolean isDesicionTreeAlgorithm(String alg) {
        return CommonConstants.RF_ALG_NAME.equalsIgnoreCase(alg) || CommonConstants.GBT_ALG_NAME.equalsIgnoreCase(alg);
    }

    public static boolean isHadoopConfigurationInjected(String key) {
        return key.startsWith("nn") || key.startsWith("guagua") || key.startsWith("shifu") || key.startsWith("mapred")
                || key.startsWith("io") || key.startsWith("hadoop") || key.startsWith("yarn");
    }

    /**
     * Assemble map data to Encog standard input format.
     * 
     * @throws NullPointerException
     *             if input is null
     * @throws NumberFormatException
     *             if column value is not number format.
     */
    public static MLDataPair assembleDataPair(ModelConfig modelConfig, List<ColumnConfig> columnConfigList,
            Map<String, ? extends Object> rawDataMap, double cutoff) {
        // if the tag is provided, ideal will be updated; otherwise it defaults to -1
        double[] ideal = { Constants.DEFAULT_IDEAL_VALUE };

        List<Double> inputList = new ArrayList<Double>();
        for(ColumnConfig config: columnConfigList) {
            String key = config.getColumnName();
            if(config.isFinalSelect() && !rawDataMap.containsKey(key)) {
                throw new IllegalStateException(String.format("Variable Missing in Test Data: %s", key));
            }

            if(config.isTarget()) {
                // TODO - should we have this? maybe not
                // ideal[0] = Double.valueOf(rawDataMap.get(key).toString());
                continue;
            } else if(config.isFinalSelect()) {
                // add log for debug purpose
                // log.info("key: " + key + ", raw_value " + rawDataMap.get(key).toString() + ", zscl_value: " +
                String val = rawDataMap.get(key) == null ? null : rawDataMap.get(key).toString();
                Double normalizeValue = Normalizer.normalize(config, val, cutoff, modelConfig.getNormalizeType());
                inputList.add(normalizeValue);
            }
        }

        // god, Double [] cannot be casted to double[], toArray doesn't work
        int size = inputList.size();
        double[] input = new double[size];
        for(int i = 0; i < size; i++) {
            input[i] = inputList.get(i);
        }

        return new BasicMLDataPair(new BasicMLData(input), new BasicMLData(ideal));
    }

    /**
     * Expanding score by expandingFactor
     */
    public static long getExpandingScore(double d, int expandingFactor) {
        return Math.round(d * expandingFactor);
    }

    /**
     * Return column name string with 'derived_' started
     * 
     * @throws NullPointerException
     *             if modelConfig is null or columnConfigList is null.
     */
    public static List<String> getDerivedColumnNames(List<ColumnConfig> columnConfigList) {
        List<String> derivedColumnNames = new ArrayList<String>();

        for(ColumnConfig config: columnConfigList) {
            if(config.getColumnName().startsWith(Constants.DERIVED)) {
                derivedColumnNames.add(config.getColumnName());
            }
        }
        return derivedColumnNames;
    }

    /**
     * Get the file separator regex
     * 
     * @return "/" - if the OS is Linux
     *         "\\\\" - if the OS is Windows
     */
    public static String getPathSeparatorRegx() {
        if(File.separator.equals(Constants.SLASH)) {
            return File.separator;
        } else {
            return Constants.BACK_SLASH + File.separator;
        }
    }

    /**
     * Update target, listMeta, listForceSelect, listForceRemove
     * 
     * @throws IOException
     * @throws IllegalArgumentException
     *             if modelConfig is null or columnConfigList is null.
     */
    public static void updateColumnConfigFlags(ModelConfig modelConfig, List<ColumnConfig> columnConfigList)
            throws IOException {
        String targetColumnName = CommonUtils.getRelativePigHeaderColumnName(modelConfig.getTargetColumnName());

        Set<String> setCategorialColumns = new HashSet<String>();
        if(CollectionUtils.isNotEmpty(modelConfig.getCategoricalColumnNames())) {
            for(String column: modelConfig.getCategoricalColumnNames()) {
                setCategorialColumns.add(CommonUtils.getRelativePigHeaderColumnName(column));
            }
        }

        Set<String> setMeta = new HashSet<String>();
        if(CollectionUtils.isNotEmpty(modelConfig.getMetaColumnNames())) {
            for(String meta: modelConfig.getMetaColumnNames()) {
                setMeta.add(CommonUtils.getRelativePigHeaderColumnName(meta));
            }
        }

        Set<String> setForceRemove = new HashSet<String>();
        if(Boolean.TRUE.equals(modelConfig.getVarSelect().getForceEnable())
                && CollectionUtils.isNotEmpty(modelConfig.getListForceRemove())) {
            // if we need to update force remove, only and if one the force is enabled
            for(String forceRemoveName: modelConfig.getListForceRemove()) {
                setForceRemove.add(CommonUtils.getRelativePigHeaderColumnName(forceRemoveName));
            }
        }

        Set<String> setForceSelect = new HashSet<String>(512);
        if(Boolean.TRUE.equals(modelConfig.getVarSelect().getForceEnable())
                && CollectionUtils.isNotEmpty(modelConfig.getListForceSelect())) {
            // if we need to update force select, only and if one the force is enabled
            for(String forceSelectName: modelConfig.getListForceSelect()) {
                setForceSelect.add(CommonUtils.getRelativePigHeaderColumnName(forceSelectName));
            }
        }

        for(ColumnConfig config: columnConfigList) {
            String varName = config.getColumnName();

            if(targetColumnName.equals(varName)) {
                config.setColumnFlag(ColumnFlag.Target);
                config.setColumnType(null);
            } else if(setMeta.contains(varName)) {
                config.setColumnFlag(ColumnFlag.Meta);
                config.setColumnType(null);
            } else if(setForceRemove.contains(varName)) {
                config.setColumnFlag(ColumnFlag.ForceRemove);
            } else if(setForceSelect.contains(varName)) {
                config.setColumnFlag(ColumnFlag.ForceSelect);
            }

            // variable type is not related with variable flag
            if(setCategorialColumns.contains(varName)) {
                config.setColumnType(ColumnType.C);
            }
        }
    }

    /**
     * To check whether there is targetColumn in columns or not
     * 
     * @return true - if the columns contains targetColumn, or false
     */
    public static boolean isColumnExists(String[] columns, String targetColunm) {
        if(ArrayUtils.isEmpty(columns) || StringUtils.isBlank(targetColunm)) {
            return false;
        }

        for(int i = 0; i < columns.length; i++) {
            if(columns[i] != null && columns[i].equalsIgnoreCase(targetColunm)) {
                return true;
            }
        }

        return false;
    }

    /**
     * Returns the element if it is in both collections.
     * - return null if any collection is null or empty
     * - return null if no element exists in both collections
     * 
     * @param leftCol
     *            - left collection
     * @param rightCol
     *            - right collection
     * @return First element that are found in both collections
     *         null if no elements in both collection or any collection is null or empty
     */
    public static <T> T containsAny(Collection<T> leftCol, Collection<T> rightCol) {
        if(CollectionUtils.isEmpty(leftCol) || CollectionUtils.isEmpty(rightCol)) {
            return null;
        }

        Iterator<T> iterator = leftCol.iterator();
        while(iterator.hasNext()) {
            T element = iterator.next();
            if(rightCol.contains(element)) {
                return element;
            }
        }

        return null;
    }

    /**
     * Escape the delimiter for Pig.... Since the Pig doesn't support invisible character
     * 
     * @param delimiter
     *            - the original delimiter
     * @return the delimiter after escape
     */
    public static String escapePigString(String delimiter) {
        StringBuffer buf = new StringBuffer();

        for(int i = 0; i < delimiter.length(); i++) {
            char c = delimiter.charAt(i);
            switch(c) {
                case '\t':
                    buf.append("\\\\t");
                    break;
                default:
                    buf.append(c);
                    break;
            }
        }

        return buf.toString();
    }

    /**
     * @param columnConfFile
     * @param delimiter
     * @return
     * @throws IOException
     */
    public static List<String> readConfFileIntoList(String columnConfFile, SourceType sourceType, String delimiter)
            throws IOException {
        List<String> columnNameList = new ArrayList<String>();

        if(StringUtils.isBlank(columnConfFile) || !ShifuFileUtils.isFileExists(columnConfFile, sourceType)) {
            return columnNameList;
        }

        List<String> strList = null;
        Reader reader = ShifuFileUtils.getReader(columnConfFile, sourceType);
        try {
            strList = IOUtils.readLines(reader);
        } finally {
            IOUtils.closeQuietly(reader);
        }

        if(CollectionUtils.isNotEmpty(strList)) {
            for(String line: strList) {
                if(line.trim().equals("") || line.trim().startsWith("#")) {
                    continue;
                }

                for(String str: Splitter.on(delimiter).split(line)) {
                    String column = CommonUtils.getRelativePigHeaderColumnName(str);
                    if(StringUtils.isNotBlank(column)) {
                        columnNameList.add(column.trim());
                    }
                }
            }
        }

        return columnNameList;
    }

    /**
     * Generate seat info for selected column in @columnConfigList
     * 
     * @param columnConfigList
     * @return
     */
    public static Map<String, Integer> generateColumnSeatMap(List<ColumnConfig> columnConfigList) {
        List<ColumnConfig> selectedColumnList = new ArrayList<ColumnConfig>();
        for(ColumnConfig columnConfig: columnConfigList) {
            if(columnConfig.isFinalSelect()) {
                selectedColumnList.add(columnConfig);
            }
        }
        Collections.sort(selectedColumnList, new Comparator<ColumnConfig>() {
            @Override
            public int compare(ColumnConfig from, ColumnConfig to) {
                return from.getColumnName().compareTo(to.getColumnName());
            }

        });

        Map<String, Integer> columnSeatMap = new HashMap<String, Integer>();
        for(int i = 0; i < selectedColumnList.size(); i++) {
            columnSeatMap.put(selectedColumnList.get(i).getColumnName(), i);
        }

        return columnSeatMap;
    }

    /**
     * Find the @ColumnConfig according the column name
     * 
     * @param columnConfigList
     * @param columnName
     * @return
     */
    public static ColumnConfig findColumnConfigByName(List<ColumnConfig> columnConfigList, String columnName) {
        for(ColumnConfig columnConfig: columnConfigList) {
            if(columnConfig.getColumnName().equalsIgnoreCase(columnName)) {
                return columnConfig;
            }
        }
        return null;
    }

    /**
     * Convert data into <key, value> map. The @inputData is String of a record, which is delimited by @delimiter
     * If fields in @inputData is not equal @header size, return null
     * 
     * @param inputData
     *            - String of a record
     * @param delimiter
     *            - the delimiter of the input data
     * @param header
     *            - the column names for all the input data
     * @return <key, value> map for the record
     */
    public static Map<String, String> convertDataIntoMap(String inputData, String delimiter, String[] header) {
        String[] input = CommonUtils.split(inputData, delimiter);
        if(input == null || input.length == 0 || input.length != header.length) {
            log.error("the wrong input data, {}", inputData);
            return null;
        }

        Map<String, String> rawDataMap = new HashMap<String, String>(input.length);
        for(int i = 0; i < header.length; i++) {
            if(input[i] == null) {
                rawDataMap.put(header[i], "");
            } else {
                rawDataMap.put(header[i], input[i]);
            }
        }

        return rawDataMap;
    }

    /**
     * Convert tuple record into <key, value> map. The @tuple is Tuple for a record
     * If @tuple size is not equal @header size, return null
     * 
     * @param tuple
     *            - Tuple of a record
     * @param header
     *            - the column names for all the input data
     * @return <key, value> map for the record
     * @throws ExecException
     *             - throw exception when operating tuple
     */
    public static Map<String, String> convertDataIntoMap(Tuple tuple, String[] header) throws ExecException {
        if(tuple == null || tuple.size() == 0 || tuple.size() != header.length) {
            log.error("Invalid input, the tuple.size is = " + (tuple == null ? null : tuple.size())
                    + ", header.length = " + header.length);
            return null;
        }

        Map<String, String> rawDataMap = new HashMap<String, String>(tuple.size());
        for(int i = 0; i < header.length; i++) {
            if(tuple.get(i) == null) {
                rawDataMap.put(header[i], "");
            } else {
                rawDataMap.put(header[i], tuple.get(i).toString());
            }
        }

        return rawDataMap;
    }

    public static boolean isGoodCandidate(boolean isBinaryClassification, ColumnConfig columnConfig) {
        if(columnConfig == null) {
            return false;
        }

        if(isBinaryClassification) {
            return columnConfig.isCandidate()
                    && (columnConfig.getKs() != null && columnConfig.getKs() > 0 && columnConfig.getIv() != null
                            && columnConfig.getIv() > 0 && columnConfig.getMean() != null
                            && columnConfig.getStdDev() != null && ((columnConfig.isCategorical()
                            && columnConfig.getBinCategory() != null && columnConfig.getBinCategory().size() > 1) || (columnConfig
                            .isNumerical() && columnConfig.getBinBoundary() != null && columnConfig.getBinBoundary()
                            .size() > 1)));
        } else {
            // multiple classification
            return columnConfig.isCandidate()
                    && (columnConfig.getMean() != null && columnConfig.getStdDev() != null && ((columnConfig
                            .isCategorical() && columnConfig.getBinCategory() != null && columnConfig.getBinCategory()
                            .size() > 1) || (columnConfig.isNumerical() && columnConfig.getBinBoundary() != null && columnConfig
                            .getBinBoundary().size() > 1)));
        }
    }

    public static boolean isGoodCandidate(ColumnConfig columnConfig) {
        if(columnConfig == null) {
            return false;
        }

        return columnConfig.isCandidate()
                && (columnConfig.getKs() != null && columnConfig.getKs() > 0 && columnConfig.getIv() != null
                        && columnConfig.getIv() > 0 && columnConfig.getMean() != null
                        && columnConfig.getStdDev() != null && ((columnConfig.isCategorical()
                        && columnConfig.getBinCategory() != null && columnConfig.getBinCategory().size() > 1) || (columnConfig
                        .isNumerical() && columnConfig.getBinBoundary() != null && columnConfig.getBinBoundary().size() > 1)));
    }

    /**
     * Return first line split string array. This is used to detect data schema.
     */
    public static String[] takeFirstLine(String dataSetRawPath, String headerDelimiter, SourceType source)
            throws IOException {
        if(dataSetRawPath == null || headerDelimiter == null || source == null) {
            throw new IllegalArgumentException("Input parameters should not be null.");
        }

        String firstValidFile = null;
        if(ShifuFileUtils.isDir(dataSetRawPath, source)) {
            FileSystem fs = ShifuFileUtils.getFileSystemBySourceType(source);
            FileStatus[] globStatus = fs.globStatus(new Path(dataSetRawPath), HIDDEN_FILE_FILTER);
            if(globStatus == null || globStatus.length == 0) {
                throw new IllegalArgumentException("No files founded in " + dataSetRawPath);
            } else {
                FileStatus[] listStatus = fs.listStatus(globStatus[0].getPath(), HIDDEN_FILE_FILTER);
                if(listStatus == null || listStatus.length == 0) {
                    throw new IllegalArgumentException("No files founded in " + globStatus[0].getPath());
                }
                firstValidFile = listStatus[0].getPath().toString();
            }
        } else {
            firstValidFile = dataSetRawPath;
        }

        BufferedReader reader = null;
        try {
            reader = ShifuFileUtils.getReader(firstValidFile, source);
            String firstLine = reader.readLine();
            if(firstLine != null && firstLine.length() > 0) {
                List<String> list = new ArrayList<String>();
                for(String unit: Splitter.on(headerDelimiter).split(firstLine)) {
                    list.add(unit);
                }
                return list.toArray(new String[0]);
            }
        } finally {
            IOUtils.closeQuietly(reader);
        }
        return new String[0];
    }

    private static final PathFilter HIDDEN_FILE_FILTER = new PathFilter() {
        public boolean accept(Path p) {
            String name = p.getName();
            return !name.startsWith("_") && !name.startsWith(".");
        }
    };

    public static String genPigFieldName(String name) {
        return ((name != null) ? name.replace('-', '_') : null);
    }

    public static String[] genPigFieldName(String[] names) {
        String[] pigScoreNames = new String[names.length];
        for(int i = 0; i < names.length; i++) {
            pigScoreNames[i] = genPigFieldName(names[i]);
        }
        return pigScoreNames;
    }
}