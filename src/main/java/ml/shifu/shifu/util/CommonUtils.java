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

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataInputStream;
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
import java.util.TreeMap;
import java.util.zip.GZIPInputStream;

import ml.shifu.shifu.column.NSColumn;
import ml.shifu.shifu.column.NSColumnUtils;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ColumnConfig.ColumnFlag;
import ml.shifu.shifu.container.obj.ColumnType;
import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelTrainConf.ALGORITHM;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.LR;
import ml.shifu.shifu.core.NNModel;
import ml.shifu.shifu.core.Normalizer;
import ml.shifu.shifu.core.TreeModel;
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
import org.apache.commons.collections.Predicate;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.commons.lang.math.NumberUtils;
import org.apache.commons.lang3.tuple.MutablePair;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.fs.RemoteIterator;
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

import com.google.common.base.Function;
import com.google.common.base.Splitter;
import com.google.common.collect.Collections2;
import com.google.common.collect.Lists;

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
     * @param modelConfig
     *            the model config
     * @param pathFinder
     *            the path finder to locate file
     * @return if copy successful
     * 
     * @throws IOException
     *             If any exception on HDFS IO or local IO.
     * 
     * @throws NullPointerException
     *             If parameter {@code modelConfig} is null
     */
    public static boolean copyConfFromLocalToHDFS(ModelConfig modelConfig, PathFinder pathFinder) throws IOException {
        FileSystem hdfs = HDFSUtils.getFS();
        FileSystem localFs = HDFSUtils.getLocalFS();

        Path pathModelSet = new Path(pathFinder.getModelSetPath(SourceType.HDFS));
        // don't check whether pathModelSet is exists, should be remove by user.
        hdfs.mkdirs(pathModelSet);

        // Copy ModelConfig
        Path srcModelConfig = new Path(pathFinder.getModelConfigPath(SourceType.LOCAL));
        Path dstModelConfig = new Path(pathFinder.getModelSetPath(SourceType.HDFS));
        hdfs.copyFromLocalFile(srcModelConfig, dstModelConfig);

        // Copy GridSearch config file if exists
        String gridConfigFile = modelConfig.getTrain().getGridConfigFile();
        if(gridConfigFile != null && !gridConfigFile.trim().equals("")) {
            Path srcGridConfig = new Path(modelConfig.getTrain().getGridConfigFile());
            Path dstGridConfig = new Path(pathFinder.getModelSetPath(SourceType.HDFS));
            hdfs.copyFromLocalFile(srcGridConfig, dstGridConfig);
        }

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
     * Sync-up the evaluation data into HDFS
     * 
     * @param modelConfig
     *            - ModelConfig
     * @param evalName
     *            eval name in ModelConfig
     * @throws IOException
     *             - error occur when copying data
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

            // sync evaluation meta.column.file to hdfs
            if(StringUtils.isNotBlank(evalConfig.getDataSet().getMetaColumnNameFile())) {
                hdfs.copyFromLocalFile(new Path(evalConfig.getDataSet().getMetaColumnNameFile()),
                        new Path(pathFinder.getEvalSetPath(evalConfig)));
            }
        }
    }

    public static String getLocalModelSetPath(Map<String, Object> otherConfigs) {
        if(otherConfigs != null && otherConfigs.get(Constants.SHIFU_CURRENT_WORKING_DIR) != null) {
            return new Path(otherConfigs.get(Constants.SHIFU_CURRENT_WORKING_DIR).toString()).toString();
        } else {
            return ".";
        }
    }

    /**
     * Load ModelConfig from local json ModelConfig.json file.
     * 
     * @return model config instance from default model config file
     * @throws IOException
     *             any io exception to load file
     */
    public static ModelConfig loadModelConfig() throws IOException {
        return loadModelConfig(Constants.LOCAL_MODEL_CONFIG_JSON, SourceType.LOCAL);
    }

    /**
     * Load model configuration from the path and the source type.
     * 
     * @param path
     *            model file path
     * @param sourceType
     *            source type of model file
     * @return model config instance
     * @throws IOException
     *             if any IO exception in parsing json.
     * 
     * @throws IllegalArgumentException
     *             if {@code path} is null or empty, if sourceType is null.
     */
    public static ModelConfig loadModelConfig(String path, SourceType sourceType) throws IOException {
        ModelConfig modelConfig = loadJSON(path, sourceType, ModelConfig.class);
        if(StringUtils.isNotBlank(modelConfig.getTrain().getGridConfigFile())) {
            String gridConfigPath = modelConfig.getTrain().getGridConfigFile().trim();
            if(sourceType == SourceType.HDFS) {
                // gridsearch config file is uploaded to modelset path
                gridConfigPath = new PathFinder(modelConfig).getPathBySourceType(
                        gridConfigPath.substring(gridConfigPath.lastIndexOf(File.separator) + 1), SourceType.HDFS);
            }
            // Only load file content. Grid search params parsing is done in {@link GridSearch} initialization.
            modelConfig.getTrain().setGridConfigFileContent(loadFileContent(gridConfigPath, sourceType));
        }
        return modelConfig;
    }

    /**
     * Load text file content, each line as a String in the List.
     * 
     * @param path
     *            file path
     * @param sourceType
     *            source type: hdfs or local
     * @return file content as a {@link List}, each line as a String
     * @throws IOException
     *             if any IO exception in reading file content.
     */
    private static List<String> loadFileContent(String path, SourceType sourceType) throws IOException {
        checkPathAndMode(path, sourceType);
        log.debug("loading {} with sourceType {}", path, sourceType);
        BufferedReader reader = null;
        try {
            reader = ShifuFileUtils.getReader(path, sourceType);
            List<String> contents = new ArrayList<String>();
            String line = null;
            while((line = reader.readLine()) != null) {
                contents.add(line);
            }
            return contents;
        } finally {
            IOUtils.closeQuietly(reader);
        }
    }

    private static void checkPathAndMode(String path, SourceType sourceType) {
        if(StringUtils.isEmpty(path) || sourceType == null) {
            throw new IllegalArgumentException(String.format(
                    "path should not be null or empty, sourceType should not be null, path:%s, sourceType:%s", path,
                    sourceType));
        }
    }

    /**
     * Load reason code map and change it to column &gt; resonCode map.
     * 
     * @param path
     *            reason code path
     * @param sourceType
     *            source type of file
     * @return reason code map
     * @throws IOException
     *             if any IO exception in parsing json.
     * 
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
     * @param path
     *            file path
     * @param sourceType
     *            source type: hdfs or local
     * @param clazz
     *            class of instance
     * @param <T>
     *            class type to load
     * @return instance from json file
     * @throws IOException
     *             if any IO exception in parsing json.
     * 
     * @throws IllegalArgumentException
     *             if {@code path} is null or empty, if sourceType is null.
     */
    public static <T> T loadJSON(String path, SourceType sourceType, Class<T> clazz) throws IOException {
        checkPathAndMode(path, sourceType);
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
     * @return column config list
     * @throws IOException
     *             if any IO exception in parsing json.
     */
    public static List<ColumnConfig> loadColumnConfigList() throws IOException {
        List<ColumnConfig> columnConfigList = loadColumnConfigList(Constants.LOCAL_COLUMN_CONFIG_JSON, SourceType.LOCAL);
        for(ColumnConfig columnConfig: columnConfigList) {
            columnConfig.setSampleValues(null);
        }
        return columnConfigList;
    }

    /**
     * Load column configuration list.
     * 
     * @param path
     *            file path
     * @param sourceType
     *            source type: hdfs or local
     * @return column config list
     * @throws IOException
     *             if any IO exception in parsing json.
     * @throws IllegalArgumentException
     *             if {@code path} is null or empty, if sourceType is null.
     */
    public static List<ColumnConfig> loadColumnConfigList(String path, SourceType sourceType) throws IOException {
        return loadColumnConfigList(path, sourceType, true);
    }

    /**
     * Load column configuration list.
     * 
     * @param path
     *            file path
     * @param sourceType
     *            source type: hdfs or local
     * @param nullSampleValues
     *            if sample values null or not to save memory especially in Pig UDF to save more memory. there is a OOM
     *            if larger ColumnConfig.json.
     * @return column config list
     * @throws IOException
     *             if any IO exception in parsing json.
     * @throws IllegalArgumentException
     *             if {@code path} is null or empty, if sourceType is null.
     */
    public static List<ColumnConfig> loadColumnConfigList(String path, SourceType sourceType, boolean nullSampleValues)
            throws IOException {
        ColumnConfig[] configList = loadJSON(path, sourceType, ColumnConfig[].class);
        List<ColumnConfig> columnConfigList = new ArrayList<ColumnConfig>();
        for(ColumnConfig columnConfig: configList) {
            if(nullSampleValues) {
                columnConfig.setSampleValues(null);
            }
            columnConfigList.add(columnConfig);
        }
        return columnConfigList;
    }

    /**
     * Return final selected column collection.
     * 
     * @param columnConfigList
     *            column config list
     * @return collection of column config list for final select is true
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
            String delimiter = StringUtils.isBlank(evalConfig.getDataSet().getHeaderDelimiter()) ? evalConfig
                    .getDataSet().getDataDelimiter() : evalConfig.getDataSet().getHeaderDelimiter();
            fields = CommonUtils.getHeaders(evalConfig.getDataSet().getHeaderPath(), delimiter, evalConfig.getDataSet()
                    .getSource());
        } else {
            fields = CommonUtils.takeFirstLine(evalConfig.getDataSet().getDataPath(), StringUtils.isBlank(evalConfig
                    .getDataSet().getHeaderDelimiter()) ? evalConfig.getDataSet().getDataDelimiter() : evalConfig
                    .getDataSet().getHeaderDelimiter(), evalConfig.getDataSet().getSource());
            // TODO - if there is no target column in eval, it may fail to check it is schema or not
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
            } /*
               * else { // namespace support
               * fields[i] = getRelativePigHeaderColumnName(fields[i]);
               * }
               */
        }
        return fields;
    }

    /**
     * Return header column list from header file.
     * 
     * @param pathHeader
     *            header path
     * @param delimiter
     *            the delimiter of headers
     * @param sourceType
     *            source type: hdfs or local
     * @return headers array
     * @throws IOException
     *             if any IO exception in reading file.
     * 
     * @throws IllegalArgumentException
     *             if sourceType is null, if pathHeader is null or empty, if delimiter is null or empty.
     * 
     * @throws RuntimeException
     *             if first line of pathHeader is null or empty.
     */
    public static String[] getHeaders(String pathHeader, String delimiter, SourceType sourceType) throws IOException {
        return getHeaders(pathHeader, delimiter, sourceType, false);
    }

    /**
     * Return header column array from header file.
     * 
     * @param pathHeader
     *            header path
     * @param delimiter
     *            the delimiter of headers
     * @param sourceType
     *            source type: hdfs or local
     * @param isFull
     *            if full header name including name space
     * @return headers array
     * @throws IOException
     *             if any IO exception in reading file.
     * 
     * @throws IllegalArgumentException
     *             if sourceType is null, if pathHeader is null or empty, if delimiter is null or empty.
     * 
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
            String columnName = StringUtils.trimToEmpty(str);
            if(!Environment.getBoolean(Constants.SHIFU_NAMESPACE_STRICT_MODE, false)) {
                columnName = getRelativePigHeaderColumnName(str);
            }
            /*
             * if(isFull) {
             * columnName = getFullPigHeaderColumnName(str);
             * } else {
             * columnName = getRelativePigHeaderColumnName(str);
             * }
             */
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
     * 
     * @param raw
     *            raw name
     * @return full name including namespace
     */
    public static String getFullPigHeaderColumnName(String raw) {
        return raw == null ? raw : raw.replaceAll(Constants.PIG_COLUMN_SEPARATOR, Constants.PIG_FULL_COLUMN_SEPARATOR);
    }

    /**
     * Get relative column name from pig header. For example, one column is a::b, return b. If b, return b.
     * 
     * @param raw
     *            raw name
     * @return relative name including namespace
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
     * @param columnConfig
     *            column config
     * @param columnVal
     *            value of the column
     * @return bin index of than value
     * @throws IllegalArgumentException
     *             if input is null or empty.
     * 
     * @throws NumberFormatException
     *             if columnVal does not contain a parsable number.
     */
    public static int getBinNum(ColumnConfig columnConfig, String columnVal) {
        if(columnConfig.isCategorical()) {
            return getCategoicalBinIndex(columnConfig.getBinCategory(), columnVal);
        } else {
            return getNumericalBinIndex(columnConfig.getBinBoundary(), columnVal);
        }
    }

    /**
     * Get numerical bin index according to string column value.
     * 
     * @param binBoundaries
     *            the bin boundaries
     * @param columnVal
     *            the column value
     * @return bin index, -1 if invalid values
     */
    public static int getNumericalBinIndex(List<Double> binBoundaries, String columnVal) {
        if(StringUtils.isBlank(columnVal)) {
            return -1;
        }
        double dval = 0.0;
        try {
            dval = Double.parseDouble(columnVal);
        } catch (Exception e) {
            return -1;
        }
        return getBinIndex(binBoundaries, dval);
    }

    /**
     * Get categorical bin index according to string column value.
     * 
     * @param binCategories
     *            the bin categories
     * @param columnVal
     *            the column value
     * @return bin index, -1 if invalid values
     */
    public static int getCategoicalBinIndex(List<String> binCategories, String columnVal) {
        if(StringUtils.isBlank(columnVal)) {
            return -1;
        }
        for(int i = 0; i < binCategories.size(); i++) {
            if(isCategoricalBinValue(binCategories.get(i), columnVal)) {
                return i;
            }
        }
        return -1;
    }

    /**
     * Check some categorical value is in the categorical value group or not
     * 
     * @param binVal
     *            - categorical value group, the format is lik cn^us^uk^jp
     * @param cval
     *            - categorical value to look up
     * @return true if the categorical value exists in group, else false
     */
    public static boolean isCategoricalBinValue(String binVal, String cval) {
        // TODO cache CommonUtils.flattenCatValGrp(binVal)??
        return binVal.equals(cval) ? true : CommonUtils.flattenCatValGrp(binVal).contains(cval);
    }

    /**
     * Return the real bin number for one value. As the first bin value is NEGATIVE_INFINITY, invalid index is 0, not
     * -1.
     * 
     * @param binBoundary
     *            bin boundary list which should be sorted.
     * @param value
     *            value of column
     * @return bin index
     * 
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
     * @param raw
     *            raw string
     * @param delimiter
     *            the delimeter to split the string
     * @return array of split Strings
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
     * @param raw
     *            raw string
     * @param delimiter
     *            the delimeter to split the string
     * @return list of split Strings
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
     * @param columnConfigList
     *            column config list
     * @return target column index
     * @throws IllegalArgumentException
     *             if columnConfigList is null or empty.
     * 
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
     * @param modelConfig
     *            ModelConfig
     * @param columnConfigList
     *            column config list
     * @param evalConfig
     *            eval config instance
     * @return the list of models
     * @throws IOException
     *             if any IO exception in reading model file.
     * 
     * @throws IllegalArgumentException
     *             if {@code modelConfig} is, if invalid model algorithm .
     * 
     * @throws IllegalStateException
     *             if not HDFS or LOCAL source type or algorithm not supported.
     */
    public static List<BasicML> loadBasicModels(ModelConfig modelConfig, List<ColumnConfig> columnConfigList,
            EvalConfig evalConfig) throws IOException {
        if(modelConfig == null
                || (!Constants.NN.equalsIgnoreCase(modelConfig.getAlgorithm())
                        && !Constants.SVM.equalsIgnoreCase(modelConfig.getAlgorithm())
                        && !Constants.LR.equalsIgnoreCase(modelConfig.getAlgorithm()) && !CommonUtils
                            .isTreeModel(modelConfig.getAlgorithm()))) {
            throw new IllegalArgumentException(modelConfig == null ? "modelConfig is null." : String.format(
                    " invalid model algorithm %s.", modelConfig.getAlgorithm()));
        }

        return loadBasicModels(modelConfig, evalConfig, modelConfig.getDataSet().getSource());
    }

    /**
     * Get bin index by binary search. The last bin in <code>binBoundary</code> is missing value bin.
     * 
     * @param binBoundary
     *            bin boundary list which should be sorted.
     * @param dVal
     *            value of column
     * @return bin index
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

    public static List<BasicML> loadBasicModels(ModelConfig modelConfig, EvalConfig evalConfig, SourceType sourceType)
            throws IOException {
        List<BasicML> models = new ArrayList<BasicML>();
        FileSystem fs = ShifuFileUtils.getFileSystemBySourceType(sourceType);

        List<FileStatus> modelFileStats = locateBasicModels(modelConfig, evalConfig, sourceType);
        if(CollectionUtils.isNotEmpty(modelFileStats)) {
            for(FileStatus f: modelFileStats) {
                models.add(loadModel(modelConfig, f.getPath(), fs));
            }
        }

        return models;
    }

    public static BasicNetwork getBasicNetwork(BasicML model) {
        if(model instanceof BasicFloatNetwork) {
            return (BasicFloatNetwork) model;
        } else if(model instanceof NNModel) {
            return ((NNModel) model).getIndependentNNModel().getBasicNetworks().get(0);
        }
        throw new IllegalArgumentException("Only nn model is supported");
    }

    /**
     * Load basic models from files.
     * 
     * @param modelConfig
     *            model config
     * @param evalConfig
     *            eval confg
     * @param sourceType
     *            source type
     * @param gbtConvertToProb
     *            convert gbt score to prob or not
     * @return list of models
     * @throws IOException
     *             if any IO exception in reading model file.
     * 
     * @throws IllegalArgumentException
     *             if {@code modelConfig} is, if invalid model algorithm .
     * 
     * @throws IllegalStateException
     *             if not HDFS or LOCAL source type or algorithm not supported.
     */
    public static List<BasicML> loadBasicModels(ModelConfig modelConfig, EvalConfig evalConfig, SourceType sourceType,
            boolean gbtConvertToProb) throws IOException {
        return loadBasicModels(modelConfig, evalConfig, sourceType, gbtConvertToProb, null);
    }

    /**
     * Load basic models from files.
     * 
     * @param modelConfig
     *            model config
     * @param evalConfig
     *            eval confg
     * @param sourceType
     *            source type
     * @param gbtConvertToProb
     *            convert gbt score to prob or not
     * @param gbtScoreConvertStrategy
     *            specify how to convert gbt raw score
     * @return list of models
     * @throws IOException
     *             if any IO exception in reading model file.
     * 
     * @throws IllegalArgumentException
     *             if {@code modelConfig} is, if invalid model algorithm .
     * 
     * @throws IllegalStateException
     *             if not HDFS or LOCAL source type or algorithm not supported.
     */
    public static List<BasicML> loadBasicModels(ModelConfig modelConfig, EvalConfig evalConfig, SourceType sourceType,
            boolean gbtConvertToProb, String gbtScoreConvertStrategy) throws IOException {
        List<BasicML> models = new ArrayList<BasicML>();
        FileSystem fs = ShifuFileUtils.getFileSystemBySourceType(sourceType);

        List<FileStatus> modelFileStats = locateBasicModels(modelConfig, evalConfig, sourceType);
        if(CollectionUtils.isNotEmpty(modelFileStats)) {
            for(FileStatus f: modelFileStats) {
                models.add(loadModel(modelConfig, f.getPath(), fs, gbtConvertToProb, gbtScoreConvertStrategy));
            }
        }

        return models;
    }

    public static int getBasicModelsCnt(ModelConfig modelConfig, EvalConfig evalConfig, SourceType sourceType)
            throws IOException {
        List<FileStatus> modelFileStats = locateBasicModels(modelConfig, evalConfig, sourceType);
        return (CollectionUtils.isEmpty(modelFileStats) ? 0 : modelFileStats.size());
    }

    public static List<FileStatus> locateBasicModels(ModelConfig modelConfig, EvalConfig evalConfig,
            SourceType sourceType) throws IOException {
        // we have to register PersistBasicFloatNetwork for loading such models
        PersistorRegistry.getInstance().add(new PersistBasicFloatNetwork());

        List<FileStatus> listStatus = findModels(modelConfig, evalConfig, sourceType);
        if(CollectionUtils.isEmpty(listStatus)) {
            // throw new ShifuException(ShifuErrorCode.ERROR_MODEL_FILE_NOT_FOUND);
            // disable exception, since we there maybe sub-models
            return listStatus;
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

        Integer kCrossValidation = modelConfig.getTrain().getNumKFold();
        if(kCrossValidation != null && kCrossValidation > 0) {
            // if kfold is enabled , bagging set it to bagging model size
            baggingModelSize = kCrossValidation;
        }

        GridSearch gs = new GridSearch(modelConfig.getTrain().getParams(), modelConfig.getTrain()
                .getGridConfigFileContent());
        if(gs.hasHyperParam()) {
            // if it is grid search, set model size to all flatten params
            baggingModelSize = gs.getFlattenParams().size();
        }

        listStatus = listStatus.size() <= baggingModelSize ? listStatus : listStatus.subList(0, baggingModelSize);
        return listStatus;
    }

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
     * @return model object or null if no modelPath file,
     * 
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
     * 
     * @throws IOException
     *             if loading file for any IOException
     */
    public static BasicML loadModel(ModelConfig modelConfig, Path modelPath, FileSystem fs, boolean gbtConvertToProb,
            String gbtScoreConvertStrategy) throws IOException {
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
                return TreeModel.loadFromStream(stream, gbtConvertToProb, gbtScoreConvertStrategy);
            } else {
                GzipStreamPair pair = isGZipFormat(stream);
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
            if(br != null) {
                IOUtils.closeQuietly(br);
            }
            if(stream != null) {
                IOUtils.closeQuietly(stream);
            }
        }
    }

    /**
     * Get ColumnConfig from ColumnConfig list by columnId, since the columnId may not represent the position
     * in ColumnConfig list after the segments (Column Expansion).
     * @param columnConfigList - list of ColumnConfig
     * @param columnId - the column id that want to search
     * @return - ColumnConfig
     */
    public static ColumnConfig getColumnConfig(List<ColumnConfig> columnConfigList, Integer columnId) {
        for ( ColumnConfig columnConfig : columnConfigList ) {
            if ( columnConfig.getColumnNum().equals(columnId) ) {
                return columnConfig;
            }
        }
        return null;
    }

    public static class GzipStreamPair {

        private DataInputStream input;

        private boolean isGzip;

        public GzipStreamPair(DataInputStream input, boolean isGzip) {
            this.input = input;
            this.isGzip = isGzip;
        }

        /**
         * @return the input
         */
        public DataInputStream getInput() {
            return input;
        }

        /**
         * @param input
         *            the input to set
         */
        public void setInput(DataInputStream input) {
            this.input = input;
        }

        /**
         * @return the isGzip
         */
        public boolean isGzip() {
            return isGzip;
        }

        /**
         * @param isGzip
         *            the isGzip to set
         */
        public void setGzip(boolean isGzip) {
            this.isGzip = isGzip;
        }

    }

    private static GzipStreamPair isGZipFormat(InputStream input) {
        DataInputStream dis = null;
        // check if gzip or not
        boolean isGZip = false;
        try {
            byte[] header = new byte[2];
            BufferedInputStream bis = new BufferedInputStream(input);
            bis.mark(2);
            int result = bis.read(header);
            bis.reset();
            int ss = (header[0] & 0xff) | ((header[1] & 0xff) << 8);
            if(result != -1 && ss == GZIPInputStream.GZIP_MAGIC) {
                dis = new DataInputStream(new GZIPInputStream(bis));
                isGZip = true;
            } else {
                dis = new DataInputStream(bis);
                isGZip = false;
            }
        } catch (java.io.IOException e) {
            dis = new DataInputStream(input);
            isGZip = false;
        }
        return new GzipStreamPair(dis, isGZip);
    }

    /**
     * Find the model files for some @ModelConfig. There is a little tricky about this function.
     * If @EvalConfig is specified, try to load the models according setting in @EvalConfig,
     * or if @EvalConfig is null or ModelsPath is blank, Shifu will try to load models under `models`
     * directory
     * 
     * @param modelConfig
     *            - @ModelConfig, need this, since the model file may exist in HDFS
     * 
     * @param evalConfig
     *            - @EvalConfig, maybe null
     * 
     * @param sourceType
     *            - Where is file system
     * 
     * @return - @FileStatus array for all found models
     * 
     * @throws IOException
     *             io exception to load files
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

    public static List<ModelSpec> loadSubModels(ModelConfig modelConfig, List<ColumnConfig> columnConfigList,
            EvalConfig evalConfig, SourceType sourceType, Boolean gbtConvertToProb) {
        return loadSubModels(modelConfig, columnConfigList, evalConfig, sourceType, gbtConvertToProb, null);
    }

    @SuppressWarnings("deprecation")
    public static List<ModelSpec> loadSubModels(ModelConfig modelConfig, List<ColumnConfig> columnConfigList,
            EvalConfig evalConfig, SourceType sourceType, Boolean gbtConvertToProb, String gbtScoreConvertStrategy) {
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

    private static ModelSpec loadSubModelSpec(ModelConfig modelConfig, List<ColumnConfig> columnConfigList,
            FileStatus fileStatus, SourceType sourceType, Boolean gbtConvertToProb, String gbtScoreConvertStrategy)
            throws IOException {
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

    @SuppressWarnings("deprecation")
    public static ALGORITHM getModelsAlgAndSpecFiles(FileStatus fileStatus, SourceType sourceType,
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

    @SuppressWarnings("deprecation")
    public static Map<String, Integer> getSubModelsCnt(ModelConfig modelConfig, List<ColumnConfig> columnConfigList,
            EvalConfig evalConfig, SourceType sourceType) throws IOException {
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

    public static Set<NSColumn> loadCandidateColumns(ModelConfig modelConfig) throws IOException {
        Set<NSColumn> candidateColumns = new HashSet<NSColumn>();
        List<String> candidates = modelConfig.getListCandidates();
        for(String candidate: candidates) {
            candidateColumns.add(new NSColumn(candidate));
        }
        return candidateColumns;
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
     * 
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
                        GzipStreamPair pair = isGZipFormat(is);
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

    /**
     * Return one HashMap Object contains keys in the first parameter, values in the second parameter. Before calling
     * this method, you should be aware that headers should be unique.
     * 
     * @param header
     *            - header that contains column name
     * @param data
     *            - raw data
     * @return key-value map for variable
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
     * @param modelConfig
     *            model config
     * @param sourceType
     *            source type
     * @return map of configurations
     * @throws IOException
     *             any io exception
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
     * @param modelConfig
     *            model config
     * @param sourceType
     *            source type
     * @param pathFinder
     *            path finder instance
     * @return map of configurations
     * @throws IOException
     *             any io exception
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
     * @param str
     *            str to be split
     * @return list of double
     * @throws IllegalArgumentException
     *             if str is not a valid list str.
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
     * Change list str to List object with int type.
     * 
     * @param str
     *            str to be split
     * @return list of int
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
     * @param str
     *            str to be split
     * @return list of string
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
     * @param str
     *            str to be split
     * @param separator
     *            the separator
     * @return list of string
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

    /*
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
     * @param modelConfig
     *            model config instance
     * @param columnConfigList
     *            column config list
     * @param rawDataMap
     *            raw data
     * @return data pair instance
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
     * @param binCategoryMap
     *            categorical map
     * @param noVarSel
     *            if after var select
     * @param modelConfig
     *            model config instance
     * @param columnConfigList
     *            column config list
     * @param rawDataMap
     *            raw data
     * @param cutoff
     *            cut off value
     * @param alg
     *            algorithm used in model
     * @return data pair instance
     * @throws NullPointerException
     *             if input is null
     * @throws NumberFormatException
     *             if column value is not number format.
     */
    public static MLDataPair assembleDataPair(Map<Integer, Map<String, Integer>> binCategoryMap, boolean noVarSel,
            ModelConfig modelConfig, List<ColumnConfig> columnConfigList, Map<String, ? extends Object> rawDataMap,
            double cutoff, String alg) {
        return assembleNsDataPair(binCategoryMap, noVarSel, modelConfig, columnConfigList,
                convertRawObjectMapToNsDataMap(rawDataMap), cutoff, alg);
    }

    /**
     * Assemble map data to Encog standard input format. If no variable selected(noVarSel = true), all candidate
     * variables will be selected.
     * 
     * @param binCategoryMap
     *            categorical map
     * @param noVarSel
     *            if after var select
     * @param modelConfig
     *            model config instance
     * @param columnConfigList
     *            column config list
     * @param rawNsDataMap
     *            raw NSColumn data
     * @param cutoff
     *            cut off value
     * @param alg
     *            algorithm used in model
     * @return data pair instance
     * @throws NullPointerException
     *             if input is null
     * @throws NumberFormatException
     *             if column value is not number format.
     */
    public static MLDataPair assembleNsDataPair(Map<Integer, Map<String, Integer>> binCategoryMap, boolean noVarSel,
            ModelConfig modelConfig, List<ColumnConfig> columnConfigList, Map<NSColumn, String> rawNsDataMap,
            double cutoff, String alg) {
        double[] ideal = { Constants.DEFAULT_IDEAL_VALUE };

        List<Double> inputList = new ArrayList<Double>();
        boolean hasCandidates = CommonUtils.hasCandidateColumns(columnConfigList);
        for(ColumnConfig config: columnConfigList) {
            if(config == null) {
                continue;
            }
            NSColumn key = new NSColumn(config.getColumnName());
            if(config.isFinalSelect() // check whole name
                    && !rawNsDataMap.containsKey(key) // and then check simple name, in case user use wrong namespace
                    && !rawNsDataMap.containsKey(new NSColumn(key.getSimpleName()))) {
                throw new IllegalStateException(String.format("Variable Missing in Test Data: %s", key));
            }

            if(config.isTarget()) {
                continue;
            } else {
                if(!noVarSel) {
                    if(config != null && !config.isMeta() && !config.isTarget() && config.isFinalSelect()) {
                        String val = getNSVariableVal(rawNsDataMap, key);
                        if(CommonUtils.isTreeModel(alg) && config.isCategorical()) {
                            Integer index = binCategoryMap.get(config.getColumnNum()).get(val == null ? "" : val);
                            if(index == null) {
                                // not in binCategories, should be missing value
                                // -1 as missing value
                                inputList.add(-1d);
                            } else {
                                inputList.add(index * 1d);
                            }
                        } else {
                            inputList.addAll(computeNumericNormResult(modelConfig, cutoff, config, val));
                        }
                    }
                } else {
                    if(!config.isMeta() && !config.isTarget() && CommonUtils.isGoodCandidate(config, hasCandidates)) {
                        String val = getNSVariableVal(rawNsDataMap, key);
                        if(CommonUtils.isTreeModel(alg) && config.isCategorical()) {
                            Integer index = binCategoryMap.get(config.getColumnNum()).get(val == null ? "" : val);
                            if(index == null) {
                                // not in binCategories, should be missing value
                                // -1 as missing value
                                inputList.add(-1d);
                            } else {
                                inputList.add(index * 1d);
                            }
                        } else {
                            inputList.addAll(computeNumericNormResult(modelConfig, cutoff, config, val));
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

    public static String getNSVariableVal(Map<NSColumn, String> rawNsDataMap, NSColumn key) {
        String val = rawNsDataMap.get(key);
        return (val == null ? rawNsDataMap.get(new NSColumn(key.getSimpleName())) : val);
    }

    /**
     * Simple name without name space part. For segment expansion, only retain raw column name but not current column
     * name.
     * 
     * @param columnConfig
     *            the column configuration
     * @param columnConfigList
     *            the column config list inculding all segment expansion columns if have
     * @param segmentExpansions
     *            segment expansion expressions
     * @param dataSetHeaders
     *            data set headers for all raw columns
     * @return the simple name not including name space part
     */
    public static String getSimpleColumnName(ColumnConfig columnConfig, List<ColumnConfig> columnConfigList,
            List<String> segmentExpansions, String[] dataSetHeaders) {
        if(segmentExpansions == null || segmentExpansions.size() == 0) {
            return getSimpleColumnName(columnConfig.getColumnName());
        }

        // if(columnConfigList.size() != dataSetHeaders.size() * (segmentExpansions.size() + 1)) {
        // throw new IllegalStateException(
        // "Segment expansion enabled but # of columns in ColumnConfig.json is not consistent with segment expansion files.");
        // }

        if(columnConfig.getColumnNum() >= dataSetHeaders.length) {
            return getSimpleColumnName(columnConfigList.get(columnConfig.getColumnNum() % dataSetHeaders.length)
                    .getColumnName());
        } else {
            return getSimpleColumnName(columnConfig.getColumnName());
        }
    }

    public static String getSimpleColumnName(String columnName) {
        // remove name-space in column name to make it be called by simple name
        if(columnName.contains(CommonConstants.NAMESPACE_DELIMITER)) {
            columnName = columnName.substring(columnName.lastIndexOf(CommonConstants.NAMESPACE_DELIMITER)
                    + CommonConstants.NAMESPACE_DELIMITER.length(), columnName.length());
        }
        return columnName;
    }

    /**
     * Assemble map data to Encog standard input format. If no variable selected(noVarSel = true), all candidate
     * variables will be selected.
     * 
     * @param binCategoryMap
     *            categorical map
     * @param noVarSel
     *            if after var select
     * @param modelConfig
     *            model config instance
     * @param columnConfigList
     *            column config list
     * @param rawNsDataMap
     *            raw NSColumn data
     * @param cutoff
     *            cut off value
     * @param alg
     *            algorithm used in model
     * @param featureSet
     *            feature set used in NN model
     * @return data pair instance
     * @throws NullPointerException
     *             if input is null
     * @throws NumberFormatException
     *             if column value is not number format.
     */
    public static MLDataPair assembleNsDataPair(Map<Integer, Map<String, Integer>> binCategoryMap, boolean noVarSel,
            ModelConfig modelConfig, List<ColumnConfig> columnConfigList, Map<NSColumn, String> rawNsDataMap,
            double cutoff, String alg, Set<Integer> featureSet) {
        if(CollectionUtils.isEmpty(featureSet)) {
            return assembleNsDataPair(binCategoryMap, noVarSel, modelConfig, columnConfigList, rawNsDataMap, cutoff,
                    alg);
        }
        double[] ideal = { Constants.DEFAULT_IDEAL_VALUE };

        List<Double> inputList = new ArrayList<Double>();
        for(ColumnConfig config: columnConfigList) {
            if(config == null) {
                continue;
            }
            NSColumn key = new NSColumn(config.getColumnName());
            if(config.isFinalSelect() // check whole name
                    && !rawNsDataMap.containsKey(key) // and then check simple name, in case user use wrong namespace
                    && !rawNsDataMap.containsKey(new NSColumn(key.getSimpleName()))) {
                throw new IllegalStateException(String.format("Variable Missing in Test Data: %s", key));
            }

            if(config.isTarget()) {
                continue;
            } else {
                if(featureSet.contains(config.getColumnNum())) {
                    String val = getNSVariableVal(rawNsDataMap, key);
                    if(CommonUtils.isTreeModel(alg) && config.isCategorical()) {
                        Integer index = binCategoryMap.get(config.getColumnNum()).get(val == null ? "" : val);
                        if(index == null) {
                            // not in binCategories, should be missing value -1 as missing value
                            inputList.add(-1d);
                        } else {
                            inputList.add(index * 1d);
                        }
                    } else {
                        inputList.addAll(computeNumericNormResult(modelConfig, cutoff, config, val));
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

    /**
     * Get all available feature ids from ColumnConfig list.
     * There are two situations for this: 1) when training model, get all available features before start
     * 2) get all available features before doing variable selection
     * 
     * @param columnConfigList
     *            - ColumnConfig list to check
     * @param isAfterVarSelect
     *            - true for training, false for variable selection
     * @return - available feature list
     */
    public static List<Integer> getAllFeatureList(List<ColumnConfig> columnConfigList, boolean isAfterVarSelect) {
        boolean hasCandidate = hasCandidateColumns(columnConfigList);

        List<Integer> features = new ArrayList<Integer>();
        for(ColumnConfig config: columnConfigList) {
            if(isAfterVarSelect) {
                if(config.isFinalSelect() && !config.isTarget() && !config.isMeta()) {
                    // only select numerical feature with getBinBoundary().size() larger than 1
                    // or categorical feature with getBinCategory().size() larger than 0
                    if((config.isNumerical() && config.getBinBoundary() != null && config.getBinBoundary().size() > 1)
                            || (config.isCategorical() && config.getBinCategory() != null && config.getBinCategory().size() > 0)) {
                        features.add(config.getColumnNum());
                    }
                }
            } else {
                if(!config.isMeta() && !config.isTarget() && CommonUtils.isGoodCandidate(config, hasCandidate)) {
                    // only select numerical feature with getBinBoundary().size() larger than 1
                    // or categorical feature with getBinCategory().size() larger than 0
                    if((config.isNumerical() && config.getBinBoundary() != null && config.getBinBoundary().size() > 1)
                            || (config.isCategorical() && config.getBinCategory() != null && config.getBinCategory().size() > 0)) {
                        features.add(config.getColumnNum());
                    }
                }
            }
        }
        return features;
    }

    /**
     * Check whether candidates are set or not
     * 
     * @param columnConfigList
     *            - ColumnConfig list to check
     * @return
     *         - true if use set candidate columns, or false
     */
    public static boolean hasCandidateColumns(List<ColumnConfig> columnConfigList) {
        int candidateCnt = 0;
        for(ColumnConfig config: columnConfigList) {
            if(ColumnConfig.ColumnFlag.Candidate.equals(config.getColumnFlag())) {
                candidateCnt++;
            }
        }

        return (candidateCnt > 0);
    }

    private static List<Double> computeNumericNormResult(ModelConfig modelConfig, double cutoff, ColumnConfig config,
            String val) {
        List<Double> normalizeValue = null;
        if(CommonUtils.isTreeModel(modelConfig.getAlgorithm())) {
            try {
                normalizeValue = Arrays.asList(new Double[] { Double.parseDouble(val) });
            } catch (Exception e) {
                normalizeValue = Arrays.asList(new Double[] { Normalizer.defaultMissingValue(config) });
            }
        } else {
            normalizeValue = Normalizer.normalize(config, val, cutoff, modelConfig.getNormalizeType());
        }
        return normalizeValue;
    }

    public static boolean isTreeModel(String alg) {
        return CommonConstants.RF_ALG_NAME.equalsIgnoreCase(alg) || CommonConstants.GBT_ALG_NAME.equalsIgnoreCase(alg);
    }

    public static boolean isNNModel(String alg) {
        return "nn".equalsIgnoreCase(alg);
    }

    public static boolean isLRModel(String alg) {
        return "lr".equalsIgnoreCase(alg);
    }

    public static boolean isRandomForestAlgorithm(String alg) {
        return CommonConstants.RF_ALG_NAME.equalsIgnoreCase(alg);
    }

    public static boolean isGBDTAlgorithm(String alg) {
        return CommonConstants.GBT_ALG_NAME.equalsIgnoreCase(alg);
    }

    public static boolean isHadoopConfigurationInjected(String key) {
        return key.startsWith("nn") || key.startsWith("guagua") || key.startsWith("shifu") || key.startsWith("mapred")
                || key.startsWith("io") || key.startsWith("hadoop") || key.startsWith("yarn") || key.startsWith("pig")
                || key.startsWith("hive") || key.startsWith("job");
    }

    /**
     * Assemble map data to Encog standard input format.
     * 
     * @param modelConfig
     *            - ModelConfig
     * @param columnConfigList
     *            - ColumnConfig list
     * @param rawDataMap
     *            - raw input key-value map
     * @param cutoff
     *            - cutoff value when normalization
     * @return
     *         - input data pair for neural network
     */
    public static MLDataPair assembleDataPair(ModelConfig modelConfig, List<ColumnConfig> columnConfigList,
            Map<String, ? extends Object> rawDataMap, double cutoff) {
        Map<NSColumn, Object> nsDataMap = new HashMap<NSColumn, Object>();
        for(Entry<String, ? extends Object> entry: rawDataMap.entrySet()) {
            nsDataMap.put(new NSColumn(entry.getKey()), entry.getValue());
        }

        // if the tag is provided, ideal will be updated; otherwise it defaults to -1
        double[] ideal = { Constants.DEFAULT_IDEAL_VALUE };

        List<Double> inputList = new ArrayList<Double>();
        for(ColumnConfig config: columnConfigList) {
            NSColumn key = new NSColumn(config.getColumnName());
            if(config.isFinalSelect() && !nsDataMap.containsKey(key)) {
                throw new IllegalStateException(String.format("Variable Missing in Test Data: %s", key));
            }

            if(config.isTarget()) {
                // TODO - should we have this? maybe not
                // ideal[0] = Double.valueOf(rawDataMap.get(key).toString());
                continue;
            } else if(config.isFinalSelect()) {
                // add log for debug purpose
                // log.info("key: " + key + ", raw_value " + rawDataMap.get(key).toString() + ", zscl_value: " +
                String val = nsDataMap.get(key) == null ? null : nsDataMap.get(key).toString();
                List<Double> normVals = Normalizer.normalize(config, val, cutoff, modelConfig.getNormalizeType());
                inputList.addAll(normVals);
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

    /*
     * Expanding score by expandingFactor
     */
    public static long getExpandingScore(double d, int expandingFactor) {
        return Math.round(d * expandingFactor);
    }

    /**
     * Return column name string with 'derived_' started
     * 
     * @param columnConfigList
     *            list of column config
     * @return list of column names
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
     * @param modelConfig
     *            model config list
     * @param columnConfigList
     *            the column config list
     * @throws IOException
     *             any io exception
     * 
     * @throws IllegalArgumentException
     *             if modelConfig is null or columnConfigList is null.
     */
    public static void updateColumnConfigFlags(ModelConfig modelConfig, List<ColumnConfig> columnConfigList)
            throws IOException {
        String targetColumnName = modelConfig.getTargetColumnName();
        String weightColumnName = modelConfig.getWeightColumnName();

        Set<NSColumn> setCategorialColumns = new HashSet<NSColumn>();
        List<String> categoricalColumnNames = modelConfig.getCategoricalColumnNames();
        if(CollectionUtils.isNotEmpty(categoricalColumnNames)) {
            for(String column: categoricalColumnNames) {
                setCategorialColumns.add(new NSColumn(column));
            }
        }

        Set<NSColumn> setHybridColumns = new HashSet<NSColumn>();
        Map<String, Double> hybridColumnNames = modelConfig.getHybridColumnNames();
        if(hybridColumnNames != null && hybridColumnNames.size() > 0) {
            for(Entry<String, Double> entry: hybridColumnNames.entrySet()) {
                setHybridColumns.add(new NSColumn(entry.getKey()));
            }
        }

        Set<NSColumn> setMeta = new HashSet<NSColumn>();
        if(CollectionUtils.isNotEmpty(modelConfig.getMetaColumnNames())) {
            for(String meta: modelConfig.getMetaColumnNames()) {
                setMeta.add(new NSColumn(meta));
            }
        }

        Set<NSColumn> setForceRemove = new HashSet<NSColumn>();
        if(Boolean.TRUE.equals(modelConfig.getVarSelect().getForceEnable())
                && CollectionUtils.isNotEmpty(modelConfig.getListForceRemove())) {
            // if we need to update force remove, only and if one the force is enabled
            for(String forceRemoveName: modelConfig.getListForceRemove()) {
                setForceRemove.add(new NSColumn(forceRemoveName));
            }
        }

        Set<NSColumn> setForceSelect = new HashSet<NSColumn>(512);
        if(Boolean.TRUE.equals(modelConfig.getVarSelect().getForceEnable())
                && CollectionUtils.isNotEmpty(modelConfig.getListForceSelect())) {
            // if we need to update force select, only and if one the force is enabled
            for(String forceSelectName: modelConfig.getListForceSelect()) {
                setForceSelect.add(new NSColumn(forceSelectName));
            }
        }

        for(ColumnConfig config: columnConfigList) {
            String varName = config.getColumnName();

            // reset it
            config.setColumnFlag(null);

            if(NSColumnUtils.isColumnEqual(weightColumnName, varName)) {
                config.setColumnFlag(ColumnFlag.Weight);
                config.setFinalSelect(false); // reset final select
            } else if(NSColumnUtils.isColumnEqual(targetColumnName, varName)) {
                config.setColumnFlag(ColumnFlag.Target);
                config.setFinalSelect(false); // reset final select
            } else if(setMeta.contains(new NSColumn(varName))) {
                config.setColumnFlag(ColumnFlag.Meta);
                config.setFinalSelect(false); // reset final select
            } else if(setForceRemove.contains(new NSColumn(varName))) {
                config.setColumnFlag(ColumnFlag.ForceRemove);
                config.setFinalSelect(false); // reset final select
            } else if(setForceSelect.contains(new NSColumn(varName))) {
                config.setColumnFlag(ColumnFlag.ForceSelect);
            }

            if(NSColumnUtils.isColumnEqual(weightColumnName, varName)) {
                // weight column is numerical
                config.setColumnType(ColumnType.N);
            } else if(NSColumnUtils.isColumnEqual(targetColumnName, varName)) {
                // target column is set to categorical column
                config.setColumnType(ColumnType.C);
            } else if(setHybridColumns.contains(new NSColumn(varName))) {
                config.setColumnType(ColumnType.H);
                String newVarName = null;
                if(Environment.getBoolean(Constants.SHIFU_NAMESPACE_STRICT_MODE, false)) {
                    newVarName = new NSColumn(varName).getFullColumnName();
                } else {
                    newVarName = new NSColumn(varName).getSimpleName();
                }
                config.setHybridThreshold(hybridColumnNames.get(newVarName));
            } else if(setCategorialColumns.contains(new NSColumn(varName))) {
                config.setColumnType(ColumnType.C);
            } else {
                config.setColumnType(ColumnType.N);
            }
        }
    }

    public static boolean isNumber(String valStr) {
        if(StringUtils.isBlank(valStr)) {
            return false;
        }
        try {
            Double.parseDouble(valStr);
            return true;
        } catch (NumberFormatException e) {
            return false;
        }
    }

    /**
     * Avoid parsing times, failed parsing is set to NaN
     * 
     * @param valStr
     *            param string
     * @return double after parsing
     */
    public static double parseNumber(String valStr) {
        if(StringUtils.isBlank(valStr)) {
            return Double.NaN;
        }
        try {
            return Double.parseDouble(valStr);
        } catch (NumberFormatException e) {
            return Double.NaN;
        }
    }

    /**
     * To check whether there is targetColumn in columns or not
     * 
     * @param columns
     *            column array
     * @param targetColumn
     *            target column
     * 
     * @return true - if the columns contains targetColumn, or false
     */
    public static boolean isColumnExists(String[] columns, String targetColumn) {
        if(ArrayUtils.isEmpty(columns) || StringUtils.isBlank(targetColumn)) {
            return false;
        }

        for(int i = 0; i < columns.length; i++) {
            if(columns[i] != null && columns[i].equalsIgnoreCase(targetColumn)) {
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
     * 
     * @param rightCol
     *            - right collection
     * @param <T>
     *            - collection type
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
                    // String column = CommonUtils.getRelativePigHeaderColumnName(str);
                    if(StringUtils.isNotBlank(str)) {
                        columnNameList.add(str.trim());
                    }
                }
            }
        }

        return columnNameList;
    }

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
     *            list of column config
     * @param columnName
     *            the column name
     * @return column config instance
     */
    public static ColumnConfig findColumnConfigByName(List<ColumnConfig> columnConfigList, String columnName) {
        for(ColumnConfig columnConfig: columnConfigList) {
            if(NSColumnUtils.isColumnEqual(columnConfig.getColumnName(), columnName)) {
                return columnConfig;
            }
        }
        return null;
    }

    /**
     * Return target column configuration itme.
     * 
     * @param columnConfigList
     *            the column config list
     * @return target column configuration.
     */
    public static ColumnConfig findTargetColumn(List<ColumnConfig> columnConfigList) {
        for(ColumnConfig columnConfig: columnConfigList) {
            if(columnConfig.isTarget()) {
                return columnConfig;
            }
        }
        return null;
    }

    /**
     * Convert data into (key, value) map. The inputData is String of a record, which is delimited by delimiter
     * If fields in inputData is not equal header size, return null
     * 
     * @param inputData
     *            - String of a record
     * @param delimiter
     *            - the delimiter of the input data
     * @param header
     *            - the column names for all the input data
     * @return (key, value) map for the record
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
     * Convert tuple record into (key, value) map. The @tuple is Tuple for a record
     * If @tuple size is not equal @header size, return null
     * 
     * @param tuple
     *            - Tuple of a record
     * @param header
     *            - the column names for all the input data
     * @return (key, value) map for the record
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

    /**
     * Convert tuple record into (NSColumn, value) map. The @tuple is Tuple for a record
     * If @tuple size is not equal @header size, return null
     * 
     * @param tuple
     *            - Tuple of a record
     * @param header
     *            - the column names for all the input data
     * @param segFilterSize
     *            segment filter size
     * @return (NSColumn, value) map for the record
     * @throws ExecException
     *             - throw exception when operating tuple
     */

    public static Map<NSColumn, String> convertDataIntoNsMap(Tuple tuple, String[] header, int segFilterSize)
            throws ExecException {
        if(tuple == null || tuple.size() == 0 || tuple.size() != header.length) {
            log.error("Invalid input, the tuple.size is = " + (tuple == null ? null : tuple.size())
                    + ", header.length = " + header.length);
            return null;
        }

        Map<NSColumn, String> rawDataNsMap = new HashMap<NSColumn, String>(tuple.size());
        for(int i = 0; i < header.length; i++) {
            if(tuple.get(i) == null) {
                rawDataNsMap.put(new NSColumn(header[i]), "");
            } else {
                rawDataNsMap.put(new NSColumn(header[i]), tuple.get(i).toString());
            }
        }

        for(int i = 0; i < segFilterSize; i++) {
            for(int j = 0; j < header.length; j++) {
                if(tuple.get(j) == null) {
                    rawDataNsMap.put(new NSColumn(header[j] + "_" + (i + 1)), "");
                } else {
                    rawDataNsMap.put(new NSColumn(header[j] + "_" + (i + 1)), tuple.get(j).toString());
                }
            }
        }

        return rawDataNsMap;
    }

    public static boolean isGoodCandidate(ColumnConfig columnConfig, boolean hasCandidate,
            boolean isBinaryClassification) {
        if(columnConfig == null) {
            return false;
        }

        if(isBinaryClassification) {
            return isGoodCandidate(columnConfig, hasCandidate);
        } else {
            // multiple classification
            return columnConfig.isCandidate(hasCandidate)
                    && (columnConfig.getMean() != null && columnConfig.getStdDev() != null && ((columnConfig
                            .isCategorical() && columnConfig.getBinCategory() != null && columnConfig.getBinCategory()
                            .size() > 1) || (columnConfig.isNumerical() && columnConfig.getBinBoundary() != null && columnConfig
                            .getBinBoundary().size() > 1)));
        }
    }

    /*
     * public static boolean isGoodCandidate(ColumnConfig columnConfig) {
     * if(columnConfig == null) {
     * return false;
     * }
     * return columnConfig.isCandidate()
     * && (columnConfig.getKs() != null && columnConfig.getKs() > 0 && columnConfig.getIv() != null
     * && columnConfig.getIv() > 0 && columnConfig.getMean() != null
     * && columnConfig.getStdDev() != null && ((columnConfig.isCategorical()
     * && columnConfig.getBinCategory() != null && columnConfig.getBinCategory().size() > 1) || (columnConfig
     * .isNumerical() && columnConfig.getBinBoundary() != null && columnConfig.getBinBoundary().size() > 1)));
     * }
     */

    public static boolean isGoodCandidate(ColumnConfig columnConfig, boolean hasCandidate) {
        if(columnConfig == null) {
            return false;
        }

        return columnConfig.isCandidate(hasCandidate)
                && (columnConfig.getKs() != null && columnConfig.getKs() > 0 && columnConfig.getIv() != null
                        && columnConfig.getIv() > 0 && columnConfig.getMean() != null
                        && columnConfig.getStdDev() != null && ((columnConfig.isCategorical()
                        && columnConfig.getBinCategory() != null && columnConfig.getBinCategory().size() > 1) || (columnConfig
                        .isNumerical() && columnConfig.getBinBoundary() != null && columnConfig.getBinBoundary().size() > 1)));
    }

    /**
     * Return first line split string array. This is used to detect data schema.
     * 
     * @param dataSetRawPath
     *            raw data path
     * @param delimeter
     *            the delimiter
     * @param source
     *            source type
     * @return the first two lines
     * @throws IOException
     *             any io exception
     */
    public static String[] takeFirstLine(String dataSetRawPath, String delimeter, SourceType source) throws IOException {
        if(dataSetRawPath == null || delimeter == null || source == null) {
            throw new IllegalArgumentException("Input parameters should not be null.");
        }

        String firstValidFile = null;
        FileSystem fs = ShifuFileUtils.getFileSystemBySourceType(source);
        FileStatus[] globStatus = fs.globStatus(new Path(dataSetRawPath), HIDDEN_FILE_FILTER);
        if(globStatus == null || globStatus.length == 0) {
            throw new IllegalArgumentException("No files founded in " + dataSetRawPath);
        } else {
            for(FileStatus fileStatus: globStatus) {
                RemoteIterator<LocatedFileStatus> iterator = fs.listFiles(fileStatus.getPath(), true);
                while(iterator.hasNext()) {
                    LocatedFileStatus lfs = iterator.next();
                    String name = lfs.getPath().getName();
                    if(name.startsWith("_") || name.startsWith(".")) {
                        // hidden files,
                        continue;
                    }
                    // 20L is min gzip file size
                    if(lfs.getLen() > 20L) {
                        firstValidFile = lfs.getPath().toString();
                        break;
                    }
                }
                if(StringUtils.isNotBlank(firstValidFile)) {
                    break;
                }
            }
        }
        log.info("The first valid file is - {}", firstValidFile);

        BufferedReader reader = null;
        try {
            reader = ShifuFileUtils.getReader(firstValidFile, source);
            String firstLine = reader.readLine();
            if(firstLine != null && firstLine.length() > 0) {
                List<String> list = new ArrayList<String>();
                for(String unit: Splitter.on(delimeter).split(firstLine)) {
                    list.add(unit);
                }
                return list.toArray(new String[0]);
            }
        } catch (Exception e) {
            log.error("Fail to read first line of file.", e);
        } finally {
            IOUtils.closeQuietly(reader);
        }
        return new String[0];
    }

    /**
     * Return first two lines split string array. This is used to detect data schema and check if data
     * schema is the
     * same as data.
     * 
     * @param dataSetRawPath
     *            raw data path
     * @param delimiter
     *            the delimiter
     * @param source
     *            source type
     * @return the first two lines
     * @throws IOException
     *             any io exception
     */
    public static String[][] takeFirstTwoLines(String dataSetRawPath, String delimiter, SourceType source)
            throws IOException {
        if(dataSetRawPath == null || delimiter == null || source == null) {
            throw new IllegalArgumentException("Input parameters should not be null.");
        }

        String firstValidFile = null;
        FileSystem fs = ShifuFileUtils.getFileSystemBySourceType(source);
        FileStatus[] globStatus = fs.globStatus(new Path(dataSetRawPath), HIDDEN_FILE_FILTER);
        if(globStatus == null || globStatus.length == 0) {
            throw new IllegalArgumentException("No files founded in " + dataSetRawPath);
        } else {
            for(FileStatus fileStatus: globStatus) {
                RemoteIterator<LocatedFileStatus> iterator = fs.listFiles(fileStatus.getPath(), true);
                while(iterator.hasNext()) {
                    LocatedFileStatus lfs = iterator.next();
                    String name = lfs.getPath().getName();
                    if(name.startsWith("_") || name.startsWith(".")) {
                        // hidden files,
                        continue;
                    }
                    if(lfs.getLen() > 1024L) {
                        firstValidFile = lfs.getPath().toString();
                        break;
                    }
                }
                if(StringUtils.isNotBlank(firstValidFile)) {
                    break;
                }
            }
        }
        log.info("The first valid file is - {}", firstValidFile);

        BufferedReader reader = null;
        try {
            reader = ShifuFileUtils.getReader(firstValidFile, source);

            String firstLine = reader.readLine();
            String[] firstArray = null;
            if(firstLine != null && firstLine.length() > 0) {
                List<String> list = new ArrayList<String>();
                for(String unit: Splitter.on(delimiter).split(firstLine)) {
                    list.add(unit);
                }
                firstArray = list.toArray(new String[0]);
            }

            String secondLine = reader.readLine();
            String[] secondArray = null;
            if(secondLine != null && secondLine.length() > 0) {
                List<String> list = new ArrayList<String>();
                for(String unit: Splitter.on(delimiter).split(secondLine)) {
                    list.add(unit);
                }
                secondArray = list.toArray(new String[0]);
            }
            String[][] results = new String[2][];
            results[0] = firstArray;
            results[1] = secondArray;
            return results;
        } finally {
            IOUtils.closeQuietly(reader);
        }
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
            pigScoreNames[i] = genPigFieldName(names[i]) + "::mean";
        }
        return pigScoreNames;
    }

    /**
     * Compute feature importance for all bagging tree models.
     * 
     * @param models
     *            the tree models, should be instance of TreeModel
     * @return feature importance per each column id
     * @throws IllegalStateException
     *             if no any feature importance from models
     */
    public static Map<Integer, MutablePair<String, Double>> computeTreeModelFeatureImportance(List<BasicML> models) {
        List<Map<Integer, MutablePair<String, Double>>> importanceList = new ArrayList<Map<Integer, MutablePair<String, Double>>>();
        for(BasicML basicModel: models) {
            if(basicModel instanceof TreeModel) {
                TreeModel model = (TreeModel) basicModel;
                Map<Integer, MutablePair<String, Double>> importances = model.getFeatureImportances();
                importanceList.add(importances);
            }
        }
        if(importanceList.size() < 1) {
            throw new IllegalStateException("Feature importance calculation abort due to no tree model found!!");
        }
        return mergeImportanceList(importanceList);
    }

    private static Map<Integer, MutablePair<String, Double>> mergeImportanceList(
            List<Map<Integer, MutablePair<String, Double>>> list) {
        Map<Integer, MutablePair<String, Double>> finalResult = new HashMap<Integer, MutablePair<String, Double>>();
        int modelSize = list.size();
        for(Map<Integer, MutablePair<String, Double>> item: list) {
            for(Entry<Integer, MutablePair<String, Double>> entry: item.entrySet()) {
                if(!finalResult.containsKey(entry.getKey())) {
                    // do average on models by dividing modelSize
                    MutablePair<String, Double> value = MutablePair.of(entry.getValue().getKey(), entry.getValue()
                            .getValue() / modelSize);
                    finalResult.put(entry.getKey(), value);
                } else {
                    MutablePair<String, Double> current = finalResult.get(entry.getKey());
                    double entryValue = entry.getValue().getValue();
                    current.setValue(current.getValue() + (entryValue / modelSize));
                    finalResult.put(entry.getKey(), current);
                }
            }
        }
        return TreeModel.sortByValue(finalResult, false);
    }

    public static void writeFeatureImportance(String fiPath, Map<Integer, MutablePair<String, Double>> importances)
            throws IOException {
        ShifuFileUtils.createFileIfNotExists(fiPath, SourceType.LOCAL);
        BufferedWriter writer = null;
        log.info("Writing feature importances to file {}", fiPath);
        try {
            writer = ShifuFileUtils.getWriter(fiPath, SourceType.LOCAL);
            writer.write("column_id\t\tcolumn_name\t\timportance");
            writer.newLine();
            for(Map.Entry<Integer, MutablePair<String, Double>> entry: importances.entrySet()) {
                String content = entry.getKey() + "\t\t" + entry.getValue().getKey() + "\t\t"
                        + entry.getValue().getValue();
                writer.write(content);
                writer.newLine();
            }
            writer.flush();
        } finally {
            IOUtils.closeQuietly(writer);
        }
    }

    public static String trimTag(String tag) {
        if(NumberUtils.isNumber(tag)) {
            tag = tag.trim();
            int firstPeriodPos = -1;
            int firstDeleteZero = -1;
            boolean hasMetNonZero = false;
            for(int i = tag.length(); i > 0; i--) {
                if((tag.charAt(i - 1) == '0' || tag.charAt(i - 1) == '.') && !hasMetNonZero) {
                    firstDeleteZero = i - 1;
                }

                if(tag.charAt(i - 1) != '0') {
                    hasMetNonZero = true;
                }

                if(tag.charAt(i - 1) == '.') {
                    firstPeriodPos = i - 1;
                }
            }

            String result = (firstDeleteZero >= 0 && firstPeriodPos >= 0) ? tag.substring(0, firstDeleteZero) : tag;
            return (firstPeriodPos == 0) ? "0" + result : result;
        } else {
            return StringUtils.trimToEmpty(tag);
        }
    }

    /**
     * Convert (String, String) raw data map to (NSColumn, String) data map
     * 
     * @param rawDataMap
     *            - (String, String) raw data map
     * @return (NSColumn, String) data map
     */
    public static Map<NSColumn, String> convertRawMapToNsDataMap(Map<String, String> rawDataMap) {
        if(rawDataMap == null) {
            return null;
        }

        Map<NSColumn, String> nsDataMap = new HashMap<NSColumn, String>();
        for(String key: rawDataMap.keySet()) {
            nsDataMap.put(new NSColumn(key), rawDataMap.get(key));
        }
        return nsDataMap;
    }

    /**
     * Convert (String, ? extends Object) raw data map to (NSColumn, String) data map
     * 
     * @param rawDataMap
     *            - (String, ? extends Object) raw data map
     * @return (NSColumn, String) data map
     */
    public static Map<NSColumn, String> convertRawObjectMapToNsDataMap(Map<String, ? extends Object> rawDataMap) {
        if(rawDataMap == null) {
            return null;
        }

        Map<NSColumn, String> nsDataMap = new HashMap<NSColumn, String>();
        for(String key: rawDataMap.keySet()) {
            Object value = rawDataMap.get(key);
            nsDataMap.put(new NSColumn(key), ((value == null) ? null : value.toString()));
        }

        return nsDataMap;
    }

    /**
     * flatten categorical value group into values list
     * 
     * @param categoricalValGrp
     *            - categorical val group, it some values like zn^us^ck^
     * @return value list of categorical val
     */
    public static List<String> flattenCatValGrp(String categoricalValGrp) {
        List<String> catVals = new ArrayList<String>();
        if(StringUtils.isNotBlank(categoricalValGrp)) {
            for(String cval: Splitter.on(Constants.CATEGORICAL_GROUP_VAL_DELIMITER).split(categoricalValGrp)) {
                catVals.add(cval);
            }
        }
        return catVals;
    }

    /**
     * Manual split function to avoid depending on guava.
     * 
     * <p>
     * Some examples: "^"=&gt;[, ]; ""=&gt;[]; "a"=&gt;[a]; "abc"=&gt;[abc]; "a^"=&gt;[a, ]; "^b"=&gt;[, b];
     * "^^b"=&gt;[, , b]
     * 
     * @param str
     *            the string to be split
     * @param delimiter
     *            the delimiter
     * @return split string array
     */
    public static String[] splitString(String str, String delimiter) {
        if(str == null || str.length() == 0) {
            return new String[] { "" };
        }

        List<String> categories = new ArrayList<String>();
        int dLen = delimiter.length();
        int begin = 0;
        for(int i = 0; i < str.length(); i++) {
            if(str.substring(i, Math.min(i + dLen, str.length())).equals(delimiter)) {
                categories.add(str.substring(begin, i));
                begin = i + dLen;
            }
            if(i == str.length() - 1) {
                categories.add(str.substring(begin, str.length()));
            }
        }

        return categories.toArray(new String[0]);
    }

}