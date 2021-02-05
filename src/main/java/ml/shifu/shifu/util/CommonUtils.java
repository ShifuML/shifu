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

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
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

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.collections.Predicate;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.commons.lang.math.NumberUtils;
import org.apache.commons.lang3.tuple.MutablePair;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.RemoteIterator;
import org.apache.pig.backend.executionengine.ExecException;
import org.apache.pig.data.Tuple;
import org.encog.mathutil.BoundMath;
import org.encog.ml.BasicML;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Function;
import com.google.common.base.Splitter;
import com.google.common.collect.Collections2;
import com.google.common.collect.Lists;

import ml.shifu.guagua.util.NumberFormatUtils;
import ml.shifu.shifu.column.NSColumn;
import ml.shifu.shifu.column.NSColumnUtils;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.TreeModel;
import ml.shifu.shifu.core.dtrain.CommonConstants;
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
        FileSystem localFs = HDFSUtils.getLocalFS();

        Path hdfsMSPath = new Path(pathFinder.getModelSetPath(SourceType.HDFS));
        FileSystem hdfs = HDFSUtils.getFS(hdfsMSPath);
        Path pathModelSet = hdfsMSPath;
        // don't check whether pathModelSet is exists, should be remove by user.
        hdfs.mkdirs(pathModelSet);

        // Copy ModelConfig
        Path srcModelConfig = new Path(pathFinder.getModelConfigPath(SourceType.LOCAL));
        hdfs.copyFromLocalFile(srcModelConfig, hdfsMSPath);

        // Copy GridSearch config file if exists
        String gridConfigFile = modelConfig.getTrain().getGridConfigFile();
        if(gridConfigFile != null && !gridConfigFile.trim().equals("")) {
            Path srcGridConfig = new Path(modelConfig.getTrain().getGridConfigFile());
            hdfs.copyFromLocalFile(srcGridConfig, hdfsMSPath);
        }

        // Copy ColumnConfig
        if(modelConfig.isMultiTask()) {
            copyMTLColumnConfigs(modelConfig, hdfs, hdfsMSPath);
        } else {
            Path srcColumnConfig = new Path(pathFinder.getColumnConfigPath(SourceType.LOCAL));
            Path dstColumnConfig = new Path(pathFinder.getColumnConfigPath(SourceType.HDFS));
            if(ShifuFileUtils.isFileExists(srcColumnConfig.toString(), SourceType.LOCAL)) {
                hdfs.copyFromLocalFile(srcColumnConfig, dstColumnConfig);
            }
        }

        // Copy column related config files
        copyColumnConfigFiles(modelConfig, hdfs, hdfsMSPath);

        // Copy others
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

    private static void copyMTLColumnConfigs(ModelConfig modelConfig, FileSystem hdfs, Path hdfsMSPath)
            throws IOException {
        PathFinder pf = new PathFinder(modelConfig);
        Path mtlHDFSPath = new Path(pf.getMTLColumnConfigFolder(SourceType.HDFS));
        hdfs.mkdirs(mtlHDFSPath);
        List<String> tagColumnNames = modelConfig.getMultiTaskTargetColumnNames();
        for(int i = 0; i < tagColumnNames.size(); i++) {
            Path srcPath = new Path(pf.getMTLColumnConfigPath(SourceType.LOCAL, i));
            hdfs.copyFromLocalFile(srcPath, mtlHDFSPath);
        }
    }

    private static void copyColumnConfigFiles(ModelConfig modelConfig, FileSystem hdfs, Path hdfsMSPath)
            throws IOException {
        Path colFilePath = new Path(hdfsMSPath, "columns");
        hdfs.mkdirs(colFilePath);
        if(StringUtils.isNotBlank(modelConfig.getDataSet().getCategoricalColumnNameFile())) {
            hdfs.copyFromLocalFile(new Path(modelConfig.getDataSet().getCategoricalColumnNameFile()), colFilePath);
        }
        if(StringUtils.isNotBlank(modelConfig.getDataSet().getMetaColumnNameFile())) {
            hdfs.copyFromLocalFile(new Path(modelConfig.getDataSet().getMetaColumnNameFile()), colFilePath);
        }
        if(StringUtils.isNotBlank(modelConfig.getVarSelect().getForceSelectColumnNameFile())) {
            hdfs.copyFromLocalFile(new Path(modelConfig.getVarSelect().getForceSelectColumnNameFile()), colFilePath);
        }
        if(StringUtils.isNotBlank(modelConfig.getVarSelect().getCandidateColumnNameFile())) {
            hdfs.copyFromLocalFile(new Path(modelConfig.getVarSelect().getCandidateColumnNameFile()), colFilePath);
        }
        if(StringUtils.isNotBlank(modelConfig.getVarSelect().getForceRemoveColumnNameFile())) {
            hdfs.copyFromLocalFile(new Path(modelConfig.getVarSelect().getForceRemoveColumnNameFile()), colFilePath);
        }
        if(StringUtils.isNotBlank(modelConfig.getDataSet().getHybridColumnNameFile())) {
            hdfs.copyFromLocalFile(new Path(modelConfig.getDataSet().getHybridColumnNameFile()), colFilePath);
        }
        if(StringUtils.isNotBlank(modelConfig.getDataSet().getSegExpressionFile())) {
            hdfs.copyFromLocalFile(new Path(modelConfig.getDataSet().getSegExpressionFile()), colFilePath);
        }
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
            FileSystem localFs = HDFSUtils.getLocalFS();
            PathFinder pathFinder = new PathFinder(modelConfig);

            Path evalDir = new Path(pathFinder.getEvalSetPath(evalConfig, SourceType.LOCAL));
            Path dst = new Path(pathFinder.getEvalSetPath(evalConfig, SourceType.HDFS));
            FileSystem hdfs = HDFSUtils.getFS(dst);
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
                // grid-search config file is uploaded to modelset path
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
        List<ColumnConfig> columnConfigList = loadColumnConfigList(Constants.LOCAL_COLUMN_CONFIG_JSON,
                SourceType.LOCAL);
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
    public static synchronized List<ColumnConfig> loadColumnConfigList(String path, SourceType sourceType,
            boolean nullSampleValues) throws IOException {
        ColumnConfig[] configList = loadJSON(path, sourceType, ColumnConfig[].class);
        List<ColumnConfig> columnConfigList = new ArrayList<ColumnConfig>();
        for(ColumnConfig columnConfig: configList) {
            // reset sample values to null to save memory
            if(nullSampleValues) {
                columnConfig.setSampleValues(null);
            }

            // construct Category Index map for fast query.
            if(columnConfig.isCategorical() && columnConfig.getColumnBinning() != null
                    && columnConfig.getColumnBinning().getBinCategory() != null) {
                List<String> categories = columnConfig.getColumnBinning().getBinCategory();
                Map<String, Integer> categoryIndexMapping = new HashMap<String, Integer>();
                for(int i = 0; i < categories.size(); i++) {
                    String category = categories.get(i);
                    if(category.contains(Constants.CATEGORICAL_GROUP_VAL_DELIMITER)) {
                        // merged category should be flatten, use split function this class to avoid depending on guava
                        String[] splits = ml.shifu.shifu.core.dtrain.StringUtils.split(category,
                                Constants.CATEGORICAL_GROUP_VAL_DELIMITER);
                        for(String str: splits) {
                            categoryIndexMapping.put(str, i);
                        }
                    } else {
                        categoryIndexMapping.put(category, i);
                    }
                }
                columnConfig.getColumnBinning().setBinCateMap(categoryIndexMapping);
            }
            columnConfigList.add(columnConfig);
        }
        return columnConfigList;
    }

    /**
     * Some column name has illegal chars which are all be normed in shifu. This is a hook to norm column name but
     * actually so far it is just return;
     *
     * @param columnName
     *            the column name to be normed
     * @return normed column name
     */
    public static String normColumnName(String columnName) {
        if(columnName == null) {
            // NPE protection
            return columnName;
        }
        return columnName.replaceAll("\\.", "_").replaceAll(" ", "_").replaceAll("/", "_").replaceAll("-", "_");
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
            fields = CommonUtils.getHeaders(modelConfig.getHeaderPath(), modelConfig.getHeaderDelimiter(),
                    modelConfig.getDataSet().getSource());
        } else {
            fields = CommonUtils.takeFirstLine(modelConfig.getDataSetRawPath(),
                    StringUtils.isBlank(modelConfig.getHeaderDelimiter()) ? modelConfig.getDataSetDelimiter()
                            : modelConfig.getHeaderDelimiter(),
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
            fields[i] = normColumnName(fields[i]);
        }
        return fields;
    }

    public static String[] getFinalHeaders(ModelConfig modelConfig, EvalConfig evalConfig) throws IOException {
        String[] fields = null;
        boolean isSchemaProvided = true;
        if(StringUtils.isNotBlank(evalConfig.getDataSet().getHeaderPath())) {
            String delimiter = StringUtils.isBlank(evalConfig.getDataSet().getHeaderDelimiter())
                    ? evalConfig.getDataSet().getDataDelimiter()
                    : evalConfig.getDataSet().getHeaderDelimiter();
            fields = CommonUtils.getHeaders(evalConfig.getDataSet().getHeaderPath(), delimiter,
                    evalConfig.getDataSet().getSource());
        } else {
            fields = CommonUtils.takeFirstLine(evalConfig.getDataSet().getDataPath(),
                    StringUtils.isBlank(evalConfig.getDataSet().getHeaderDelimiter())
                            ? evalConfig.getDataSet().getDataDelimiter()
                            : evalConfig.getDataSet().getHeaderDelimiter(),
                    evalConfig.getDataSet().getSource());
            if(evalConfig.isMultiTask()) {
                List<String> tarColumns = evalConfig.getMultiTaskTargetColumnNames();
                if(CollectionUtils.isNotEmpty(tarColumns)) {
                    for(String column: tarColumns) {
                        if(!StringUtils.join(fields, "").contains(column)) {
                            isSchemaProvided = false;
                            break;
                        }
                    }
                }
            } else if(StringUtils.join(fields, "").contains(modelConfig.getTargetColumnName(evalConfig, ""))) {
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
            fields[i] = normColumnName(fields[i]);
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
            throw new IllegalArgumentException(
                    String.format("Null or empty parameters srcDataPath:%s, delimiter:%s, sourceType:%s", pathHeader,
                            delimiter, sourceType));
        }
        BufferedReader reader = null;
        String pigHeaderStr = null;

        try {
            reader = ShifuFileUtils.getReader(pathHeader, sourceType);
            pigHeaderStr = reader.readLine();
            if(StringUtils.isEmpty(pigHeaderStr)) {
                throw new RuntimeException(
                        String.format("Cannot reade header info from the first line of file: %s", pathHeader));
            }
        } catch (Exception e) {
            log.error(
                    "Error in getReader, this must be catched in this method to make sure the next reader can be returned.",
                    e);
            throw new ShifuException(ShifuErrorCode.ERROR_HEADER_NOT_FOUND);
        } finally {
            IOUtils.closeQuietly(reader);
        }
        return calculateHeaders(pigHeaderStr, delimiter, isFull);
    }

    /**
     * Return header column array from header string.
     *
     * @param pigHeaderStr
     *            header string
     * @param delimiter
     *            the delimiter of headers
     * @param isFull
     *            if full header name including name space
     * @return headers array
     */
    public static String[] calculateHeaders(String pigHeaderStr, String delimiter, boolean isFull) {
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
            columnName = normColumnName(columnName);
            if(headerSet.contains(columnName)) {
                columnName = getUniqueName(headerSet, columnName + "_dup" + index);
            }

            headerSet.add(columnName);
            index++;
            headerList.add(columnName);
        }
        return headerList.toArray(new String[0]);
    }

    /**
     * Get the unique name.
     *
     * @return name if name set doesn't contains it. If name exist in name set, it will check name_1, name_2, name_n to
     *         find one which doesn't
     *         exist in the set.
     */
    public static String getUniqueName(Set<String> nameSet, String name) {
        if(nameSet == null || name == null) {
            return name;
        }
        String newName = name;
        for(int i = 1; nameSet.contains(newName); i++) {
            newName = name + "_" + i;
        }
        return newName;
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
            throw new IllegalArgumentException(String
                    .format("raw and delimeter should not be null or empty, raw:%s, delimeter:%s", raw, delimiter));
        }
        List<String> headerList = new ArrayList<String>();
        for(String str: Splitter.on(delimiter).split(raw)) {
            headerList.add(str);
        }
        return headerList;
    }

    /**
     * Common split function to ignore special character like '|'.
     *
     * @param raw
     *            raw string
     * @param splitter
     *            the splitter to split the string
     * @return list of split Strings
     * @throws IllegalArgumentException
     *             {@code raw} and {@code splitter} is null or empty.
     */
    public static List<String> splitAndReturnList(String raw, Splitter splitter) {
        if(StringUtils.isEmpty(raw) || splitter == null) {
            throw new IllegalArgumentException(
                    String.format("raw and delimeter should not be null or empty, raw:%s, splitter:%s", raw, splitter));
        }
        List<String> headerList = new ArrayList<String>();
        for(String str: splitter.split(raw)) {
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
        return getTargetColumnConfig(columnConfigList).getColumnNum();
    }

    /**
     * Get target ColumnConfig.
     *
     * @param columnConfigList
     *            column config list
     * @return target ColumnConfig
     * @throws IllegalArgumentException
     *             if columnConfigList is null or empty.
     *
     * @throws IllegalStateException
     *             if no target column can be found.
     */
    public static ColumnConfig getTargetColumnConfig(List<ColumnConfig> columnConfigList) {
        if(CollectionUtils.isEmpty(columnConfigList)) {
            throw new IllegalArgumentException("columnConfigList should not be null or empty.");
        }
        // I need cast operation because of common-collections doesn't support generic.
        ColumnConfig cc = (ColumnConfig) CollectionUtils.find(columnConfigList, new Predicate() {
            @Override
            public boolean evaluate(Object object) {
                return ((ColumnConfig) object).isTarget();
            }
        });
        if(cc == null) {
            throw new IllegalStateException("No target column can be found, please check your column configurations");
        }
        return cc;
    }

    /**
     * Get ColumnConfig from ColumnConfig list by columnId, since the columnId may not represent the position
     * in ColumnConfig list after the segments (Column Expansion).
     *
     * @param columnConfigList
     *            - list of ColumnConfig
     * @param columnId
     *            - the column id that want to search
     * @return - ColumnConfig
     */
    public static ColumnConfig getColumnConfig(List<ColumnConfig> columnConfigList, Integer columnId) {
        for(ColumnConfig columnConfig: columnConfigList) {
            if(columnConfig.getColumnNum().equals(columnId)) {
                return columnConfig;
            }
        }
        return null;
    }

    public static boolean isLinearTarget(ModelConfig modelConfig, List<ColumnConfig> columnConfigList) {
        ColumnConfig columnConfig = getTargetColumnConfig(columnConfigList);
        if(columnConfig == null) {
            throw new ShifuException(ShifuErrorCode.ERROR_NO_TARGET_COLUMN, "Target column is not detected.");
        }
        return (CollectionUtils.isEmpty(modelConfig.getTags()) && columnConfig.isNumerical());
    }

    public static Set<NSColumn> loadCandidateColumns(ModelConfig modelConfig) throws IOException {
        Set<NSColumn> candidateColumns = new HashSet<NSColumn>();
        List<String> candidates = modelConfig.getListCandidates();
        for(String candidate: candidates) {
            candidateColumns.add(new NSColumn(candidate));
        }
        return candidateColumns;
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

    // /**
    // * Return all parameters for pig execution.
    // *
    // * @param modelConfig
    // * model config
    // * @param sourceType
    // * source type
    // * @return map of configurations
    // * @throws IOException
    // * any io exception
    // * @throws IllegalArgumentException
    // * if modelConfig is null.
    // */
    // public static Map<String, String> getPigParamMap(ModelConfig modelConfig, SourceType sourceType)
    // throws IOException {
    // if(modelConfig == null) {
    // throw new IllegalArgumentException("modelConfig should not be null.");
    // }
    // PathFinder pathFinder = new PathFinder(modelConfig);
    //
    // Map<String, String> pigParamMap = new HashMap<String, String>();
    // pigParamMap.put(Constants.NUM_PARALLEL, Environment.getInt(Environment.HADOOP_NUM_PARALLEL, 400).toString());
    // log.info("jar path is {}", pathFinder.getJarPath());
    // pigParamMap.put(Constants.PATH_JAR, pathFinder.getJarPath());
    //
    // pigParamMap.put(Constants.PATH_RAW_DATA, modelConfig.getDataSetRawPath());
    // pigParamMap.put(Constants.PATH_NORMALIZED_DATA, pathFinder.getNormalizedDataPath(sourceType));
    // // default norm is not for clean, so set it to false, this will be overrided in Train#Norm for tree models
    // pigParamMap.put(Constants.IS_NORM_FOR_CLEAN, Boolean.FALSE.toString());
    // pigParamMap.put(Constants.PATH_PRE_TRAINING_STATS, pathFinder.getPreTrainingStatsPath(sourceType));
    // pigParamMap.put(Constants.PATH_STATS_BINNING_INFO, pathFinder.getUpdatedBinningInfoPath(sourceType));
    // pigParamMap.put(Constants.PATH_STATS_PSI_INFO, pathFinder.getPSIInfoPath(sourceType));
    //
    // pigParamMap.put(Constants.WITH_SCORE, Boolean.FALSE.toString());
    // pigParamMap.put(Constants.STATS_SAMPLE_RATE, modelConfig.getBinningSampleRate().toString());
    // pigParamMap.put(Constants.PATH_MODEL_CONFIG, pathFinder.getModelConfigPath(sourceType));
    // pigParamMap.put(Constants.PATH_COLUMN_CONFIG, pathFinder.getColumnConfigPath(sourceType));
    // pigParamMap.put(Constants.PATH_SELECTED_RAW_DATA, pathFinder.getSelectedRawDataPath(sourceType));
    // pigParamMap.put(Constants.PATH_BIN_AVG_SCORE, pathFinder.getBinAvgScorePath(sourceType));
    // pigParamMap.put(Constants.PATH_TRAIN_SCORE, pathFinder.getTrainScoresPath(sourceType));
    //
    // pigParamMap.put(Constants.SOURCE_TYPE, sourceType.toString());
    // pigParamMap.put(Constants.JOB_QUEUE,
    // Environment.getProperty(Environment.HADOOP_JOB_QUEUE, Constants.DEFAULT_JOB_QUEUE));
    // return pigParamMap;
    // }

    public static Map<String, String> getPigParamMap(ModelConfig modelConfig, SourceType sourceType,
            PathFinder pathFinder) throws IOException {
        return getPigParamMap(modelConfig, sourceType, pathFinder, -1);
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
     * @param mtlIndex
     *            the multi task learning index
     * @return map of configurations
     * @throws IOException
     *             any io exception
     * @throws IllegalArgumentException
     *             if modelConfig is null.
     */
    public static Map<String, String> getPigParamMap(ModelConfig modelConfig, SourceType sourceType,
            PathFinder pathFinder, int mtlIndex) throws IOException {
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

        if(modelConfig.isMultiTask()) {
            pigParamMap.put(Constants.PATH_PRE_TRAINING_STATS,
                    pathFinder.getPreTrainingStatsPath(sourceType, mtlIndex));
            pigParamMap.put(Constants.PATH_STATS_BINNING_INFO,
                    pathFinder.getUpdatedBinningInfoPath(sourceType, mtlIndex));
            pigParamMap.put(Constants.PATH_STATS_PSI_INFO, pathFinder.getPSIInfoPath(sourceType, mtlIndex));
            pigParamMap.put(Constants.PATH_COLUMN_CONFIG, pathFinder.getMTLColumnConfigPath(sourceType, mtlIndex));
        } else {
            pigParamMap.put(Constants.PATH_PRE_TRAINING_STATS, pathFinder.getPreTrainingStatsPath(sourceType));
            pigParamMap.put(Constants.PATH_STATS_BINNING_INFO, pathFinder.getUpdatedBinningInfoPath(sourceType));
            pigParamMap.put(Constants.PATH_STATS_PSI_INFO, pathFinder.getPSIInfoPath(sourceType));
            pigParamMap.put(Constants.PATH_COLUMN_CONFIG, pathFinder.getColumnConfigPath(sourceType));
        }

        pigParamMap.put(Constants.PATH_STATS_PSI_INFO, pathFinder.getPSIInfoPath(sourceType));

        pigParamMap.put(Constants.WITH_SCORE, Boolean.FALSE.toString());
        pigParamMap.put(Constants.STATS_SAMPLE_RATE, modelConfig.getBinningSampleRate().toString());
        pigParamMap.put(Constants.PATH_MODEL_CONFIG, pathFinder.getModelConfigPath(sourceType));
        pigParamMap.put(Constants.PATH_SELECTED_RAW_DATA, pathFinder.getSelectedRawDataPath(sourceType));
        pigParamMap.put(Constants.PATH_BIN_AVG_SCORE, pathFinder.getBinAvgScorePath(sourceType));
        pigParamMap.put(Constants.PATH_TRAIN_SCORE, pathFinder.getTrainScoresPath(sourceType));

        pigParamMap.put(Constants.SOURCE_TYPE, sourceType.toString());
        pigParamMap.put(Constants.JOB_QUEUE,
                Environment.getProperty(Environment.HADOOP_JOB_QUEUE, Constants.DEFAULT_JOB_QUEUE));
        pigParamMap.put(Constants.DATASET_NAME, modelConfig.getBasic().getName());

        pigParamMap.put(Constants.SHIFU_OUTPUT_DELIMITER, CommonUtils.escapePigString(
                Environment.getProperty(Constants.SHIFU_OUTPUT_DATA_DELIMITER, Constants.DEFAULT_DELIMITER)));

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
        return checkAndReturnSplitCollections(str);
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
        return checkAndReturnSplitCollections(str, separator);
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

    public static boolean isTreeModel(String alg) {
        return CommonConstants.RF_ALG_NAME.equalsIgnoreCase(alg) || CommonConstants.GBT_ALG_NAME.equalsIgnoreCase(alg);
    }

    public static boolean isTensorFlowModel(String alg) {
        return CommonConstants.TF_ALG_NAME.equalsIgnoreCase(alg);
    }

    public static boolean isNNModel(String alg) {
        return "nn".equalsIgnoreCase(alg);
    }

    public static boolean isLRModel(String alg) {
        return "lr".equalsIgnoreCase(alg);
    }

    public static boolean isWDLModel(String alg) {
        return "wdl".equalsIgnoreCase(alg);
    }

    public static boolean isMTLModel(String alg) {
        return "mtl".equalsIgnoreCase(alg);
    }

    public static boolean isWeightColumn(String weightColumnName, ColumnConfig columnConfig) {
        return StringUtils.isNotBlank(weightColumnName) // weight is set && equals with name of ColumnConfig
                && StringUtils.equals(StringUtils.trim(weightColumnName), columnConfig.getColumnName());
    }

    public static boolean isRandomForestAlgorithm(String alg) {
        return CommonConstants.RF_ALG_NAME.equalsIgnoreCase(alg);
    }

    public static boolean isGBDTAlgorithm(String alg) {
        return CommonConstants.GBT_ALG_NAME.equalsIgnoreCase(alg);
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

    public static List<String> readConfNamesAsList(String columnConfFile, SourceType sourceType, String delimiter)
            throws IOException {
        List<String> columnNameList = new ArrayList<String>();

        List<String> fileLines = readConfFileIntoList(columnConfFile, sourceType);
        if(CollectionUtils.isEmpty(fileLines)) {
            return fileLines;
        }

        for(String line: fileLines) {
            for(String str: Splitter.on(delimiter).split(line)) {
                // String column = CommonUtils.getRelativePigHeaderColumnName(str);
                if(StringUtils.isNotBlank(str)) {
                    str = StringUtils.trim(str);
                    str = normColumnName(str);
                    columnNameList.add(str);
                }
            }
        }

        return columnNameList;
    }

    public static List<String> readConfFileIntoList(String configFile, SourceType sourceType) throws IOException {
        List<String> fileLines = new ArrayList<String>();

        if(StringUtils.isBlank(configFile) || !ShifuFileUtils.isFileExists(configFile, sourceType)) {
            return fileLines;
        }

        List<String> strList = null;
        Reader reader = null;
        try {
            reader = ShifuFileUtils.getReader(configFile, sourceType);
            strList = IOUtils.readLines(reader);
        } finally {
            IOUtils.closeQuietly(reader);
        }

        if(CollectionUtils.isNotEmpty(strList)) {
            for(String line: strList) { // skip empty line and line start with "#"
                if(StringUtils.isBlank(line) || line.trim().startsWith("#")) {
                    continue;
                }
                fileLines.add(StringUtils.trim(line));
            }
        }

        return fileLines;
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

    /**
     * Check whether to normalize one variable or not
     *
     * @param columnConfig
     *            - ColumnConfig to check
     * @param hasCandidate
     *            - Are candidates set or not
     * @param isBinaryClassification
     *            - Is it binary classification?
     * @return
     *         true - should normalize
     *         The variable is finalSelected and it is good variable
     *         Or
     *         It's a good candidate
     *         false - don't normalize
     */
    public static boolean isToNormVariable(ColumnConfig columnConfig, boolean hasCandidate,
            boolean isBinaryClassification) {
        if(columnConfig == null) {
            return false;
        }
        return (columnConfig.isFinalSelect() && isGoodVariable(columnConfig, isBinaryClassification))
                || isGoodCandidate(columnConfig, hasCandidate, isBinaryClassification);
    }

    /**
     * Check the variable is good candidate or not
     *
     * @param columnConfig
     *            - ColumnConfig to check
     * @param hasCandidate
     *            - Are candidates set or not
     * @param isBinaryClassification
     *            - Is it binary classification?
     * @return
     *         true - is good candidate
     *         false - bad candidate
     */
    public static boolean isGoodCandidate(ColumnConfig columnConfig, boolean hasCandidate,
            boolean isBinaryClassification) {
        if(columnConfig == null) {
            return false;
        }

        if(isBinaryClassification) {
            return isGoodCandidate(columnConfig, hasCandidate);
        } else {
            // multiple classification
            return columnConfig.isCandidate(hasCandidate) && isGoodVariable(columnConfig, isBinaryClassification);
        }
    }

    /**
     * Check the variable is good candidate or not
     *
     * @param columnConfig
     *            - ColumnConfig to check
     * @param hasCandidate
     *            - Are candidates set or not
     * @return
     *         true - is good candidate
     *         false - bad candidate
     */
    public static boolean isGoodCandidate(ColumnConfig columnConfig, boolean hasCandidate) {
        if(columnConfig == null) {
            return false;
        }

        return columnConfig.isCandidate(hasCandidate) && isGoodVariable(columnConfig, true);
    }

    /**
     * Check whether a variable is good or bad
     *
     * @param columnConfig
     *            - ColumnConfig to check
     * @param isBinaryClassification
     *            - Is it binary classification?
     * @return
     *         true - is good variable
     *         false - bad variable
     */
    public static boolean isGoodVariable(ColumnConfig columnConfig, boolean isBinaryClassification) {
        boolean varCondition = (columnConfig.getMean() != null && columnConfig.getStdDev() != null
                && ((columnConfig.isCategorical() && columnConfig.getBinCategory() != null
                        && columnConfig.getBinCategory().size() > 0)
                        || (columnConfig.isNumerical() && columnConfig.getBinBoundary() != null
                                && columnConfig.getBinBoundary().size() > 0)));
        if(isBinaryClassification) {
            varCondition = varCondition && (columnConfig.getKs() != null && columnConfig.getKs() > 0
                    && columnConfig.getIv() != null && columnConfig.getIv() > 0);
        }
        return varCondition;
    }

    /**
     * Return first line split string array. This is used to detect data schema.
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
    public static String[] takeFirstLine(String dataSetRawPath, String delimiter, SourceType source)
            throws IOException {
        if(dataSetRawPath == null || delimiter == null || source == null) {
            throw new IllegalArgumentException("Input parameters should not be null.");
        }

        String firstValidFile = null;
        Path filePath = new Path(dataSetRawPath);
        FileSystem fs = ShifuFileUtils.getFileSystemBySourceType(source, filePath);
        FileStatus[] globStatus = fs.globStatus(filePath, HiddenPathFilter.getHiddenPathFilter());
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
                for(String unit: Splitter.on(delimiter).split(firstLine)) {
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
        // "Segment expansion enabled but # of columns in ColumnConfig.json is not consistent with segment expansion
        // files.");
        // }

        if(columnConfig.getColumnNum() >= dataSetHeaders.length) {
            return getSimpleColumnName(
                    columnConfigList.get(columnConfig.getColumnNum() % dataSetHeaders.length).getColumnName());
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
        Path filePath = new Path(dataSetRawPath);
        FileSystem fs = ShifuFileUtils.getFileSystemBySourceType(source, filePath);
        FileStatus[] globStatus = fs.globStatus(filePath, HiddenPathFilter.getHiddenPathFilter());
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
                    MutablePair<String, Double> value = MutablePair.of(entry.getValue().getKey(),
                            entry.getValue().getValue() / modelSize);
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

    public static double[] floatToDouble(float[] src) {
        if(src == null) {
            return null;
        }

        double[] output = new double[src.length];
        for(int i = 0; i < src.length; i++) {
            output[i] = src[i];
        }
        return output;
    }

    public static float[] doubleToFloat(double[] src) {
        if(src == null) {
            return null;
        }

        float[] output = new float[src.length];
        for(int i = 0; i < src.length; i++) {
            output[i] = (float) src[i];
        }
        return output;
    }

    public static double[] minus(double[] src, double[] target) {
        if(src == null || target == null) {
            return null;
        }
        assert src.length == target.length;

        double[] output = new double[src.length];
        for(int i = 0; i < src.length; i++) {
            output[i] = src[i] - target[i];
        }
        return output;
    }

    public static double[] plus(double[] src, double[] target) {
        if(src == null || target == null) {
            return null;
        }
        assert src.length == target.length;

        double[] output = new double[src.length];
        for(int i = 0; i < src.length; i++) {
            output[i] = src[i] + target[i];
        }
        return output;
    }

    public static double[] minus(double[] src, float[] target) {
        if(src == null || target == null) {
            return null;
        }
        assert src.length == target.length;

        double[] output = new double[src.length];
        for(int i = 0; i < src.length; i++) {
            output[i] = src[i] - target[i];
        }
        return output;
    }

    public static double[] sigmoid(double[] src) {
        if(src == null) {
            return null;
        }

        double[] output = new double[src.length];
        for(int i = 0; i < src.length; i++) {
            output[i] = sigmoid(src[i]);
        }
        return output;
    }

    /**
     * Inject Shifu or Hadoop parameters into MapReduce / Pig jobs, by using visitor.
     *
     * @param visitor
     *            - provider to do injection
     */
    public static void injectHadoopShifuEnvironments(ValueVisitor visitor) {
        for(Map.Entry<Object, Object> entry: Environment.getProperties().entrySet()) {
            if(CommonUtils.isHadoopConfigurationInjected(entry.getKey().toString())) {
                if(StringUtils.equalsIgnoreCase(entry.getKey().toString(), Constants.SHIFU_OUTPUT_DATA_DELIMITER)) {
                    visitor.inject(entry.getKey(), Base64Utils.base64Encode(entry.getValue().toString()));
                } else {
                    visitor.inject(entry.getKey(), entry.getValue());
                }
            }
        }
    }

    /**
     * Check whether the prefix of key is Shifu or Hadoop-related.
     *
     * @param key
     *            - key to check
     * @return
     *         true - is Shifu or Hadoop related keys
     *         or false
     */
    public static boolean isHadoopConfigurationInjected(String key) {
        return key.startsWith("nn") || key.startsWith("guagua") || key.startsWith("shifu") || key.startsWith("mapred")
                || key.startsWith("io") || key.startsWith("hadoop") || key.startsWith("yarn") || key.startsWith("pig")
                || key.startsWith("hive") || key.startsWith("job");
    }

    public static boolean getBooleanValue(Object object, boolean defaultValue) {
        if(object == null) {
            return defaultValue;
        }
        return Boolean.TRUE.toString().equalsIgnoreCase(object.toString());
    }

    /**
     * Return float value parsed from input string, NaN by default changed to 0.
     * 
     * @param input
     *            the input string
     * @return parsed float value
     */
    public static float getFloatValue(String input) {
        float floatValue = input.length() == 0 ? 0f : NumberFormatUtils.getFloat(input, 0f);
        return (Double.isNaN(floatValue) || Double.isNaN(floatValue)) ? 0f : floatValue;
    }

    /**
     * Sigmoid function definition.
     * 
     * @param logit
     *            the logit value
     * @return sigmoid value
     */
    public static double sigmoid(double logit) {
        return 1.0d / (1.0d + BoundMath.exp(-1 * logit));
    }

    /**
     * Derived function for sigmoid function.
     * 
     * @param result
     *            logit result
     * @return sigmoid derived value
     */
    public static double sigmoidDerivedFunction(double result) {
        return result * (1d - result);
    }

    /**
     * Read the iterable into String array
     * @param split - iterable of text elements
     * @return - elements of text
     */
    public static String[] readIterableToArray(Iterable<String> split) {
        List<String> fields = new ArrayList<>();
        for (String str : split) {
            fields.add(str);
        }
        return fields.toArray(new String[0]);
    }
}
