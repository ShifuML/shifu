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
package ml.shifu.shifu.udf;

import java.util.HashMap;
import java.util.Map;
import ml.shifu.guagua.util.NumberFormatUtils;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelNormalizeConf.NormType;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.Environment;
import ml.shifu.shifu.util.HDFSUtils;
import org.apache.hadoop.fs.FileSystem;
import org.apache.pig.EvalFunc;
import org.apache.pig.impl.util.UDFContext;
import org.apache.pig.tools.pigstats.PigStatusReporter;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * AbstractTrainerUDF class is the abstract class for most UDF
 * It will load and host @ModelConfig and @ColumnConfig, and find target column id
 */
public abstract class AbstractTrainerUDF<T> extends EvalFunc<T> {

    // This cache aims to decrease the memory usage when column config is very large.
    private static Map<String, List<ColumnConfig>> columnConfigCache = new HashMap<>(1);

    protected ModelConfig modelConfig;
    protected List<ColumnConfig> columnConfigList;

    protected int tagColumnNum;

    // cache tags in set for search
    protected Set<String> posTagSet;
    protected Set<String> negTagSet;
    protected Set<String> tagSet;

    protected int maxCategorySize;

    protected boolean hasCandidates;

    /**
     * Constructor with SourceType, ModelConfig path and ColumnConfig path
     * 
     * @param source
     *            - source type of ModelConfig file and ColumnConfig file
     * @param pathModelConfig
     *            - the path of ModelConfig
     * @param pathColumnConfig
     *            - the path of ColumnConfig
     * @throws IOException
     *             throw exceptions when loading configuration
     */
    public AbstractTrainerUDF(String source, String pathModelConfig, String pathColumnConfig) throws IOException {
        // inject fs.defaultFS from UDFContext.getUDFContext().getJobConf()
        if (UDFContext.getUDFContext() != null && UDFContext.getUDFContext().getJobConf() != null) {
            HDFSUtils.getConf().set(FileSystem.FS_DEFAULT_NAME_KEY,
                    UDFContext.getUDFContext().getJobConf().get(FileSystem.FS_DEFAULT_NAME_KEY));
        }

        SourceType sourceType = SourceType.valueOf(source);

        if(pathModelConfig != null) {
            modelConfig = CommonUtils.loadModelConfig(pathModelConfig, sourceType);
            if(modelConfig.isMultiTask()) {
                int mtlIndex = -1;
                if(UDFContext.getUDFContext() != null && UDFContext.getUDFContext().getJobConf() != null) {
                    mtlIndex = NumberFormatUtils
                            .getInt(UDFContext.getUDFContext().getJobConf().get(CommonConstants.MTL_INDEX), -1);
                } else {
                    // "when do local initilization mtlIndex is -1, set to 0 to pass");
                    mtlIndex = 0;
                }
                modelConfig.setMtlIndex(mtlIndex);
            }
        }

        columnConfigList = loadColumnConfigList(pathColumnConfig, sourceType);
        tagColumnNum = CommonUtils.getTargetColumnNum(columnConfigList);

        if(modelConfig != null && modelConfig.getPosTags() != null) {
            this.posTagSet = new HashSet<>(modelConfig.getPosTags());
        }
        if(modelConfig != null && modelConfig.getNegTags() != null) {
            this.negTagSet = new HashSet<>(modelConfig.getNegTags());
        }
        if(modelConfig != null && modelConfig.getFlattenTags() != null) {
            this.tagSet = new HashSet<>(modelConfig.getFlattenTags());
        }

        if(UDFContext.getUDFContext() != null && UDFContext.getUDFContext().getJobConf() != null) {
            this.maxCategorySize = UDFContext.getUDFContext().getJobConf().getInt(Constants.SHIFU_MAX_CATEGORY_SIZE,
                    Constants.MAX_CATEGORICAL_BINC_COUNT);
        } else {
            this.maxCategorySize = Environment.getInt(Constants.SHIFU_MAX_CATEGORY_SIZE,
                    Constants.MAX_CATEGORICAL_BINC_COUNT);
        }

        this.hasCandidates = CommonUtils.hasCandidateColumns(this.columnConfigList);
    }
    
    /**
     * Load column config list.
     *
     * @param pathColumnConfig is the column config file path.
     * @param sourceType       is the source type: HDFS, LOCAL, etc.
     * @return the column config list.
     * @throws IOException if load column config failed.
     */
    private static synchronized List<ColumnConfig> loadColumnConfigList(String pathColumnConfig, SourceType sourceType) throws IOException {
        // Return the cached column config if it exists. This action will avoid loading same column config file more times.
        String key = pathColumnConfig + "," + sourceType;
        List<ColumnConfig> cachedColumnConfigList = columnConfigCache.get(key);
        if (cachedColumnConfigList == null) {
            cachedColumnConfigList = CommonUtils.loadColumnConfigList(pathColumnConfig, sourceType);
            columnConfigCache.put(key, cachedColumnConfigList);
        }
        return cachedColumnConfigList;
    }

    /**
     * Constructor with SourceType, and ColumnConfig path
     * 
     * @param source
     *            - source type of ColumnConfig file
     * @param pathColumnConfig
     *            - the path of ColumnConfig
     * @throws IOException
     *             throw exceptions when loading configuration
     */
    public AbstractTrainerUDF(String source, String pathColumnConfig) throws IOException {
        this(source, null, pathColumnConfig);
    }

    /*
     * Check whether is a pig environment, for example, in unit test, PigStatusReporter.getInstance() is null
     */
    @SuppressWarnings("deprecation")
    protected boolean isPigEnabled(String group, String counter) {
        return PigStatusReporter.getInstance() != null
                && PigStatusReporter.getInstance().getCounter(group, counter) != null;
    }

    /**
     * Get property value from UDF job context, or get it from @Environment
     * 
     * @param udfPropertyName
     *            UDF property name
     * @param defval
     *            default value, if there is no such property
     * @return property value or default value
     */
    protected String getUdfProperty(String udfPropertyName, String defval) {
        String udfPropertyVal;
        if(UDFContext.getUDFContext() != null && UDFContext.getUDFContext().getJobConf() != null) {
            udfPropertyVal = UDFContext.getUDFContext().getJobConf().get(udfPropertyName, defval);
        } else {
            udfPropertyVal = Environment.getProperty(udfPropertyName, defval);
        }
        return udfPropertyVal;
    }

    /**
     * Get property value from UDF job context, or get it from @Environment
     * 
     * @param udfPropertyName
     *            UDF property name
     * @return property value or null
     */
    protected String getUdfProperty(String udfPropertyName) {
        return getUdfProperty(udfPropertyName, null);
    }

    /**
     * Generate the normalized Column names for one config
     * 
     * @param config
     *            - ColumnConfig to norm
     * @param normType
     *            - normalization type
     * @return
     *         if the NormType is ONEHOT, it will be normalized to multi variables
     *         or it will be just one normalized column name
     */
    protected List<String> genNormColumnNames(ColumnConfig config, NormType normType) {
        List<String> normalizedNames = new ArrayList<>();
        if(NormType.ONEHOT.equals(normType) && config.isNumerical()) { // ONEHOT and numerical variable
            for(int i = 0; i < config.getBinBoundary().size(); i++) {
                normalizedNames.add(CommonUtils.normColumnName(config.getColumnName()) + "_" + i);
            }
            normalizedNames.add(CommonUtils.normColumnName(config.getColumnName()) + "_missing");
        } else if((NormType.ONEHOT.equals(normType) || NormType.ZSCALE_ONEHOT.equals(normType))
                && config.isCategorical()) { // ONEHOT or ZSCALE_ONEHOT for categorical variable
            for(int i = 0; i < config.getBinCategory().size(); i++) {
                normalizedNames.add(CommonUtils.normColumnName(config.getColumnName()) + "_" + i);
            }
            normalizedNames.add(CommonUtils.normColumnName(config.getColumnName()) + "_missing");
        } else if(NormType.ZSCALE_APPEND_INDEX.equals(normType) || NormType.ZSCORE_APPEND_INDEX.equals(normType)
                || NormType.WOE_APPEND_INDEX.equals(normType) || NormType.WOE_ZSCALE_APPEND_INDEX.equals(normType)) {
            normalizedNames.add(CommonUtils.normColumnName(config.getColumnName()));
            normalizedNames.add(CommonUtils.normColumnName(config.getColumnName()) + "_index");
        } else {
            normalizedNames.add(CommonUtils.normColumnName(config.getColumnName()));
        }
        return normalizedNames;
    }

    protected List<String> genMTLNormColumnNames(ColumnConfig config, NormType normType, int mtlIndex) {
        List<String> normalizedNames = new ArrayList<>();
        if(NormType.ONEHOT.equals(normType) && config.isNumerical()) { // ONEHOT and numerical variable
            for(int i = 0; i < config.getBinBoundary().size(); i++) {
                normalizedNames.add(CommonUtils.normColumnName(config.getColumnName()) + "_" + mtlIndex + "_" + i);
            }
            normalizedNames.add(CommonUtils.normColumnName(config.getColumnName()) + "_" + mtlIndex + "_missing");
        } else if((NormType.ONEHOT.equals(normType) || NormType.ZSCALE_ONEHOT.equals(normType))
                && config.isCategorical()) { // ONEHOT or ZSCALE_ONEHOT for categorical variable
            for(int i = 0; i < config.getBinCategory().size(); i++) {
                normalizedNames.add(CommonUtils.normColumnName(config.getColumnName()) + "_" + mtlIndex + "_" + i);
            }
            normalizedNames.add(CommonUtils.normColumnName(config.getColumnName()) + "_" + mtlIndex + "_missing");
        } else {
            normalizedNames.add(CommonUtils.normColumnName(config.getColumnName()) + "_" + mtlIndex);
        }
        return normalizedNames;
    }
}
