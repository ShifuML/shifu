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

import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.Environment;

import org.apache.pig.EvalFunc;
import org.apache.pig.impl.util.UDFContext;
import org.apache.pig.tools.pigstats.PigStatusReporter;

/**
 * AbstractTrainerUDF class is the abstract class for most UDF
 * It will load and host @ModelConfig and @ColumnConfig, and find target column id
 */
public abstract class AbstractTrainerUDF<T> extends EvalFunc<T> {

    protected ModelConfig modelConfig;
    protected List<ColumnConfig> columnConfigList;

    // Need to specify the default value as -1, or it won't report error if it doesn't find target column
    protected int tagColumnNum = -1;

    // cache tags in set for search
    protected Set<String> posTagSet;
    protected Set<String> negTagSet;
    protected Set<String> tagSet;

    protected int maxCategorySize = Constants.MAX_CATEGORICAL_BINC_COUNT;

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
        SourceType sourceType = SourceType.valueOf(source);

        if(pathModelConfig != null) {
            modelConfig = CommonUtils.loadModelConfig(pathModelConfig, sourceType);
        }

        columnConfigList = CommonUtils.loadColumnConfigList(pathColumnConfig, sourceType);
        for(ColumnConfig config: columnConfigList) {
            if(config.isTarget()) {
                tagColumnNum = config.getColumnNum();
                break;
            }
        }

        if(tagColumnNum == -1) {
            throw new RuntimeException("No Valid Target.");
        }

        if(modelConfig != null && modelConfig.getPosTags() != null) {
            this.posTagSet = new HashSet<String>(modelConfig.getPosTags());
        }
        if(modelConfig != null && modelConfig.getNegTags() != null) {
            this.negTagSet = new HashSet<String>(modelConfig.getNegTags());
        }
        if(modelConfig != null && modelConfig.getFlattenTags() != null) {
            this.tagSet = new HashSet<String>(modelConfig.getFlattenTags());
        }

        if(UDFContext.getUDFContext() != null && UDFContext.getUDFContext().getJobConf() != null) {
            this.maxCategorySize = UDFContext.getUDFContext().getJobConf()
                    .getInt(Constants.SHIFU_MAX_CATEGORY_SIZE, Constants.MAX_CATEGORICAL_BINC_COUNT);
        } else {
            this.maxCategorySize = Environment.getInt(Constants.SHIFU_MAX_CATEGORY_SIZE,
                    Constants.MAX_CATEGORICAL_BINC_COUNT);
        }

        this.hasCandidates = CommonUtils.hasCandidateColumns(this.columnConfigList);
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
}
