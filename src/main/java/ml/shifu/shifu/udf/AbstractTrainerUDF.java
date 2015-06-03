/**
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

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.util.CommonUtils;
import org.apache.pig.EvalFunc;

import java.io.IOException;
import java.util.List;


/**
 * AbstractTrainerUDF class is the abstract class for most UDF
 * It will load and host @ModelConfig and @ColumnConfig, and find target column id
 */
public abstract class AbstractTrainerUDF<T> extends EvalFunc<T> {
    protected ModelConfig modelConfig;
    protected List<ColumnConfig> columnConfigList;

    // Need to specify the default value as -1, or it won't report error if it doesn't find target column
    protected int tagColumnNum = -1;

    /**
     * Constructor with SourceType, ModelConfig path and ColumnConfig path
     *
     * @param source           - source type of ModelConfig file and ColumnConfig file
     * @param pathModelConfig  - the path of ModelConfig
     * @param pathColumnConfig - the path of ColumnConfig
     * @throws IOException throw exceptions when loading configuration
     */
    public AbstractTrainerUDF(String source, String pathModelConfig, String pathColumnConfig) throws IOException {
        SourceType sourceType = SourceType.valueOf(source);

        if (pathModelConfig != null) {
            modelConfig = CommonUtils.loadModelConfig(pathModelConfig, sourceType);
        }

        columnConfigList = CommonUtils.loadColumnConfigList(pathColumnConfig, sourceType);
        for (ColumnConfig config : columnConfigList) {
            if (config.isTarget()) {
                tagColumnNum = config.getColumnNum();
                break;
            }
        }

        if (tagColumnNum == -1) {
            throw new RuntimeException("No Valid Target.");
        }
    }

    /**
     * Constructor with SourceType, and ColumnConfig path
     *
     * @param source           - source type of ColumnConfig file
     * @param pathColumnConfig - the path of ColumnConfig
     * @throws IOException throw exceptions when loading configuration
     */
    public AbstractTrainerUDF(String source, String pathColumnConfig) throws IOException {
        this(source, null, pathColumnConfig);
    }
}
