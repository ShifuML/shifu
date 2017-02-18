/*
 * Copyright [2013-2016] PayPal Software Foundation
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
package ml.shifu.shifu.core.pmml.builder.creator;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import org.dmg.pmml.DataType;
import org.dmg.pmml.OpType;

import java.util.List;

/**
 * Created by zhanhu on 3/29/16.
 */
public abstract class AbstractPmmlElementCreator<T> {

    protected boolean isConcise;
    protected ModelConfig modelConfig;
    protected List<ColumnConfig> columnConfigList;

    public AbstractPmmlElementCreator(ModelConfig modelConfig, List<ColumnConfig> columnConfigList) {
        this(modelConfig, columnConfigList, false);
    }

    public AbstractPmmlElementCreator(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, boolean isConcise) {
        this.modelConfig = modelConfig;
        this.columnConfigList = columnConfigList;
        this.isConcise = isConcise;
    }

    public abstract T build();

    /**
     * Get @OpType from ColumnConfig
     * Meta Column vs. ORDINAL
     * Target Column vs. CATEGORICAL
     * Categorical Column vs. CATEGORICAL
     * Numerical Column vs. CONTINUOUS
     *
     * @param columnConfig
     *            - ColumnConfig for variable
     * @return OpType
     */
    protected OpType getOptype(ColumnConfig columnConfig) {
        if(columnConfig.isMeta()) {
            return OpType.ORDINAL;
        } else if(columnConfig.isTarget()) {
            return OpType.CATEGORICAL;
        } else {
            return (columnConfig.isCategorical() ? OpType.CATEGORICAL : OpType.CONTINUOUS);
        }
    }

    /**
     * Get DataType from OpType
     * CONTINUOUS vs. DOUBLE
     * Other vs. STRING
     *
     * @param optype
     *            OpType
     * @return DataType
     */
    protected DataType getDataType(OpType optype) {
        return (optype.equals(OpType.CONTINUOUS) ? DataType.DOUBLE : DataType.STRING);
    }
}
