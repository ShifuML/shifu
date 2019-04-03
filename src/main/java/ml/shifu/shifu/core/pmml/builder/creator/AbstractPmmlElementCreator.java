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
import ml.shifu.shifu.util.CommonUtils;

import org.apache.commons.collections.CollectionUtils;
import org.dmg.pmml.DataType;
import org.dmg.pmml.OpType;
import org.encog.ml.BasicML;

import java.io.IOException;
import java.util.List;
import java.util.Set;

/**
 * Created by zhanhu on 3/29/16.
 */
public abstract class AbstractPmmlElementCreator<T> {

    protected boolean isConcise;
    protected ModelConfig modelConfig;
    protected List<ColumnConfig> columnConfigList;

    /**
     * Raw data set headers, after segment expansions, columnConfigList is not for all columns but with column segment
     * expansion. By using data set headers, still we can check raw columns.
     */
    protected String[] datasetHeaders;

    /**
     * Different segment expansions which is used to address raw columns.
     */
    protected List<String> segmentExpansions;

    public AbstractPmmlElementCreator(ModelConfig modelConfig, List<ColumnConfig> columnConfigList) {
        this(modelConfig, columnConfigList, false);
    }

    public AbstractPmmlElementCreator(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, boolean isConcise) {
        this.modelConfig = modelConfig;
        this.columnConfigList = columnConfigList;
        this.isConcise = isConcise;
        try {
            this.datasetHeaders = CommonUtils.getFinalHeaders(modelConfig);
            this.segmentExpansions = modelConfig.getSegmentFilterExpressions();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

    }

    public abstract T build(BasicML basicML);

    /**
     * Get OpType from ColumnConfig
     * Meta Column -&gt; ORDINAL
     * Target Column -&gt; CATEGORICAL
     * Categorical Column -&gt; CATEGORICAL
     * Numerical Column -&gt; CONTINUOUS
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
     * CONTINUOUS -&gt; DOUBLE
     * Other -&gt; STRING
     * 
     * @param optype
     *            OpType
     * @return DataType
     */
    protected DataType getDataType(OpType optype) {
        return (optype.equals(OpType.CONTINUOUS) ? DataType.DOUBLE : DataType.STRING);
    }

    protected boolean isActiveColumn(Set<Integer> featureSet, ColumnConfig columnConfig) {
        boolean isActiveInputColumn = (columnConfig.isFinalSelect() && (CollectionUtils.isEmpty(featureSet) || featureSet
                .contains(columnConfig.getColumnNum())));
        // active input and active target
        return isActiveInputColumn || columnConfig.isTarget();
    }
}
