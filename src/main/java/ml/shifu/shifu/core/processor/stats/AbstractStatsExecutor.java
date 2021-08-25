/*
 * Copyright [2013-2015] PayPal Software Foundation
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
package ml.shifu.shifu.core.processor.stats;

import java.util.List;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.processor.BasicModelProcessor;

/**
 * Created by zhanhu on 6/30/16.
 */
public abstract class AbstractStatsExecutor {

    protected BasicModelProcessor processor;
    protected ModelConfig modelConfig;
    protected List<ColumnConfig> columnConfigList;

    private int mtlIndex = -1;

    public AbstractStatsExecutor(BasicModelProcessor processor, ModelConfig modelConfig,
            List<ColumnConfig> columnConfigList) {
        this.processor = processor;
        this.modelConfig = modelConfig;
        this.columnConfigList = columnConfigList;
    }

    public abstract boolean doStats() throws Exception;

    /**
     * @return the mtlIndex
     */
    public int getMtlIndex() {
        return mtlIndex;
    }

    /**
     * @param mtlIndex
     *            the mtlIndex to set
     */
    public void setMtlIndex(int mtlIndex) {
        this.mtlIndex = mtlIndex;
    }

}
