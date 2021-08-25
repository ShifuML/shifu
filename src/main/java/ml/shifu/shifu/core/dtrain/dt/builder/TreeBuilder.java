/*
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
package ml.shifu.shifu.core.dtrain.dt.builder;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.dtrain.dt.TreeNode;

import java.util.List;

public abstract class TreeBuilder {

    protected ModelConfig modelConfig;
    protected List<ColumnConfig> columnConfigList;

    /**
     * Trees for fail over or continuous model training, this is recovered from hdfs and no need back up
     */
    protected List<TreeNode> recoverTrees;

    public TreeBuilder(ModelConfig modelConfig, List<ColumnConfig> columnConfigList) {
        this.modelConfig = modelConfig;
        this.columnConfigList = columnConfigList;
    }
}
