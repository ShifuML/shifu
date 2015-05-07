/**
 * Copyright [2012-2014] eBay Software Foundation
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
package ml.shifu.shifu.core.dvarsel;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.dvarsel.VarSelWorkerResult;

import java.util.List;

/**
 * Created on 11/24/2014.
 */
public abstract class AbstractMasterConductor {
    protected ModelConfig modelConfig;
    protected List<ColumnConfig> columnConfigList;

    public AbstractMasterConductor(ModelConfig modelConfig, List<ColumnConfig> columnConfigList) {
        this.modelConfig = modelConfig;
        this.columnConfigList = columnConfigList;
    }

    public abstract int getEstimateIterationCnt();
    public abstract boolean isToStop();

    public abstract List<CandidateSeed> getNextWorkingSet();
    public abstract void consumeWorkerResults(Iterable<VarSelWorkerResult> workerResults);

    public abstract CandidateSeed voteBestSeed();
}
