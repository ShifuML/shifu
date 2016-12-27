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
package ml.shifu.shifu.core.dvarsel;
/*
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.dvarsel.dataset.TrainingDataSet;

import java.util.List;

/**
 * Created on 11/24/2014.
 */
public abstract class AbstractWorkerConductor {
    protected ModelConfig modelConfig;
    protected List<ColumnConfig> columnConfigList;
    protected boolean isInitialized;
    protected TrainingDataSet trainingDataSet;

    public AbstractWorkerConductor(ModelConfig modelConfig, List<ColumnConfig> columnConfigList) {
        this.modelConfig = modelConfig;
        this.columnConfigList = columnConfigList;
        this.isInitialized = false;
    }

    public abstract void consumeMasterResult(VarSelMasterResult masterResult);
    public abstract VarSelWorkerResult generateVarSelResult();
    public abstract VarSelWorkerResult getDefaultWorkerResult();

    public boolean isInitialized() {
        return isInitialized;
    }

    public void retainData(TrainingDataSet trainingDataSet) {
        this.trainingDataSet = trainingDataSet;
    }
}
