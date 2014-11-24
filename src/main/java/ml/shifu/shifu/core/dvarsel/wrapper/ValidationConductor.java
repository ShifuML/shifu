package ml.shifu.shifu.core.dvarsel.wrapper;
/*
 * Licensed to the Apache Software Foundation (ASF) under one
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
import ml.shifu.shifu.core.dvarsel.util.NNValidator;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;

import java.util.List;
import java.util.Set;

/**
 * Created on 11/24/2014.
 */
public class ValidationConductor {

    private ModelConfig modelConfig;
    private List<ColumnConfig> columnConfigList;
    private Set<Integer>  workingColumnSet;
    private TrainingDataSet trainingDataSet;

    public ValidationConductor(ModelConfig modelConfig,
                               List<ColumnConfig> columnConfigList,
                               Set<Integer> workingColumnSet,
                               TrainingDataSet trainingDataSet) {
        this.modelConfig = modelConfig;
        this.columnConfigList = columnConfigList;
        this.workingColumnSet = workingColumnSet;
        this.trainingDataSet = trainingDataSet;
    }


    public double runValidate() {
        //1. prepare training data
        MLDataSet trainingData = new BasicMLDataSet();
        MLDataSet testingData = new BasicMLDataSet();

        this.trainingDataSet.generateValidateData(this.workingColumnSet,
                this.modelConfig.getCrossValidationRate(),
                trainingData,
                testingData);

        //2. build validator
        NNValidator validator = new NNValidator(modelConfig, columnConfigList, workingColumnSet, trainingData, testingData);

        //3. train and get validation error
        return validator.validate();
    }

}
