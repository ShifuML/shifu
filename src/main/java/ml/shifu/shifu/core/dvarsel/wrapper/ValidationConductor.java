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
package ml.shifu.shifu.core.dvarsel.wrapper;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.alg.NNTrainer;
import ml.shifu.shifu.core.dvarsel.dataset.TrainingDataSet;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;

import java.io.IOException;
import java.util.List;
import java.util.Set;

/**
 * Created on 11/24/2014.
 */
public class ValidationConductor {

    private ModelConfig modelConfig;
    @SuppressWarnings("unused")
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
                this.modelConfig.getValidSetRate(),
                trainingData,
                testingData);

        //2. build NNTrainer
        NNTrainer trainer = new NNTrainer(this.modelConfig, 1, false);
        trainer.setTrainSet(trainingData);
        trainer.setValidSet(testingData);
        trainer.disableModelPersistence();
        trainer.disableLogging();

        //3. train and get validation error
        double validateError = Double.MAX_VALUE;
        try {
            validateError = trainer.train();
        } catch ( IOException e ) {
            // Ignore the exception when nn files
            validateError = trainer.getBaseMSE();
        }
        return validateError;
    }

}
