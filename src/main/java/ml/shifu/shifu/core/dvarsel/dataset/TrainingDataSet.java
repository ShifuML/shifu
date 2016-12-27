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
package ml.shifu.shifu.core.dvarsel.dataset;

import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;

/**
 * Created on 11/24/2014.
 */
public class TrainingDataSet {

    private static Random rd = new Random(System.currentTimeMillis());

    private List<Integer> dataColumnIdList;
    private List<TrainingRecord> trainingRecords;

    public TrainingDataSet(List<Integer> dataColumnIdList) {
        this.dataColumnIdList = dataColumnIdList;
        this.trainingRecords = new ArrayList<TrainingRecord>();
    }

    public void addTrainingRecord(TrainingRecord trainingRecord) {
        if (trainingRecord != null ) {
            this.trainingRecords.add(trainingRecord);
        }
    }

    public void generateValidateData(Set<Integer> workingColumnIdSet,
                                     double validationRate,
                                     MLDataSet trainingData,
                                     MLDataSet testingData ) {
        for ( TrainingRecord trainingRecord : trainingRecords ) {
            MLDataPair pair = trainingRecord.toMLDataPair(dataColumnIdList, workingColumnIdSet);

            double seed = rd.nextDouble();
            if (seed > validationRate) {
                trainingData.add(pair);
            } else {
                testingData.add(pair);
            }
        }
    }

    public List<Integer> getDataColumnIdList() {
        return this.dataColumnIdList;
    }
}
