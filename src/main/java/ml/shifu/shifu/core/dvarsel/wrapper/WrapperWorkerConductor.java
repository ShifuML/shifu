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
import ml.shifu.shifu.core.dvarsel.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;

/**
 * Created on 11/24/2014.
 */
public class WrapperWorkerConductor extends AbstractWorkerConductor {
    private static final Logger LOG = LoggerFactory.getLogger(WrapperWorkerConductor.class);
    private Random rd = new Random(System.currentTimeMillis());

    private List<CandidateSeed> seedList;
    private double workerSampleRate;

    public WrapperWorkerConductor(ModelConfig modelConfig, List<ColumnConfig> columnConfigList) {
        super(modelConfig, columnConfigList);
        this.workerSampleRate = (Double) modelConfig.getVarSelect().getParams()
                .get(CandidateGenerator.WORKER_SAMPLE_RATE);
    }

    @Override
    public void consumeMasterResult(VarSelMasterResult masterResult) {
        this.seedList = masterResult.getSeedList();
    }

    @Override
    public VarSelWorkerResult generateVarSelResult() {
        List<CandidatePerf> perfList = new ArrayList<CandidatePerf>();

        for( CandidateSeed seed : seedList ) {
            if ( rd.nextDouble() < this.workerSampleRate ) {
                LOG.info("Start to test seed id = {} ", seed.getId());
                ValidationConductor validationConductor = new ValidationConductor(
                        modelConfig, columnConfigList,
                        new HashSet<Integer>(seed.getColumnIdList()), trainingDataSet);
                double validateError = validationConductor.runValidate();
                perfList.add(new CandidatePerf(seed.getId(), validateError));

                LOG.info("The validation error is {} for {}", validateError, seed.getColumnIdList());
            }
        }

        return new VarSelWorkerResult(perfList);
    }

    @Override
    public VarSelWorkerResult getDefaultWorkerResult() {
        return new VarSelWorkerResult(new ArrayList<CandidatePerf>());
    }

}
