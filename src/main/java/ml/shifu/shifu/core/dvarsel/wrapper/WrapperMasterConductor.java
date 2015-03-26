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
package ml.shifu.shifu.core.dvarsel.wrapper;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.dvarsel.AbstractMasterConductor;
import ml.shifu.shifu.core.dvarsel.CandidateSeed;
import ml.shifu.shifu.core.dvarsel.CandidateSeeds;
import ml.shifu.shifu.core.dvarsel.VarSelWorkerResult;
import ml.shifu.shifu.util.CommonUtils;

import java.util.*;

/**
 * Created on 11/24/2014.
 */
public class WrapperMasterConductor extends AbstractMasterConductor {
    private CandidateGenerator candidateGenerator;
    private CandidateSeeds seeds;

    private int iterationCount = 0;

    public WrapperMasterConductor(ModelConfig modelConfig, List<ColumnConfig> columnConfigList) {
        super(modelConfig, columnConfigList);

        List<Integer> variables = new ArrayList<Integer>(columnConfigList.size());
        for (ColumnConfig columnConfig : columnConfigList) {
            if ( CommonUtils.isGoodCandidate(columnConfig) ) {
                variables.add(columnConfig.getColumnNum());
            }
        }
        this.candidateGenerator = new CandidateGenerator(this.modelConfig.getVarSelect().getParams(), variables);
        this.seeds = candidateGenerator.initSeeds();
    }

    @Override
    public int getEstimateIterationCnt() {
        return (candidateGenerator.getExpectIterationCount() < iterationCount ?
                0 :
                candidateGenerator.getExpectIterationCount() - iterationCount);
    }

    @Override
    public boolean isToStop() {
        return (iterationCount >= candidateGenerator.getExpectIterationCount());
    }

    @Override
    public List<CandidateSeed> getNextWorkingSet() {
        return seeds.getCandidateSeeds();
    }

    @Override
    public void consumeWorkerResults(Iterable<VarSelWorkerResult> workerResults) {
        this.iterationCount++;
        this.seeds = candidateGenerator.nextGeneration(workerResults, this.seeds);
    }
}
