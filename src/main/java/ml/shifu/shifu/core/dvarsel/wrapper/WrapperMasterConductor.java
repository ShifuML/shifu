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
import ml.shifu.shifu.core.dvarsel.VarSelWorkerResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Created on 11/24/2014.
 */
public class WrapperMasterConductor extends AbstractMasterConductor {

    private static final Logger LOG = LoggerFactory.getLogger(WrapperMasterConductor.class);

    private int expectVarCount;
    private Set<Integer> workingSet;

    public WrapperMasterConductor(ModelConfig modelConfig, List<ColumnConfig> columnConfigList) {
        super(modelConfig, columnConfigList);

        this.workingSet = new HashSet<Integer>();
        this.expectVarCount = this.modelConfig.getVarSelectFilterNum();
        for ( ColumnConfig columnConfig : columnConfigList ) {
            if ( columnConfig.isCandidate() && columnConfig.isForceSelect() ) {
                workingSet.add(columnConfig.getColumnNum());
            }
        }

        LOG.info("Expected variable count is - {}, base working set size is - {}", expectVarCount, workingSet.size());
    }

    @Override
    public int getEstimateIterationCnt() {
        return expectVarCount - workingSet.size();
    }

    @Override
    public boolean isToStop() {
        return (workingSet.size() == expectVarCount);
    }

    @Override
    public List<Integer> getNextWorkingSet() {
        return new ArrayList<Integer>(workingSet);
    }

    @Override
    public void consumeWorkerResults(Iterable<VarSelWorkerResult> workerResults) {
        int[] voteStats = new int[columnConfigList.size() + 1];

        for (VarSelWorkerResult workerResult : workerResults ) {
            for ( Integer columnId : workerResult.getColumnIdList() ) {
                voteStats[columnId + 1]++;
            }
        }

        // get max voted column id
        int maxVotedColumnId = -1;
        int maxVoteCount = Integer.MIN_VALUE;
        for ( int i = 0; i < voteStats.length; i ++ ) {
            if ( voteStats[i] > maxVoteCount ) {
                maxVoteCount = voteStats[i];
                maxVotedColumnId = i;
            }
        }

        LOG.info("Column - {} get most votes - {}", (maxVotedColumnId - 1), maxVoteCount);
        // no voted columnId found
        if ( maxVotedColumnId > 0 ) {
            workingSet.add(maxVotedColumnId - 1);
        }
    }

}
