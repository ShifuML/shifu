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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.dvarsel.AbstractMasterConductor;
import ml.shifu.shifu.core.dvarsel.CandidatePopulation;
import ml.shifu.shifu.core.dvarsel.CandidateSeed;
import ml.shifu.shifu.core.dvarsel.VarSelWorkerResult;
import ml.shifu.shifu.util.CommonUtils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created on 11/24/2014.
 */
public class WrapperMasterConductor extends AbstractMasterConductor {

    private static final Logger LOG = LoggerFactory.getLogger(WrapperMasterConductor.class);

    private CandidateGenerator candidateGenerator;
    private CandidatePopulation seeds;

    private int iterationCount = 0;

    private final int BEST_SEED_CNT = 5;
    private final int MAX_ITERATIONS_TO_KEEP = 5;
    private final int SC_QUEUE_LEN = BEST_SEED_CNT * MAX_ITERATIONS_TO_KEEP;

    private SeedCredit[] seedCreditQueue;
    private int scCnt;

    public WrapperMasterConductor(ModelConfig modelConfig, List<ColumnConfig> columnConfigList) {
        super(modelConfig, columnConfigList);

        List<Integer> variables = new ArrayList<Integer>(columnConfigList.size());
        boolean hasCandidates = CommonUtils.hasCandidateColumns(columnConfigList);
        for (ColumnConfig columnConfig : columnConfigList) {
            if ( CommonUtils.isGoodCandidate(columnConfig, hasCandidates) ) {
                variables.add(columnConfig.getColumnNum());
            }
        }
        this.candidateGenerator = new CandidateGenerator(this.modelConfig.getVarSelect().getParams(), variables);
        this.seeds = candidateGenerator.initSeeds();

        this.seedCreditQueue = new SeedCredit[SC_QUEUE_LEN];
        this.scCnt = 0;
    }

    @Override
    public int getEstimateIterationCnt() {
        return (candidateGenerator.getExpectIterationCount() < iterationCount ?
                0 : candidateGenerator.getExpectIterationCount() - iterationCount);
    }

    @Override
    public boolean isToStop() {
        return (iterationCount > candidateGenerator.getExpectIterationCount());
    }

    @Override
    public List<CandidateSeed> getNextWorkingSet() {
        return seeds.getSeedList();
    }

    @Override
    public void consumeWorkerResults(Iterable<VarSelWorkerResult> workerResults) {
        this.iterationCount++;
        this.seeds = candidateGenerator.nextGeneration(workerResults, this.seeds);

        for ( int i = 0; i < BEST_SEED_CNT; i ++ ) {
            seedCreditQueue[(scCnt ++) % this.SC_QUEUE_LEN] =
                    new SeedCredit(BEST_SEED_CNT - i, this.seeds.getSeedById(i));
        }
    }

    @Override
    public CandidateSeed voteBestSeed() {
        Map<CandidateSeed, Integer> seedCreditMap = new HashMap<CandidateSeed, Integer>();
        for ( int i = 0; i < seedCreditQueue.length; i ++ ) {
            SeedCredit seedCredit = seedCreditQueue[i];
            if ( seedCredit != null ) {
                CandidateSeed candidateSeed = seedCredit.getSeed();
                if ( !seedCreditMap.containsKey(candidateSeed) ) {
                    seedCreditMap.put(candidateSeed, Integer.valueOf(0));
                }

                seedCreditMap.put(seedCredit.getSeed(),
                        Integer.valueOf(seedCreditMap.get(candidateSeed) + seedCredit.getCredit()));
            }
        }

        CandidateSeed bestSeed = null;
        int maxCredit = Integer.MIN_VALUE;

        Iterator<Map.Entry<CandidateSeed, Integer>> iterator = seedCreditMap.entrySet().iterator();
        while ( iterator.hasNext() ) {
            Map.Entry<CandidateSeed, Integer> entry = iterator.next();
            if ( entry.getValue() > maxCredit ) {
                maxCredit = entry.getValue();
                bestSeed = entry.getKey();
            }
        }

        LOG.info("With max credit - {}, the best candidate is {}", maxCredit, bestSeed);
        return bestSeed;
    }
}
