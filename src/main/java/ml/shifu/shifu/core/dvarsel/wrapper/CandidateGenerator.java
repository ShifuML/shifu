/*
 * Copyright [2013-2015] PayPal Software Foundation
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

import ml.shifu.shifu.core.dvarsel.CandidatePerf;
import ml.shifu.shifu.core.dvarsel.CandidatePopulation;
import ml.shifu.shifu.core.dvarsel.CandidateSeed;
import ml.shifu.shifu.core.dvarsel.VarSelWorkerResult;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.collections.ListUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.Map.Entry;

public class CandidateGenerator {
    private static final Logger LOG = LoggerFactory.getLogger(CandidateGenerator.class);

    public static final String WORKER_SAMPLE_RATE = "worker_sample_rate";
    public static final String POPULATION_MULTIPLY_CNT = "population_multiply_cnt";
    public static final String POPULATION_LIVE_SIZE = "population_live_size";
    public static final String EXPECT_VARIABLE_CNT = "expect_variable_cnt";
    public static final String HYBRID_PERCENT = "hybrid_percent";
    public static final String MUTATION_PERCENT = "mutation_percent";

    private final int iteratorSeedCount;
    private final int expectVariableCount;
    private final int expectIterationCount;

    private final List<Integer> variables;

    private int inheritPercent;
    private int crossPercent;
    private int seedId = 1;

    private Random rd = new Random(System.currentTimeMillis());

    public CandidateGenerator(Map<String, Object> params, List<Integer> variables) {
        this.expectIterationCount = (Integer) params.get(POPULATION_MULTIPLY_CNT);
        this.iteratorSeedCount = (Integer) params.get(POPULATION_LIVE_SIZE);
        if (this.iteratorSeedCount < 1) {
            LOG.error("Iterator seed count should be larger than 1.");
            throw new ShifuException(ShifuErrorCode.ERROR_SHIFU_CONFIG, "Iterator seed count should be larger than 1.");
        }

        this.expectVariableCount = (Integer) params.get(EXPECT_VARIABLE_CNT);
        if (this.expectVariableCount < 1) {
            LOG.error("Expect variable count should be larger than 1.");
            throw new ShifuException(ShifuErrorCode.ERROR_SHIFU_CONFIG, "Expect variable count should be larger than 1.");
        }

        this.variables = variables;

        this.crossPercent = (Integer) params.get(HYBRID_PERCENT);
        if (this.crossPercent < 0 || this.crossPercent > 100) {
            LOG.error("Cross percent should be larger than 0 and less than 100");
            throw new ShifuException(ShifuErrorCode.ERROR_SHIFU_CONFIG, "Cross percent should be larger than 0 and less than 100.");
        }

        int mutationPercent = (Integer) params.get(MUTATION_PERCENT);
        if (mutationPercent < 0 || mutationPercent > 100) {
            LOG.error("Mutation percent should be larger than 0 and less 100");
            throw new ShifuException(ShifuErrorCode.ERROR_SHIFU_CONFIG, "Mutation percent should be larger than 0 and less than 100.");
        }

        this.inheritPercent = 100 - crossPercent - mutationPercent;
        if (this.inheritPercent < 0 || this.inheritPercent > 100) {
            LOG.error("Cross percent add mutation percent should be larger than 0 and less than 100");
            throw new ShifuException(ShifuErrorCode.ERROR_SHIFU_CONFIG, "Cross percent add mutation percent should be larger than 0 and less than 100.");
        }
    }

    public int getExpectIterationCount() {
        return expectIterationCount;
    }

    public CandidatePopulation initSeeds() {
        CandidatePopulation seeds = new CandidatePopulation(iteratorSeedCount);
        for (int seedIndex = 0; seedIndex < iteratorSeedCount; seedIndex++) {
            List<Integer> variableList = new ArrayList<Integer>(expectVariableCount);
            for (int varIndex = 0; varIndex < expectVariableCount; varIndex++) {
                variableList.add(randomVariable(rd, variableList));
            }
            seeds.addCandidateSeed(new CandidateSeed(this.genSeedId(), variableList));
        }
        return seeds;
    }

    private Integer randomVariable(Random random, List<Integer> variableList) {
        Integer variable = variables.get((int)(random.nextDouble() * (variables.size() - 1)));
        if (variableList.contains(variable)) {
            variable = randomVariable(random, variableList);
        }
        return variable;
    }

    public CandidatePopulation nextGeneration(Iterable<VarSelWorkerResult> workerResults, CandidatePopulation seeds) {
        if ( hasNoneResults(workerResults) ) {
            return seeds;
        }

        List<CandidatePerf> perfs = getIndividual(workerResults);
        Collections.sort(perfs, new Comparator<CandidatePerf>() {
            @Override
            public int compare(CandidatePerf cpa, CandidatePerf cpb) {
                return cpa.getVerror() < cpb.getVerror() ? -1 : 1;
            }
        });

        for (int i = 0; i < 5; i++) {
            LOG.info("The error rate is {}, the best-{} seed: {} ", perfs.get(i).getVerror(), i, seeds.getSeedById(perfs.get(i).getId()));
        }
        LOG.info("Worst seed: {}", perfs.get(perfs.size() - 1).toString());

        List<CandidatePerf> bestPerfs = perfs.subList(0, getLastBestIndex(perfs) + 1);
        List<CandidatePerf> ordinaryPerfs = perfs.subList(getLastBestIndex(perfs) + 1, getFistWorstIndex(perfs));
        List<CandidatePerf> worstPerfs = perfs.subList(getFistWorstIndex(perfs), perfs.size());

        List<CandidateSeed> bestSeeds = filter(seeds, bestPerfs);
        List<CandidateSeed> ordinarySeeds = filter(seeds, ordinaryPerfs);
        List<CandidateSeed> worstSeeds = filter(seeds, worstPerfs);

        CandidatePopulation result = new CandidatePopulation(iteratorSeedCount);
        result.addCandidateSeedList(inherit(bestSeeds));
        result.addCandidateSeedList(hybrid(ordinarySeeds));
        result.addCandidateSeedList(mutate(worstSeeds));
        LOG.debug("new generation:" + result);
        return result;
    }

    private boolean hasNoneResults(Iterable<VarSelWorkerResult> workerResults) {
        for ( VarSelWorkerResult result : workerResults ) {
            if ( result.getSeedPerfList().size() > 0 ) {
                return false;
            }
        }
        return true;
    }

    private int getLastBestIndex(List<CandidatePerf> perfs) {
        return perfs.size() * inheritPercent / 100;
    }

    private int getFistWorstIndex(List<CandidatePerf> perfs) {
        return perfs.size() * (100 - crossPercent) / 100;
    }

    private List<CandidateSeed> inherit(List<CandidateSeed> bestSeeds) {
        return bestSeeds;
    }

    private List<CandidateSeed> hybrid(List<CandidateSeed> ordinarySeedList) {
        List<CandidateSeed> result = new ArrayList<CandidateSeed>(ordinarySeedList.size());

        int childCnt = 0;
        while ( childCnt < ordinarySeedList.size() ) {
            CandidateSeed father = ordinarySeedList.get(rd.nextInt(ordinarySeedList.size()));
            CandidateSeed mather = ordinarySeedList.get(rd.nextInt(ordinarySeedList.size()));

            CandidateSeed child = hybrid(father, mather);
            if ( child != null ) {
                result.add(child);
                childCnt++;
            }
        }

        return result;
    }

    private CandidateSeed hybrid(CandidateSeed father, CandidateSeed mather) {
        Set<Integer> geneSet = new HashSet<Integer>();
        geneSet.addAll(father.getColumnIdList());
        geneSet.addAll(mather.getColumnIdList());

        List<Integer> wholeGeneList = new ArrayList<Integer>(geneSet);

        List<Integer> indexList = new ArrayList<Integer>(wholeGeneList.size());
        for ( int i = 0; i < wholeGeneList.size(); i ++ ) {
            indexList.add(i);
        }
        Collections.shuffle(indexList);

        List<Integer> childGeneList = new ArrayList<Integer>(father.getColumnIdList().size());
        for ( int i = 0; i < father.getColumnIdList().size(); i ++ ) {
            childGeneList.add(wholeGeneList.get(indexList.get(i)));
        }

        return new CandidateSeed(this.genSeedId(), childGeneList);
    }

    private List<CandidateSeed> mutate(List<CandidateSeed> worstSeeds) {
        List<CandidateSeed> result = new ArrayList<CandidateSeed>(worstSeeds.size());
        for ( CandidateSeed seed : worstSeeds ) {
            result.add(doMutation(seed));
        }
        return result;
    }

    @SuppressWarnings("unchecked")
    private CandidateSeed doMutation(CandidateSeed seed) {
        List<Integer> geneList = new ArrayList<Integer>();

        List<Integer> unselectedGeneList = ListUtils.subtract(variables, seed.getColumnIdList());
        Collections.shuffle(unselectedGeneList);

        int replaceCnt = 0;
        for ( int i = 0; i < seed.getColumnIdList().size(); i ++ ) {
            if ( rd.nextDouble() < 0.05 ) {
                replaceCnt ++;
            } else {
                geneList.add(seed.getColumnIdList().get(i));
            }
        }

        if ( replaceCnt > 0 ) {
            geneList.addAll(unselectedGeneList.subList(0, replaceCnt));
        }

        return new CandidateSeed(this.genSeedId(), geneList);
    }

    private List<CandidateSeed> filter(CandidatePopulation seeds, final List<CandidatePerf> perfList) {
        List<CandidateSeed> result = new ArrayList<CandidateSeed>(perfList.size());
        for ( CandidatePerf perf : perfList ) {
            result.add(seeds.getSeedById(perf.getId()));
        }
        return result;
    }

    private List<CandidatePerf> getIndividual(Iterable<VarSelWorkerResult> workerResults) {
        Map<Integer, List<Double>> errorMap = new HashMap<Integer, List<Double>>();
        for (VarSelWorkerResult workerResult : workerResults) {
            List<CandidatePerf> seedPerfList = workerResult.getSeedPerfList();

            for (CandidatePerf perf : seedPerfList) {
                if (!errorMap.containsKey(perf.getId())) {
                    errorMap.put(perf.getId(), new ArrayList<Double>());
                }
                errorMap.get(perf.getId()).add(perf.getVerror());
            }
        }

        List<CandidatePerf> perfs = new ArrayList<CandidatePerf>(errorMap.size());
        for (Entry<Integer, List<Double>> entry : errorMap.entrySet()) {
            double vError = mean(entry.getValue());
            perfs.add(new CandidatePerf(entry.getKey(), vError));
        }
        return perfs;
    }

    private double mean(List<Double> values) {
        if ( CollectionUtils.isEmpty(values)) {
            return 999.0;
        }

        double result = 0;
        for (Double value : values) {
            result += value;
        }
        return result / values.size();
    }

    private int genSeedId() {
        return (seedId ++);
    }
}
