package ml.shifu.shifu.core.dvarsel.wrapper;

import ml.shifu.shifu.core.dvarsel.CandidatePerf;
import ml.shifu.shifu.core.dvarsel.CandidateSeed;
import ml.shifu.shifu.core.dvarsel.CandidateSeeds;
import ml.shifu.shifu.core.dvarsel.VarSelWorkerResult;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.collections.Predicate;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

public class CandidateGenerator {
    private static final Logger logger = LoggerFactory.getLogger(CandidateGenerator.class);

    public static final String WORKER_SAMPLE_RATE = "worker_sample_rate";
    public static final String EXPECT_ITERATION_COUNT = "expect_iteration_count";
    public static final String ITERATION_SEED_COUNT = "iteration_seed_count";
    public static final String EXPECT_VARIABLE_COUNT = "expect_variable_count";
    public static final String CROSS_PERCENT = "cross_percent";
    public static final String MUTATION_PERCENT = "mutation_percent";

    private final int iteratorSeedCount;
    private final int expectVariableCount;
    private final int expectIterationCount;

    private final List<Integer> variables;

    private int inheritPercent;
    private int crossPercent;

    public CandidateGenerator(Map<String, Object> params, List<Integer> variables) {
        this.expectIterationCount = (Integer) params.get(EXPECT_ITERATION_COUNT);
        this.iteratorSeedCount = (Integer) params.get(ITERATION_SEED_COUNT);
        if (this.iteratorSeedCount < 1) {
            logger.error("Iterator seed count should be larger than 1.");
            throw new ShifuException(ShifuErrorCode.ERROR_SHIFU_CONFIG, "Iterator seed count should be larger than 1.");
        }

        this.expectVariableCount = (Integer) params.get(EXPECT_VARIABLE_COUNT);
        if (this.expectVariableCount < 1) {
            logger.error("Expect variable count should be larger than 1.");
            throw new ShifuException(ShifuErrorCode.ERROR_SHIFU_CONFIG, "Expect variable count should be larger than 1.");
        }

        this.variables = variables;

        this.crossPercent = (Integer) params.get(CROSS_PERCENT);
        if (this.crossPercent < 0 || this.crossPercent > 100) {
            logger.error("Cross percent should be larger than 0 and less than 100");
            throw new ShifuException(ShifuErrorCode.ERROR_SHIFU_CONFIG, "Cross percent should be larger than 0 and less than 100.");
        }

        int mutationPercent = (Integer) params.get(MUTATION_PERCENT);
        if (mutationPercent < 0 || mutationPercent > 100) {
            logger.error("Mutation percent should be larger than 0 and less 100");
            throw new ShifuException(ShifuErrorCode.ERROR_SHIFU_CONFIG, "Mutation percent should be larger than 0 and less than 100.");
        }

        this.inheritPercent = 100 - crossPercent - mutationPercent;
        if (this.inheritPercent < 0 || this.inheritPercent > 100) {
            logger.error("Cross percent add mutation percent should be larger than 0 and less than 100");
            throw new ShifuException(ShifuErrorCode.ERROR_SHIFU_CONFIG, "Cross percent add mutation percent should be larger than 0 and less than 100.");
        }
    }

    public int getExpectIterationCount() {
        return expectIterationCount;
    }

    /**
     * Init random seeds
     */
    public CandidateSeeds initSeeds() {
        CandidateSeeds seeds = new CandidateSeeds(iteratorSeedCount);
        Random seedRandom = new Random();
        for (int seedIndex = 0; seedIndex < iteratorSeedCount; seedIndex++) {
            List<Integer> variableList = new ArrayList<Integer>(expectVariableCount);
            for (int varIndex = 0; varIndex < expectVariableCount; varIndex++) {
                variableList.add(randomVariable(seedRandom, variableList));
            }
            seeds.addCandidateSeed(seedIndex, variableList);
        }
        return seeds;
    }

    private Integer randomVariable(Random seedRandom, List<Integer> variableList) {
        Integer variable = variables.get(Double.valueOf(seedRandom.nextDouble() * (variables.size() - 1)).intValue());
        if (variableList.contains(variable)) {
            variable = randomVariable(seedRandom, variableList);
        }
        return variable;
    }

    public CandidateSeeds nextGeneration(Iterable<VarSelWorkerResult> workerResults, CandidateSeeds seeds) {
        List<CandidatePerf> perfs = getIndividual(workerResults);

        Collections.sort(perfs, new Comparator<CandidatePerf>() {
            @Override
            public int compare(CandidatePerf o1, CandidatePerf o2) {
                return o1.getVerror() < o2.getVerror() ? -1 : 1;
            }
        });
        List<CandidatePerf> bestPerfs = perfs.subList(0, perfs.size() * inheritPercent / 100);
        List<CandidatePerf> worstPerfs = perfs.subList(perfs.size() * inheritPercent / 100 + 1, perfs.size() * inheritPercent / 100 + 1 + perfs.size() * crossPercent / 100);
        List<CandidatePerf> ordinaryPerfs = perfs.subList(perfs.size() * crossPercent / 100 + 1, perfs.size() - 1);

        List<CandidateSeed> bestSeeds = filter(seeds.getCandidateSeeds(), bestPerfs);
        List<CandidateSeed> worstSeeds = filter(seeds.getCandidateSeeds(), worstPerfs);
        List<CandidateSeed> ordinarySeeds = filter(seeds.getCandidateSeeds(), ordinaryPerfs);

        CandidateSeeds result = new CandidateSeeds(iteratorSeedCount);
        result.addCandidateSeeds(inherit(bestSeeds));
        System.out.println("result:" + result);
        result.addCandidateSeeds(cross(worstSeeds, result));
        System.out.println("result:" + result);
        result.addCandidateSeeds(mutate(ordinarySeeds, result));
        System.out.println("result:" + result);
        return result;
    }

    private List<CandidateSeed> inherit(List<CandidateSeed> bestSeeds) {
        return bestSeeds;
    }

    private List<CandidateSeed> cross(List<CandidateSeed> ordinarySeeds, CandidateSeeds allSeeds) {
        List<CandidateSeed> result = new ArrayList<CandidateSeed>(ordinarySeeds.size());
        for (int index = 0; index < ordinarySeeds.size() / 2; index++) {
            CandidateSeed first = ordinarySeeds.get(index);
            CandidateSeed second = ordinarySeeds.get(ordinarySeeds.size() - index - 1);
            while (allSeeds.contains(first) || allSeeds.contains(second)) {
                crossTwoSeed(first, second);
            }
            result.add(first);
            result.add(second);
        }
        return result;
    }

    private void crossTwoSeed(CandidateSeed first, CandidateSeed second) {
        Random splitIndexRandom = new Random();
        int splitIndex = Double.valueOf(splitIndexRandom.nextDouble() * (first.getColumnIdList().size() - 2) + 1).intValue();
        for (int i = 0; i < splitIndex; i++) {
            int id = first.getColumnIdList().get(i);
            first.getColumnIdList().set(i, second.getColumnIdList().get(i));
            second.getColumnIdList().set(i, id);
        }
    }

    private List<CandidateSeed> mutate(List<CandidateSeed> worstSeeds, CandidateSeeds seeds) {
        List<CandidateSeed> result = new ArrayList<CandidateSeed>(worstSeeds.size());
        Random candidateRandom = new Random();
        for (CandidateSeed seed : worstSeeds) {
            seed = _mutate(candidateRandom, seed, seeds);
            result.add(seed);
        }
        return result;
    }

    private CandidateSeed _mutate(Random candidateRandom, CandidateSeed worseSeed, CandidateSeeds allSeeds) {
        int index = Double.valueOf(candidateRandom.nextDouble() * (expectVariableCount - 1)).intValue();

        CandidateSeed result = new CandidateSeed(worseSeed.getId(), new ArrayList<Integer>(worseSeed.getColumnIdList()));
        result.getColumnIdList().set(index, randomVariable(candidateRandom, worseSeed.getColumnIdList()));
        if (allSeeds.contains(result)) {
            result = _mutate(candidateRandom, worseSeed, allSeeds);
        }
        return result;
    }

    private List<CandidateSeed> filter(List<CandidateSeed> seeds, final List<CandidatePerf> perfs) {
        List<CandidateSeed> result = new ArrayList<CandidateSeed>(seeds);
        CollectionUtils.filter(result, new Predicate() {
            @Override
            public boolean evaluate(Object object) {
                CandidateSeed seed = (CandidateSeed) object;
                for (CandidatePerf perf : perfs) {
                    if (perf.getId() == seed.getId()) {
                        return true;
                    }
                }
                return false;
            }
        });
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
        for (Integer id : errorMap.keySet()) {
            double vError = mean(errorMap.get(id));
            perfs.add(new CandidatePerf(id, vError));
        }
        return perfs;
    }

    private double mean(List<Double> values) {
        if (values == null || values.isEmpty()) {
            return 0;
        }

        double result = 0;
        for (Double value : values) {
            result += value;
        }
        return result / values.size();
    }
}
