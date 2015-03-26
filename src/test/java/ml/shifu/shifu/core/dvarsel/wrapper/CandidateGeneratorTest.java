package ml.shifu.shifu.core.dvarsel.wrapper;

import ml.shifu.shifu.core.dvarsel.CandidatePerf;
import ml.shifu.shifu.core.dvarsel.CandidateSeed;
import ml.shifu.shifu.core.dvarsel.CandidateSeeds;
import ml.shifu.shifu.core.dvarsel.VarSelWorkerResult;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.util.*;

import static org.testng.Assert.*;

public class CandidateGeneratorTest {
    private CandidateGenerator generator;

    private static final int EXPECT_ITERATION_COUNT = 4;
    private static final int ITERATION_SEED_COUNT = 10;
    private static final int EXPECT_VARIABLE_COUNT = 10;
    private static final int CROSS_PERCENT = 60;
    private static final int MUTATION_PERCENT = 20;

    @BeforeClass
    public void setUpBeforeClass() throws Exception {
        Map<String, Object> params = new HashMap<String, Object>();
        params.put("EXPECT_ITERATION_COUNT", EXPECT_ITERATION_COUNT);
        params.put("ITERATION_SEED_COUNT", ITERATION_SEED_COUNT);
        params.put("EXPECT_VARIABLE_COUNT", EXPECT_VARIABLE_COUNT);
        params.put("CROSS_PERCENT", CROSS_PERCENT);
        params.put("MUTATION_PERCENT", MUTATION_PERCENT);

        List<Integer> variables = new ArrayList<Integer>(100);
        for (int i = 0; i < 100; i++) {
            variables.add(i);
        }
        generator = new CandidateGenerator(params, variables);
    }

    @Test
    public void testInitSeeds() throws Exception {
        CandidateSeeds seeds = generator.initSeeds();
        System.out.println(seeds);
    }

    @Test
    public void testNextGeneration() throws Exception {
        CandidateSeeds seed = generator.initSeeds();
        System.out.println(seed);

        Random random = new Random();

        for (int i = 0; i < EXPECT_ITERATION_COUNT - 1; i++) {
            seed = generateNext(seed, random);
            System.out.println(seed);
        }
    }

    private CandidateSeeds generateNext(CandidateSeeds seed, Random random) {
        List<VarSelWorkerResult> workerResults = new ArrayList<VarSelWorkerResult>();
        workerResults.add(new VarSelWorkerResult(getCandidatePerfs(random, seed)));
        workerResults.add(new VarSelWorkerResult(getCandidatePerfs(random, seed)));
        workerResults.add(new VarSelWorkerResult(getCandidatePerfs(random, seed)));

        final List<VarSelWorkerResult> workerResultList = Collections.unmodifiableList(workerResults);
        Iterable<VarSelWorkerResult> results = new Iterable<VarSelWorkerResult>() {
            @Override
            public Iterator<VarSelWorkerResult> iterator() {
                return workerResultList.iterator();
            }
        };
        return generator.nextGeneration(results, seed);
    }

    private List<CandidatePerf> getCandidatePerfs(Random random, CandidateSeeds seed) {
        List<CandidatePerf> seedPerfList = new ArrayList<CandidatePerf>();
        for (CandidateSeed s : seed.getCandidateSeeds()) {
            seedPerfList.add(new CandidatePerf(s.getId(), random.nextDouble()));
        }
        return seedPerfList;
    }
}