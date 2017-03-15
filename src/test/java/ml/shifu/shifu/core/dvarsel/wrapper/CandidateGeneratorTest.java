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
import ml.shifu.shifu.core.dvarsel.CandidateSeed;
import ml.shifu.shifu.core.dvarsel.CandidatePopulation;
import ml.shifu.shifu.core.dvarsel.VarSelWorkerResult;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.util.*;

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
        params.put(CandidateGenerator.POPULATION_MULTIPLY_CNT, EXPECT_ITERATION_COUNT);
        params.put(CandidateGenerator.POPULATION_LIVE_SIZE, ITERATION_SEED_COUNT);
        params.put(CandidateGenerator.EXPECT_VARIABLE_CNT, EXPECT_VARIABLE_COUNT);
        params.put(CandidateGenerator.HYBRID_PERCENT, CROSS_PERCENT);
        params.put(CandidateGenerator.MUTATION_PERCENT, MUTATION_PERCENT);

        List<Integer> variables = new ArrayList<Integer>(100);
        for (int i = 0; i < 100; i++) {
            variables.add(i);
        }
        generator = new CandidateGenerator(params, variables);
    }

    @Test
    public void testInitSeeds() throws Exception {
        CandidatePopulation seeds = generator.initSeeds();
        System.out.println(seeds);
    }

    @Test
    public void testNextGeneration() throws Exception {
        CandidatePopulation seed = generator.initSeeds();
        System.out.println(seed);

        Random random = new Random();

        for (int i = 0; i < EXPECT_ITERATION_COUNT - 1; i++) {
            seed = generateNext(seed, random);
            System.out.println(seed);

            Assert.assertEquals(10, seed.getSeedList().size());
        }
    }

    private CandidatePopulation generateNext(CandidatePopulation seed, Random random) {
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

    private List<CandidatePerf> getCandidatePerfs(Random random, CandidatePopulation seed) {
        List<CandidatePerf> seedPerfList = new ArrayList<CandidatePerf>();
        for (CandidateSeed s : seed.getSeedList()) {
            seedPerfList.add(new CandidatePerf(s.getId(), random.nextDouble()));
        }
        return seedPerfList;
    }
}