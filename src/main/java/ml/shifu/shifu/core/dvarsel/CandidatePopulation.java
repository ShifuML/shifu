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
package ml.shifu.shifu.core.dvarsel;

import java.util.*;

/**
 * Created by Karl Yang on 2015/3/25
 */
public class CandidatePopulation {
    private Map<Integer, CandidateSeed> seedMapping;
    private List<CandidateSeed> seedList;

    public CandidatePopulation(int size) {
        this.seedList = new ArrayList<CandidateSeed>(size);
        this.seedMapping = new HashMap<Integer, CandidateSeed>(size * 4/ 3);
    }

    public void addCandidateSeed(CandidateSeed candidateSeed) {
        this.seedList.add(candidateSeed);
        this.seedMapping.put(candidateSeed.getId(), candidateSeed);
    }

    public void addCandidateSeedList(List<CandidateSeed> seedList) {
        this.seedList.addAll(seedList);
        for (CandidateSeed candidateSeed : seedList) {
            this.seedMapping.put(candidateSeed.getId(), candidateSeed);
        }
    }

    public List<CandidateSeed> getSeedList() {
        return Collections.unmodifiableList(seedList);
    }

    public boolean contains(CandidateSeed seed) {
        return this.seedMapping.containsKey(seed.getId());
    }

    @Override
    public String toString() {
        return seedList.toString();
    }

    public CandidateSeed getSeedById(int id) {
        return seedMapping.get(id);
    }
}
