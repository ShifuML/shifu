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
