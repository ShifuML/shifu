package ml.shifu.shifu.core.dvarsel;

import java.util.*;

/**
 * Created by Karl Yang on 2015/3/25
 */
public class CandidateSeeds {
    private Map<Integer, CandidateSeed> candidateSeedMap;
    private List<CandidateSeed> candidateSeeds;

    public CandidateSeeds() {
        this.candidateSeeds = new ArrayList<CandidateSeed>();
        this.candidateSeedMap = new HashMap<Integer, CandidateSeed>();
    }

    public CandidateSeeds(int size) {
        this.candidateSeeds = new ArrayList<CandidateSeed>(size);
        this.candidateSeedMap = new HashMap<Integer, CandidateSeed>(size * 4/ 3);
    }

    public void addCandidateSeed(int id, List<Integer> variables) {
        CandidateSeed seed = new CandidateSeed(id, variables);
        this.candidateSeeds.add(seed);
        this.candidateSeedMap.put(seed.getId(), seed);
    }

    public void addCandidateSeed(CandidateSeed candidateSeed) {
        this.candidateSeeds.add(candidateSeed);
        this.candidateSeedMap.put(candidateSeed.getId(), candidateSeed);
    }

    public void addCandidateSeeds(List<CandidateSeed> candidateSeeds) {
        this.candidateSeeds.addAll(candidateSeeds);
        for (CandidateSeed candidateSeed : candidateSeeds) {
            this.candidateSeedMap.put(candidateSeed.getId(), candidateSeed);
        }
    }

    public List<CandidateSeed> getCandidateSeeds() {
        return Collections.unmodifiableList(candidateSeeds);
    }

    public boolean contains(CandidateSeed worseSeed) {
        for (CandidateSeed candidateSeed : candidateSeeds) {
            if (candidateSeed.sameAs(worseSeed)) {
                return true;
            }
        }
        return false;
    }

    @Override
    public String toString() {
        return candidateSeeds.toString();
    }

    public CandidateSeed getSeedById(int id) {
        return candidateSeedMap.get(id);
    }
}
