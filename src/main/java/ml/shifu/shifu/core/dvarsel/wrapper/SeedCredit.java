package ml.shifu.shifu.core.dvarsel.wrapper;

import ml.shifu.shifu.core.dvarsel.CandidateSeed;

/**
 * Created by zhanhu on 2015/4/14.
 */
public class SeedCredit {
    private int credit;
    private CandidateSeed seed;

    public SeedCredit(int credit, CandidateSeed seed) {
        this.credit = credit;
        this.seed = seed;
    }

    public int getCredit() {
        return credit;
    }

    public CandidateSeed getSeed() {
        return seed;
    }
}
