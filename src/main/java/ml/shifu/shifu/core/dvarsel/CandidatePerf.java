package ml.shifu.shifu.core.dvarsel;

/**
 * Created by zhanhu on 2015/3/25.
 */
public class CandidatePerf {

    private int id;
    private double verror;

    public CandidatePerf(int id, double verror) {
        this.id = id;
        this.verror = verror;
    }

    public int getId() {
        return this.id;
    }

    public double getVerror() {
        return this.verror;
    }
}
