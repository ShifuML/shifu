package ml.shifu.shifu.core.dvarsel;

import java.util.List;

/**
 * Created by zhanhu on 2015/3/25.
 */
public class CandidateSeed {
    private int id;
    private List<Integer> columnIdList;

    public CandidateSeed(int id, List<Integer> columnIdList) {
        this.id = id;
        this.columnIdList = columnIdList;
    }

    public int getId() {
        return this.id;
    }

    public List<Integer> getColumnIdList() {
        return this.columnIdList;
    }
}
