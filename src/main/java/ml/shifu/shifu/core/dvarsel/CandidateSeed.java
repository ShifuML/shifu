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

    public boolean sameAs(CandidateSeed worseSeed) {
        if (columnIdList.size() != worseSeed.getColumnIdList().size()) {
            return false;
        }
        for (Integer columnId : columnIdList) {
            if (!worseSeed.getColumnIdList().contains(columnId)) {
                return false;
            }
        }
        return true;
    }

    @Override
    public int hashCode() {
        return id;
    }

    @Override
    public boolean equals(Object obj) {
        if ( obj == null || !(obj instanceof CandidateSeed)) {
            return false;
        }

        CandidateSeed cs = (CandidateSeed) obj;
        if ( cs == this ) {
            return true;
        }

        return (id == cs.getId());
    }

    @Override
    public String toString() {
        return "CandidateSeed{"
                + "id=" + id
                + ", columnIdList=" + columnIdList
                + "}";
    }
}
