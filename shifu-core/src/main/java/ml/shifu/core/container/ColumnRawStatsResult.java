package ml.shifu.core.container;

public class ColumnRawStatsResult extends ColumnDerivedResult {

    private Integer cntTotal = 0;
    private Integer cntValidPositive = 0;
    private Integer cntValidNegative = 0;
    private Integer cntIgnoredByTag = 0;

    private Integer cntIsNull = 0;
    private Integer cntIsNaN = 0;
    private Integer cntIsNumber = 0;
    private Integer cntUniqueValues = 0;


    public Integer getCntTotal() {
        return cntTotal;
    }

    public void setCntTotal(Integer cntTotal) {
        this.cntTotal = cntTotal;
    }


    public Integer getCntValidNegative() {
        return cntValidNegative;
    }

    public void setCntValidNegative(Integer cntValidNegative) {
        this.cntValidNegative = cntValidNegative;
    }

    public Integer getCntValidPositive() {
        return cntValidPositive;
    }

    public void setCntValidPositive(Integer cntValidPositive) {
        this.cntValidPositive = cntValidPositive;
    }

    public Integer getCntIgnoredByTag() {
        return cntIgnoredByTag;
    }

    public void setCntIgnoredByTag(Integer cntIgnoredByTag) {
        this.cntIgnoredByTag = cntIgnoredByTag;
    }

    public Integer getCntIsNumber() {
        return cntIsNumber;
    }

    public void setCntIsNumber(Integer cntIsNumber) {
        this.cntIsNumber = cntIsNumber;
    }

    public Integer getCntUniqueValues() {
        return cntUniqueValues;
    }

    public void setCntUniqueValues(Integer cntUniqueValues) {
        this.cntUniqueValues = cntUniqueValues;
    }


    public Integer getCntIsNull() {
        return cntIsNull;
    }

    public void setCntIsNull(Integer cntIsNull) {
        this.cntIsNull = cntIsNull;
    }

    public Integer getCntIsNaN() {
        return cntIsNaN;
    }

    public void setCntIsNaN(Integer cntIsNaN) {
        this.cntIsNaN = cntIsNaN;
    }


}
