package ml.shifu.shifu.core.history;

/**
 * Created by zhanhu on 1/29/18.
 */
public enum VarSelReason {
    HIGH_MISSING_RATE("Column is with very high missing rate."),
    IV_TOO_LOW("IV of column is less than minimal IV threshold."),
    KS_TOO_LOW("KS of column is less than minimal KS threshold."),
    HIGH_CORRELATED("Absolute correlation value with some column is larger than "
            + "correlationThreshold value set in VarSelect#correlationThreshold.");

    private String desc;

    private VarSelReason(String desc) {
        this.desc = desc;
    }

    public String getDesc() {
        return desc;
    }

    public void setDesc(String desc) {
        this.desc = desc;
    }
}
