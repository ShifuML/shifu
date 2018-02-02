package ml.shifu.shifu.core.history;

import ml.shifu.shifu.container.obj.ColumnConfig;
import org.apache.commons.lang.StringUtils;

/**
 * Created by zhanhu on 1/29/18.
 */
public class VarSelDesc {

    private int columnId = -1;
    private String columnName;
    private boolean oldSelStatus = true;
    private boolean newSelStatus = false;
    private VarSelReason reason;

    public VarSelDesc(ColumnConfig columnConfig, VarSelReason reason) {
        this.columnId = columnConfig.getColumnNum();
        this.columnName = columnConfig.getColumnName();
        this.reason = reason;
    }

    public VarSelDesc(int columnId, String columnName, boolean oldSelStatus, boolean newSelStatus, VarSelReason reason) {
        this.columnId = columnId;
        this.columnName = columnName;
        this.oldSelStatus = oldSelStatus;
        this.newSelStatus = newSelStatus;
        this.reason = reason;
    }

    public int getColumnId() {
        return columnId;
    }

    public void setColumnId(int columnId) {
        this.columnId = columnId;
    }

    public String getColumnName() {
        return columnName;
    }

    public void setColumnName(String columnName) {
        this.columnName = columnName;
    }

    public boolean getOldSelStatus() {
        return oldSelStatus;
    }

    public void setOldSelStatus(boolean oldSelStatus) {
        this.oldSelStatus = oldSelStatus;
    }

    public boolean getNewSelStatus() {
        return newSelStatus;
    }

    public void setNewSelStatus(boolean newSelStatus) {
        this.newSelStatus = newSelStatus;
    }

    public VarSelReason getReason() {
        return reason;
    }

    public void setReason(VarSelReason reason) {
        this.reason = reason;
    }

    @Override
    public String toString() {
        return columnId + "," + columnName + "," + this.oldSelStatus + "," + this.newSelStatus + "," + this.reason.name();
    }

    public static VarSelDesc fromString(String text) {
        VarSelDesc varSelDesc = null;
        String trimmedText = StringUtils.trimToEmpty(text);
        if ( StringUtils.isNotBlank(trimmedText) && !trimmedText.startsWith("#") ) {
            // skip empty line and lines start with '#'
            String[] fields = trimmedText.split(",");
            if ( fields != null && fields.length == 5 ) {
                int columnNum = Integer.parseInt(fields[0]);
                String columnName = fields[1];
                boolean oldSelStatus = Boolean.parseBoolean(fields[2]);
                boolean newSelStatus = Boolean.parseBoolean(fields[3]);
                VarSelReason reason = VarSelReason.valueOf(fields[4]);
                varSelDesc = new VarSelDesc(columnNum, columnName, oldSelStatus, newSelStatus, reason);
            }
        }
        return varSelDesc;
    }
}
