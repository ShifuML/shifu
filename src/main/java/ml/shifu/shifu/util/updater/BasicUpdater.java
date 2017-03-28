package ml.shifu.shifu.util.updater;

import ml.shifu.shifu.column.NSColumn;
import ml.shifu.shifu.column.NSColumnUtils;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.validator.ModelInspector;
import org.apache.commons.collections.CollectionUtils;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

/**
 * Created by zhanhu on 2/22/17.
 */
public class BasicUpdater {

    protected String targetColumnName;
    protected Set<NSColumn> setCategorialColumns;
    protected Set<NSColumn> setMeta;
    protected Set<NSColumn> setForceRemove;
    protected Set<NSColumn> setForceSelect;
    protected String weightColumnName;

    public BasicUpdater(ModelConfig modelConfig) throws IOException {
        this.targetColumnName = modelConfig.getTargetColumnName();
        this.weightColumnName = modelConfig.getWeightColumnName();

        this.setCategorialColumns = new HashSet<NSColumn>();
        if(CollectionUtils.isNotEmpty(modelConfig.getCategoricalColumnNames())) {
            for(String column: modelConfig.getCategoricalColumnNames()) {
                setCategorialColumns.add(new NSColumn(column));
            }
        }

        this.setMeta = new HashSet<NSColumn>();
        if(CollectionUtils.isNotEmpty(modelConfig.getMetaColumnNames())) {
            for(String meta: modelConfig.getMetaColumnNames()) {
                setMeta.add(new NSColumn(meta));
            }
        }

        this.setForceRemove = new HashSet<NSColumn>();
        if(Boolean.TRUE.equals(modelConfig.getVarSelect().getForceEnable())
                && CollectionUtils.isNotEmpty(modelConfig.getListForceRemove())) {
            // if we need to update force remove, only and if one the force is enabled
            for(String forceRemoveName: modelConfig.getListForceRemove()) {
                setForceRemove.add(new NSColumn(forceRemoveName));
            }
        }

        this.setForceSelect = new HashSet<NSColumn>(512);
        if(Boolean.TRUE.equals(modelConfig.getVarSelect().getForceEnable())
                && CollectionUtils.isNotEmpty(modelConfig.getListForceSelect())) {
            // if we need to update force select, only and if one the force is enabled
            for(String forceSelectName: modelConfig.getListForceSelect()) {
                setForceSelect.add(new NSColumn(forceSelectName));
            }
        }
    }

    public void updateColumnConfig(ColumnConfig columnConfig) {
        String varName = columnConfig.getColumnName();

        columnConfig.setColumnFlag(null);
        if(NSColumnUtils.isColumnEqual(this.targetColumnName, varName)) {
            columnConfig.setColumnFlag(ColumnConfig.ColumnFlag.Target);
            columnConfig.setColumnType(null);
        } else if(this.setMeta.contains(new NSColumn(varName))) {
            columnConfig.setColumnFlag(ColumnConfig.ColumnFlag.Meta);
            columnConfig.setColumnType(null);
        } else if(this.setForceRemove.contains(new NSColumn(varName))) {
            columnConfig.setColumnFlag(ColumnConfig.ColumnFlag.ForceRemove);
        } else if(this.setForceSelect.contains(new NSColumn(varName))) {
            columnConfig.setColumnFlag(ColumnConfig.ColumnFlag.ForceSelect);
        } else if(NSColumnUtils.isColumnEqual(this.weightColumnName, varName)) {
            columnConfig.setColumnFlag(ColumnConfig.ColumnFlag.Weight);
            columnConfig.setColumnType(null);
        }

        // variable type is not related with variable flag
        if(this.setCategorialColumns.contains(new NSColumn(varName))) {
            columnConfig.setColumnType(ColumnConfig.ColumnType.C);
        } else {
            columnConfig.setColumnType(ColumnConfig.ColumnType.N);
        }
    }

    public static BasicUpdater getUpdater(ModelConfig modelConfig, ModelInspector.ModelStep step) throws IOException {
        BasicUpdater updater = null;
        switch(step) {
            case INIT:
            case STATS:
                updater = new BasicUpdater(modelConfig);
                break;
            case VARSELECT:
                updater = new VarSelUpdater(modelConfig);
                break;
            case TRAIN:
                updater = new TrainUpdater(modelConfig);
                break;
            default:
                updater = new VoidUpdater(modelConfig);
                break;
        }
        return updater;
    }
}
