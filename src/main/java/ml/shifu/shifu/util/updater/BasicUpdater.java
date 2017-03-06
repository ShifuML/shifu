package ml.shifu.shifu.util.updater;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.validator.ModelInspector;
import ml.shifu.shifu.util.CommonUtils;
import org.apache.commons.collections.CollectionUtils;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

/**
 * Created by zhanhu on 2/22/17.
 */
public class BasicUpdater {

    protected String targetColumnName;
    protected Set<String> setCategorialColumns;
    protected Set<String> setMeta;
    protected Set<String> setForceRemove;
    protected Set<String> setForceSelect;

    public BasicUpdater(ModelConfig modelConfig) throws IOException {
        this.targetColumnName = CommonUtils.getRelativePigHeaderColumnName(modelConfig.getTargetColumnName());

        this.setCategorialColumns = new HashSet<String>();
        if(CollectionUtils.isNotEmpty(modelConfig.getCategoricalColumnNames())) {
            for(String column: modelConfig.getCategoricalColumnNames()) {
                setCategorialColumns.add(CommonUtils.getRelativePigHeaderColumnName(column));
            }
        }

        this.setMeta = new HashSet<String>();
        if(CollectionUtils.isNotEmpty(modelConfig.getMetaColumnNames())) {
            for(String meta: modelConfig.getMetaColumnNames()) {
                setMeta.add(CommonUtils.getRelativePigHeaderColumnName(meta));
            }
        }

        this.setForceRemove = new HashSet<String>();
        if(Boolean.TRUE.equals(modelConfig.getVarSelect().getForceEnable())
                && CollectionUtils.isNotEmpty(modelConfig.getListForceRemove())) {
            // if we need to update force remove, only and if one the force is enabled
            for(String forceRemoveName: modelConfig.getListForceRemove()) {
                setForceRemove.add(CommonUtils.getRelativePigHeaderColumnName(forceRemoveName));
            }
        }

        this.setForceSelect = new HashSet<String>(512);
        if(Boolean.TRUE.equals(modelConfig.getVarSelect().getForceEnable())
                && CollectionUtils.isNotEmpty(modelConfig.getListForceSelect())) {
            // if we need to update force select, only and if one the force is enabled
            for(String forceSelectName: modelConfig.getListForceSelect()) {
                setForceSelect.add(CommonUtils.getRelativePigHeaderColumnName(forceSelectName));
            }
        }
    }

    public void updateColumnConfig(ColumnConfig columnConfig) {
        String varName = columnConfig.getColumnName();

        if(this.targetColumnName.equals(varName)) {
            columnConfig.setColumnFlag(ColumnConfig.ColumnFlag.Target);
            columnConfig.setColumnType(null);
        } else if(this.setMeta.contains(varName)) {
            columnConfig.setColumnFlag(ColumnConfig.ColumnFlag.Meta);
            columnConfig.setColumnType(null);
        } else if(this.setForceRemove.contains(varName)) {
            columnConfig.setColumnFlag(ColumnConfig.ColumnFlag.ForceRemove);
        } else if(this.setForceSelect.contains(varName)) {
            columnConfig.setColumnFlag(ColumnConfig.ColumnFlag.ForceSelect);
        }

        // variable type is not related with variable flag
        if(this.setCategorialColumns.contains(varName)) {
            columnConfig.setColumnType(ColumnConfig.ColumnType.C);
        } else {
            columnConfig.setColumnType(ColumnConfig.ColumnType.N);
        }
    }

    public static BasicUpdater getUpdater(ModelConfig modelConfig, ModelInspector.ModelStep step) throws IOException {
        BasicUpdater updater = null;
        switch (step ) {
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
