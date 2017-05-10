package ml.shifu.shifu.util.updater;

import ml.shifu.shifu.column.NSColumn;
import ml.shifu.shifu.column.NSColumnUtils;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ColumnConfig.ColumnType;
import ml.shifu.shifu.core.validator.ModelInspector;
import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.lang.StringUtils;

import java.io.IOException;
import java.util.HashSet;
import java.util.List;
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
    protected Set<NSColumn> setCandidates;
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

        this.setCandidates = new HashSet<NSColumn>();
        List<String> candidates = modelConfig.getListCandidates();
        if (CollectionUtils.isNotEmpty(candidates)) {
            for ( String candidate : candidates ) {
                this.setCandidates.add(new NSColumn(candidate));
            }
        }
    }

    public void updateColumnConfig(ColumnConfig columnConfig) {
        String varName = columnConfig.getColumnName();

        // reset flag at first
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
            if ( CollectionUtils.isEmpty(this.setCandidates)
                    || (CollectionUtils.isNotEmpty(this.setCandidates) && this.setCandidates.contains(new NSColumn(varName))) ) {
                columnConfig.setColumnFlag(ColumnConfig.ColumnFlag.ForceSelect);
            }
        } else if(NSColumnUtils.isColumnEqual(this.weightColumnName, varName)) {
            columnConfig.setColumnFlag(ColumnConfig.ColumnFlag.Weight);
            columnConfig.setColumnType(null);
        }

        if(NSColumnUtils.isColumnEqual(weightColumnName, varName)) {
            // weight column is numerical
            columnConfig.setColumnType(ColumnType.N);
        } else if(NSColumnUtils.isColumnEqual(targetColumnName, varName)) {
            // target column is set to categorical column
            columnConfig.setColumnType(ColumnType.C);
        } else if(setCategorialColumns.contains(new NSColumn(varName))) {
            columnConfig.setColumnType(ColumnType.C);
        } else {
            // meta and other columns are set to numerical if user not set it in categorical column configuration file
            columnConfig.setColumnType(ColumnType.N);
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
