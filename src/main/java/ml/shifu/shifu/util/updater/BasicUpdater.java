package ml.shifu.shifu.util.updater;

import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import ml.shifu.shifu.column.NSColumn;
import ml.shifu.shifu.column.NSColumnUtils;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ColumnType;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.validator.ModelInspector;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.Environment;

import org.apache.commons.collections.CollectionUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by zhanhu on 2/22/17.
 */
public class BasicUpdater {

    @SuppressWarnings("unused")
    private final static Logger LOG = LoggerFactory.getLogger(BasicUpdater.class);

    protected ModelConfig modelConfig;
    protected List<ColumnConfig> columnConfigList;
    protected Map<String, ColumnConfig> columnConfigMap;

    protected String targetColumnName;
    protected String weightColumnName;

    protected Set<NSColumn> setCategoricalColumns;
    protected Set<NSColumn> setMeta;
    protected Set<NSColumn> setForceRemove;
    protected Set<NSColumn> setForceSelect;
    protected Set<NSColumn> setCandidates;

    protected Set<NSColumn> setHybridColumns;
    protected Map<String, Double> hybridColumnNames;
    protected Map<String, Integer> categoricalColumnHashSeeds;

    private int mtlIndex = -1;

    protected boolean isForSegs;
    protected List<String> segs;
    // The column amount in original data.
    protected int originalColumnAmount;

    public BasicUpdater(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, int mtlIndex) throws IOException {
        this.modelConfig = modelConfig;
        this.columnConfigList = columnConfigList;
        this.segs = modelConfig.getSegmentFilterExpressions();
        this.isForSegs = (this.segs.size() > 0);
        this.mtlIndex = mtlIndex;
        this.modelConfig.setMtlIndex(mtlIndex);
        initColumnConfigMapForSegs();

        this.targetColumnName = modelConfig.isMultiTask() ? modelConfig.getMultiTaskTargetColumnNames().get(mtlIndex)
                : modelConfig.getTargetColumnName();

        this.setMeta = loadNSColumns(modelConfig.getMetaColumnNames());
        this.setCategoricalColumns = loadNSColumns(modelConfig.getCategoricalColumnNames());
        this.categoricalColumnHashSeeds = modelConfig.getCategoricalColumnHashSeedConf();
        this.setHybridColumns = new HashSet<NSColumn>();
        this.hybridColumnNames = modelConfig.getHybridColumnNames();
        if(this.hybridColumnNames != null && this.hybridColumnNames.size() > 0) {
            for(Entry<String, Double> entry: this.hybridColumnNames.entrySet()) {
                this.setHybridColumns.add(new NSColumn(entry.getKey()));
            }
        }

        this.setForceRemove = new HashSet<NSColumn>();
        if(Boolean.TRUE.equals(modelConfig.getVarSelect().getForceEnable())
                && CollectionUtils.isNotEmpty(modelConfig.getListForceRemove())) {
            // if we need to update force remove, only and if one the force is enabled
            // this.setForceRemove = loadNSColumns(modelConfig.getListForceRemove());
            for(String forceRemoveName: modelConfig.getListForceRemove()) {
                this.setForceRemove.add(new NSColumn(forceRemoveName));
            }
        }

        this.setForceSelect = new HashSet<NSColumn>();
        if(Boolean.TRUE.equals(modelConfig.getVarSelect().getForceEnable())
                && CollectionUtils.isNotEmpty(modelConfig.getListForceSelect())) {
            // if we need to update force select, only and if one the force is enabled
            // this.setForceSelect = loadNSColumns(modelConfig.getListForceSelect());
            for(String forceSelectName: modelConfig.getListForceSelect()) {
                this.setForceSelect.add(new NSColumn(forceSelectName));
            }
        }

        this.setCandidates = loadNSColumns(modelConfig.getListCandidates());
    }

    protected Set<NSColumn> loadNSColumns(List<String> columnNames) {
        Set<NSColumn> nsColumns = new HashSet<>();
        if(CollectionUtils.isNotEmpty(columnNames)) {
            for(String column: columnNames) {
                nsColumns.add(new NSColumn(column));
                addSegColumns(nsColumns, column);
            }
        }
        return nsColumns;
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
            if(CollectionUtils.isEmpty(this.setCandidates) || (CollectionUtils.isNotEmpty(this.setCandidates) // candidates
                                                                                                              // is not
                                                                                                              // empty
                    && this.setCandidates.contains(new NSColumn(varName)))) {
                columnConfig.setColumnFlag(ColumnConfig.ColumnFlag.ForceSelect);
            }
        } else if(this.setCandidates.contains(new NSColumn(varName))) {
            columnConfig.setColumnFlag(ColumnConfig.ColumnFlag.Candidate);
        }

        if(NSColumnUtils.isColumnEqual(targetColumnName, varName)) {
            List<String> tags = this.modelConfig.isMultiTask() ? this.modelConfig.getMTLTags(mtlIndex)
                    : this.modelConfig.getTags();
            if(CollectionUtils.isEmpty(tags)) {
                // allow tags are empty to support linear target
                // set columnType to N
                columnConfig.setColumnType(ColumnType.N);
            } else {
                // target column is set to categorical column
                columnConfig.setColumnType(ColumnType.C);
            }
        } else if(setHybridColumns.contains(new NSColumn(varName))) {
            columnConfig.setColumnType(ColumnType.H);
            String newVarName = null;
            if(Environment.getBoolean(Constants.SHIFU_NAMESPACE_STRICT_MODE, false)) {
                newVarName = new NSColumn(varName).getFullColumnName();
            } else {
                newVarName = new NSColumn(varName).getSimpleName();
            }
            columnConfig.setHybridThreshold(hybridColumnNames.get(newVarName));
        } else if(setCategoricalColumns.contains(new NSColumn(varName))) {
            columnConfig.setColumnType(ColumnType.C);
        } else {
            // meta and other columns are set to numerical if user not set it in categorical column configuration file
            columnConfig.setColumnType(ColumnType.N);
        }
        if(this.categoricalColumnHashSeeds != null && this.categoricalColumnHashSeeds.containsKey(varName)) {
            columnConfig.setHashSeed(this.categoricalColumnHashSeeds.get(varName));
        }
    }

    public static BasicUpdater getUpdater(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, ModelInspector.ModelStep step, int mtlIndex)
            throws IOException {
        BasicUpdater updater = null;
        switch(step) {
            case INIT:
            case STATS:
            case NORMALIZE:
                updater = new BasicUpdater(modelConfig, columnConfigList, mtlIndex);
                break;
            case VARSELECT:
                updater = new VarSelUpdater(modelConfig, columnConfigList, mtlIndex);
                break;
            case TRAIN:
                updater = new TrainUpdater(modelConfig, columnConfigList, mtlIndex);
                break;
            default:
                updater = new VoidUpdater(modelConfig, columnConfigList, mtlIndex);
                break;
        }
        return updater;
    }

    /**
     * @return the mtlIndex
     */
    public int getMtlIndex() {
        return mtlIndex;
    }

    /**
     * @param mtlIndex
     *            the mtlIndex to set
     */
    public void setMtlIndex(int mtlIndex) {
        this.mtlIndex = mtlIndex;
    }

    /**
     * Init the column config map if segment exists.
     */
    private void initColumnConfigMapForSegs() {
        if (isForSegs) {
            originalColumnAmount = 0;
            columnConfigMap = new HashMap<>(columnConfigList.size());
            for (ColumnConfig config : columnConfigList) {
                columnConfigMap.put(config.getColumnName(), config);
                if (config.isSegment() != null && !config.isSegment()) {
                    originalColumnAmount++;
                }
            }
            LOG.info("Init column config map in updater, totalColumnAmount={}, originalColumnAmount={}, segAmount={}.",
                columnConfigList.size(), originalColumnAmount, segs.size());
        }
    }

    /**
     * Add segment column names to the set.
     *
     * @param segColumnSet is the set which will hold segment column names.
     * @param columnName   is the original column name (non-segment column).
     */
    private void addSegColumns(Set<NSColumn> segColumnSet, String columnName) {
        // Only do it when segment exists.
        if (this.isForSegs) {
            ColumnConfig originalColumnConfig = columnConfigMap.get(columnName);
            if (originalColumnConfig == null) {
                return;
            }
            for (int i = 0; i < segs.size(); i++) {
                // Calculate the segment's column config number(index), and put it into the set.
                int segColumnConfigNum = originalColumnConfig.getColumnNum() + (i + 1) * originalColumnAmount;
                if (segColumnConfigNum >= columnConfigList.size()) {
                    break;
                }
                ColumnConfig segColumnConfig = columnConfigList.get(segColumnConfigNum);
                segColumnSet.add(new NSColumn(segColumnConfig.getColumnName()));
                LOG.debug("Add {}({})'s segment {}({}).", columnName, originalColumnConfig.getColumnNum(), segColumnConfig.getColumnName(),
                    segColumnConfigNum);
            }
        }
    }
}
