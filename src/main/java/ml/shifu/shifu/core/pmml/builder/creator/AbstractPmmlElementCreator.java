package ml.shifu.shifu.core.pmml.builder.creator;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import org.dmg.pmml.DataType;
import org.dmg.pmml.OpType;

import java.util.List;

/**
 * Created by zhanhu on 3/29/16.
 */
public abstract class AbstractPmmlElementCreator<T> {

    protected boolean isConcise;
    protected ModelConfig modelConfig;
    protected List<ColumnConfig> columnConfigList;

    public AbstractPmmlElementCreator(ModelConfig modelConfig, List<ColumnConfig> columnConfigList) {
        this(modelConfig, columnConfigList, false);
    }

    public AbstractPmmlElementCreator(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, boolean isConcise) {
        this.modelConfig = modelConfig;
        this.columnConfigList = columnConfigList;
        this.isConcise = isConcise;
    }

    public abstract T build();

    /**
     * Get @OpType from ColumnConfig
     * Meta Column -> ORDINAL
     * Target Column -> CATEGORICAL
     * Categorical Column -> CATEGORICAL
     * Numerical Column -> CONTINUOUS
     *
     * @param columnConfig
     *            - ColumnConfig for variable
     * @return OpType
     */
    protected OpType getOptype(ColumnConfig columnConfig) {
        if(columnConfig.isMeta()) {
            return OpType.ORDINAL;
        } else if(columnConfig.isTarget()) {
            return OpType.CATEGORICAL;
        } else {
            return (columnConfig.isCategorical() ? OpType.CATEGORICAL : OpType.CONTINUOUS);
        }
    }

    /**
     * Get DataType from OpType
     * CONTINUOUS -> DOUBLE
     * Other -> STRING
     *
     * @param optype
     *            OpType
     * @return DataType
     */
    protected DataType getDataType(OpType optype) {
        return (optype.equals(OpType.CONTINUOUS) ? DataType.DOUBLE : DataType.STRING);
    }
}
