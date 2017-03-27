package ml.shifu.shifu.core.model;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelTrainConf.ALGORITHM;
import org.encog.ml.BasicML;

import java.util.List;

/**
 * Created by zhanhu on 1/20/17.
 */
public class ModelSpec {
    private String modelName;
    private ModelConfig modelConfig;
    private List<ColumnConfig> columnConfigList;
    private ALGORITHM algorithm;
    private List<BasicML> models;

    public ModelSpec(String modelName,
                     ModelConfig modelConfig,
                     List<ColumnConfig> columnConfigList,
                     ALGORITHM algorithm, List<BasicML> models) {
        this.modelName = modelName;
        this.modelConfig = modelConfig;
        this.columnConfigList = columnConfigList;
        this.algorithm = algorithm;
        this.models = models;
    }

    public String getModelName() {
        return modelName;
    }

    public ALGORITHM getAlgorithm() {
        return algorithm;
    }

    public List<BasicML> getModels() {
        return models;
    }

    public ModelConfig getModelConfig() {
        return modelConfig;
    }

    public List<ColumnConfig> getColumnConfigList() {
        return columnConfigList;
    }
}
