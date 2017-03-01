package ml.shifu.shifu.core.model;

import ml.shifu.shifu.container.obj.ModelTrainConf.*;
import org.encog.ml.BasicML;

import java.util.List;

/**
 * Created by zhanhu on 1/20/17.
 */
public class ModelSpec {
    private String modelName;
    private ALGORITHM algorithm;
    private List<BasicML> models;

    public ModelSpec(String modelName, ALGORITHM algorithm, List<BasicML> models) {
        this.modelName = modelName;
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

}
