package ml.shifu.shifu.container.obj;

import org.apache.commons.collections.ListUtils;

import java.util.List;

/**
 * Created by zhanhu on 11/28/16.
 */
public class VarTrainConf {

    private List<String> variables;
    private ModelTrainConf modelTrainConf;

    public List<String> getVariables() {
        return variables;
    }

    public void setVariables(List<String> variables) {
        this.variables = variables;
    }

    public ModelTrainConf getModelTrainConf() {
        return modelTrainConf;
    }

    public void setModelTrainConf(ModelTrainConf modelTrainConf) {
        this.modelTrainConf = modelTrainConf;
    }

    @Override
    public boolean equals(Object obj) {
        if ( obj == null || !(obj instanceof VarTrainConf) ) {
            return false;
        }

        VarTrainConf other = (VarTrainConf) obj;
        if ( this == other ) {
            return true;
        }

        return ListUtils.isEqualList(this.variables, other.getVariables())
                && modelTrainConf.equals(other.getModelTrainConf());
    }
}
