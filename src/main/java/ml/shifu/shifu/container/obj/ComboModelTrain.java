package ml.shifu.shifu.container.obj;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.collections.ListUtils;

import java.util.List;

/**
 * Created by zhanhu on 11/18/16.
 */
public class ComboModelTrain {

    private String uidColumnName;

    private List<ModelTrainConf> modelTrainConfList;

    private ModelTrainConf fusionModelTrainConf;

    public String getUidColumnName() {
        return uidColumnName;
    }

    public void setUidColumnName(String uidColumnName) {
        this.uidColumnName = uidColumnName;
    }

    public List<ModelTrainConf> getModelTrainConfList() {
        return modelTrainConfList;
    }

    public void setModelTrainConfList(List<ModelTrainConf> modelTrainConfList) {
        this.modelTrainConfList = modelTrainConfList;
    }

    public ModelTrainConf getFusionModelTrainConf() {
        return fusionModelTrainConf;
    }

    public void setFusionModelTrainConf(ModelTrainConf fusionModelTrainConf) {
        this.fusionModelTrainConf = fusionModelTrainConf;
    }

    @Override
    public boolean equals(Object obj) {
        if ( obj == null || !(obj instanceof  ComboModelTrain) ) {
            return false;
        }

        ComboModelTrain other = (ComboModelTrain) obj;
        if ( other == this ) {
            return true;
        }

        return uidColumnName.equals(other.getUidColumnName())
                && fusionModelTrainConf.equals(other.getFusionModelTrainConf())
                && compareTrainConfList(modelTrainConfList, other.getModelTrainConfList());
    }

    private boolean compareTrainConfList(List<ModelTrainConf> modelTrainConfList,
                                         List<ModelTrainConf> otherModelTrainConfList) {
        if ( modelTrainConfList == null && otherModelTrainConfList == null ) {
            return true;
        } else if ( (modelTrainConfList == null && otherModelTrainConfList != null)
                || (modelTrainConfList != null && otherModelTrainConfList == null)
                || modelTrainConfList.size() != otherModelTrainConfList.size() ) {
            return false;
        }

        for ( int i = 0; i < modelTrainConfList.size(); i ++ ) {
            if ( ! modelTrainConfList.get(i).equals(otherModelTrainConfList.get(i)) ) {
                return false;
            }
        }

        return true;
    }

}
