package ml.shifu.shifu.container.obj;

import org.apache.commons.collections.ListUtils;

import java.util.List;

/**
 * Created by zhanhu on 11/18/16.
 */
public class ComboModelTrain {

    private String uidColumnName;

    private List<VarTrainConf> varTrainConfList;

    private ModelTrainConf fusionModelTrainConf;

    public String getUidColumnName() {
        return uidColumnName;
    }

    public void setUidColumnName(String uidColumnName) {
        this.uidColumnName = uidColumnName;
    }

    public List<VarTrainConf> getVarTrainConfList() {
        return varTrainConfList;
    }

    public void setVarTrainConfList(List<VarTrainConf> varTrainConfList) {
        this.varTrainConfList = varTrainConfList;
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
                && ListUtils.isEqualList(varTrainConfList, other.getVarTrainConfList());
    }

}
