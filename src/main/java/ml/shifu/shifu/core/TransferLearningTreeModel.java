package ml.shifu.shifu.core;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.encog.ml.BasicML;
import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;

import ml.shifu.shifu.column.NSColumn;
import ml.shifu.shifu.core.dtrain.dt.IndependentTreeModel;
import ml.shifu.shifu.util.CommonUtils;

public class TransferLearningTreeModel extends BasicML {

    private static final long serialVersionUID = -8269024520194949153L;
    
    /**
     * Tree model instance without dependency on encog.
     */
    private transient IndependentTreeModel independentTreeModel;
    
    private transient List<IndependentTreeModel> baseTreeModels;
    
    /**
     * 
     * @param independentTreeModel
     */
    public TransferLearningTreeModel(IndependentTreeModel independentTreeModel, List<IndependentTreeModel> baseTreeModels) {
        this.independentTreeModel = independentTreeModel;
        this.baseTreeModels = baseTreeModels;
    }

    public final MLData compute(Map<NSColumn, String> rawNsDataMap) {
        HashMap<String, Object> rawDataMap = new HashMap<String, Object>();
        for (Map.Entry<NSColumn, String> entry : rawNsDataMap.entrySet()) {
            rawDataMap.put(entry.getKey().getSimpleName(), entry.getValue());
        }
        
        double[] res = this.getIndependentTreeModel().compute(rawDataMap);
        
        for (IndependentTreeModel baseTreeModel : this.baseTreeModels) {
            res = CommonUtils.merge(res, baseTreeModel.compute(rawDataMap));
        }
        
        return new BasicMLData(res);
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (IndependentTreeModel baseTreeModel : baseTreeModels) {
            sb.append(baseTreeModel.getTrees().toString()).append("&");
        }
        sb.append(this.getIndependentTreeModel().getTrees().toString());
        
        return sb.toString();
    }
    
    public IndependentTreeModel getIndependentTreeModel() {
        return independentTreeModel;
    }
    
    @Override
    public void updateProperties() {
        // No need implementation
    }
}
