package ml.shifu.plugin.pig;

import ml.shifu.core.di.builtin.executor.PMMLModelExecutor;
import ml.shifu.core.util.PMMLUtils;
import org.apache.pig.EvalFunc;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.dmg.pmml.FieldUsageType;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;

import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class PigModelExecUDF extends EvalFunc<Tuple> {

    private PMML pmml;
    private PMMLModelExecutor modelExecutor;
    private String[] header;

    public PigModelExecUDF(String pathPMML, String headerString) throws Exception {
        this.pmml = PMMLUtils.loadPMML(pathPMML);
        this.modelExecutor = new PMMLModelExecutor(pmml);
        this.header = headerString.split(",");
    }

    public Tuple exec(Tuple input) throws IOException {
        Tuple tuple = TupleFactory.getInstance().newTuple();

        if (header.length != input.size()) {
            throw new RuntimeException("Data Mismatch: header fields = " + header.length + ", data fields: " + input.size());
        }

        Map<String, Object> rawDataMap = new HashMap<String, Object>();

        for (int i = 0; i < header.length; i++) {
            rawDataMap.put(header[i], input.get(i).toString());
        }

        Object result = modelExecutor.exec(rawDataMap);
        tuple.append(Double.valueOf(result.toString()));

        Set<String> supplementaryFields = new HashSet<String>();
        String targetColumnName = "";
        for (Model model : pmml.getModels()) {
            for (MiningField miningField : model.getMiningSchema().getMiningFields()) {
                if (miningField.getUsageType().equals(FieldUsageType.SUPPLEMENTARY)) {
                    supplementaryFields.add(miningField.getName().getValue());
                }
                if(miningField.getUsageType().equals(FieldUsageType.TARGET)){
                	targetColumnName = miningField.getName().getValue();
                }
            }
        }

        tuple.append(rawDataMap.get(targetColumnName));
        
        for (String fieldName : supplementaryFields) {
            tuple.append(rawDataMap.get(fieldName));
        }

        return tuple;
    }
}
