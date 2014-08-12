package ml.shifu.core.di.builtin.executor;

import org.dmg.pmml.FieldName;
import org.dmg.pmml.PMML;
import org.jpmml.evaluator.Evaluator;
import org.jpmml.evaluator.FieldValue;
import org.jpmml.evaluator.NeuralNetworkEvaluator;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class PMMLModelExecutor {

    private Evaluator evaluator;

    public PMMLModelExecutor(PMML pmml) {
        this.evaluator = new NeuralNetworkEvaluator(pmml);
    }

    public Object exec(Map<String, Object> rawDataMap) {

        Map<FieldName, FieldValue> arguments = new LinkedHashMap<FieldName, FieldValue>();

        List<FieldName> activeFields = evaluator.getActiveFields();
        for (FieldName activeField : activeFields) {
            Object rawValue = rawDataMap.get(activeField.getValue());
            FieldValue activeValue = evaluator.prepare(activeField, rawValue);

            arguments.put(activeField, activeValue);
        }

        Map<FieldName, ?> results = evaluator.evaluate(arguments);
        FieldName targetName = evaluator.getTargetField();

        return results.get(targetName);
    }

}
