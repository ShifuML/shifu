package ml.shifu.shifu.di.builtin;

import ml.shifu.shifu.di.spi.TransformationExecutor;
import ml.shifu.shifu.util.PMMLUtils;
import org.dmg.pmml.*;
import org.jpmml.evaluator.NormalizationUtil;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class StandardTransformationExecutor implements TransformationExecutor {

    public  Object transform(DerivedField derivedField, Object origin) {

        Expression expression = derivedField.getExpression();

        if (expression instanceof NormContinuous) {
            return NormalizationUtil.normalize((NormContinuous) expression, Double.parseDouble(origin.toString()));
        } else {
            throw new RuntimeException("Invalid Expression");
        }

    }

    public List<Object> transform(MiningSchema miningSchema, Map<FieldName, DerivedField> fieldNameToDerivedFieldMap, Map<FieldName, Integer> fieldNameToFieldNumberMap,  List<Object> raw) {
        List<Object> transformed = new ArrayList<Object>();

        for (MiningField miningField : miningSchema.getMiningFields()) {

            int fieldNum = fieldNameToFieldNumberMap.get(miningField.getName());

            if (miningField.getUsageType().equals(FieldUsageType.ACTIVE)) {
                DerivedField derivedField = fieldNameToDerivedFieldMap.get(miningField.getName());
                transformed.add(transform(derivedField, raw.get(fieldNum)));
            } else {
                transformed.add(raw.get(fieldNum));
            }
        }
        return transformed;
    }


}
