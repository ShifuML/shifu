package ml.shifu.core.di.builtin.transform;

import org.dmg.pmml.*;
import org.jpmml.evaluator.DiscretizationUtil;
import org.jpmml.evaluator.FieldValue;
import org.jpmml.evaluator.FieldValueUtil;
import org.jpmml.evaluator.NormalizationUtil;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class DefaultTransformationExecutor {

    public List<Object> transform(List<DerivedField> derivedFields, Map<String, Object> rawDataMap) {


        List<Object> result = new ArrayList<Object>();
        for (DerivedField derivedField : derivedFields) {
            result.add(transform(derivedField, rawDataMap));
        }

        return result;
    }

    public Object transform(DerivedField derivedField, Map<String, Object> rawDataMap) {
        Expression expression = derivedField.getExpression();

        //TODO: finish the list
        if (expression instanceof NormContinuous) {
            NormContinuous normContinuous = (NormContinuous) expression;
            Double value = Double.valueOf(rawDataMap.get(normContinuous.getField().getValue()).toString());
            return NormalizationUtil.normalize(normContinuous, value);
        } else if (expression instanceof Discretize) {
            Discretize discretize = (Discretize) expression;
            Double value = Double.valueOf(rawDataMap.get(discretize.getField().getValue()).toString());
            return DiscretizationUtil.discretize(discretize, value);
        } else if (expression instanceof MapValues) {
            MapValues mapValues = (MapValues) expression;


            //return ExpressionUtil.evaluate(expression, );
            return mapValue(mapValues, rawDataMap);
        } else if (expression instanceof FieldRef) {
            return rawDataMap.get(((FieldRef) expression).getField().getValue());
        } else {
            throw new RuntimeException("Invalid Expression(Field: " + derivedField.getName().getValue() + ")");
        }
    }

    /*
    public List<Object> transform(MiningSchema miningSchema, Map<FieldName, DerivedField> fieldNameToDerivedFieldMap, Map<FieldName, Integer> fieldNameToFieldNumberMap, List<Object> raw) {
        List<Object> transformed = new ArrayList<Object>();

        for (MiningField miningField : miningSchema.getMiningFields()) {

            int fieldNum = fieldNameToFieldNumberMap.get(miningField.getName());

            //if (miningField.getUsageType().equals(FieldUsageType.ACTIVE)) {
            if (fieldNameToDerivedFieldMap.containsKey(miningField.getName())) {
                DerivedField derivedField = fieldNameToDerivedFieldMap.get(miningField.getName());
                transformed.add(transform(derivedField, raw.get(fieldNum)));
            } else {
                transformed.add(raw.get(fieldNum));
            }
        }
        return transformed;
    }  */

    private String mapValue(MapValues mapValues, Map<String, Object> rawDataMap) {

        Map<String, FieldValue> values = new LinkedHashMap<String, FieldValue>();

        List<FieldColumnPair> fieldColumnPairs = mapValues.getFieldColumnPairs();
        for (FieldColumnPair fieldColumnPair : fieldColumnPairs) {
            FieldValue value = FieldValueUtil.create(mapValues.getDataType(), null, rawDataMap.get(fieldColumnPair.getField().getValue()));

            //if(value == null){
            //    return FieldValueUtil.create(mapValues.getDataType(), null, mapValues.getMapMissingTo());
            //}

            values.put(fieldColumnPair.getColumn(), value);
        }

        return DiscretizationUtil.mapValue(mapValues, values).getValue().toString();

    }


}
