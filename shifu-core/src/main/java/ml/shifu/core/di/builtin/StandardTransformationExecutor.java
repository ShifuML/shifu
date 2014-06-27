package ml.shifu.core.di.builtin;

import ml.shifu.core.di.spi.TransformationExecutor;
import ml.shifu.core.util.CommonUtils;
import org.dmg.pmml.*;
import org.jpmml.evaluator.DiscretizationUtil;
import org.jpmml.evaluator.NormalizationUtil;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class StandardTransformationExecutor implements TransformationExecutor {

    public Object transform(DerivedField derivedField, Object origin) {

        Expression expression = derivedField.getExpression();

        //TODO: finish the list
        if (expression instanceof NormContinuous) {
            return NormalizationUtil.normalize((NormContinuous) expression, Double.parseDouble(origin.toString()));
        } else if (expression instanceof Discretize) {
            return DiscretizationUtil.discretize((Discretize) expression, Double.parseDouble(origin.toString()));
        } else if (expression instanceof MapValues) {
            return mapValue((MapValues) expression, origin.toString());
        } else {
            throw new RuntimeException("Invalid Expression(Field: " + derivedField.getName().getValue() + ")");
        }

    }

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
    }

    private String mapValue(MapValues mapValues, String origin) {
        InlineTable inlineTable = mapValues.getInlineTable();
        List<Row> rows = inlineTable.getRows();

        Map<String, String> valueMap = new HashMap<String, String>();
        for (Row row : rows) {
            String[] raw = CommonUtils.stringToStringList(row.getContent().toString()).get(0).split(" ");
            if (raw[0].equals(origin)) {
                return raw[1];
            }
        }

        // TODO: deal with missing values
        throw new RuntimeException("Unknown value: " + origin);
    }


}
