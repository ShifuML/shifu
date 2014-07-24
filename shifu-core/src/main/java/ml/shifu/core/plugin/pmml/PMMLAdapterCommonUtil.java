package ml.shifu.core.plugin.pmml;

import java.util.ArrayList;
import java.util.List;

import org.dmg.pmml.Constant;
import org.dmg.pmml.DataDictionary;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.FieldUsageType;
import org.dmg.pmml.LocalTransformations;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.NeuralInput;
import org.dmg.pmml.NeuralInputs;
import org.dmg.pmml.NeuralOutput;
import org.dmg.pmml.NeuralOutputs;
import org.dmg.pmml.NumericPredictor;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMML;
import org.dmg.pmml.RegressionModel;
import org.dmg.pmml.RegressionNormalizationMethodType;
import org.dmg.pmml.RegressionTable;


public class PMMLAdapterCommonUtil {



    public static NeuralInputs getNeuralInputs(MiningSchema schema) {
        NeuralInputs pmmlModel = new NeuralInputs();
        int index = 0;
        List<String> activeFields = getSchemaFieldViaUsageType(schema,
                FieldUsageType.ACTIVE);
        for (int i = 0; i < activeFields.size(); i++) {

            DerivedField field = new DerivedField(OpType.CONTINUOUS,
                    DataType.DOUBLE).withName(
                    new FieldName(activeFields.get(i))).withExpression(
                    new FieldRef(new FieldName(activeFields.get(i))));
            pmmlModel
                    .withNeuralInputs(new NeuralInput(field, "0," + (index++)));
        }
        DerivedField field = new DerivedField(OpType.CONTINUOUS,
                DataType.DOUBLE).withName(new FieldName(AdapterConstants.biasValue))
                .withExpression(
                        new FieldRef(new FieldName(AdapterConstants.biasValue)));
        pmmlModel.withNeuralInputs(new NeuralInput(field, AdapterConstants.biasValue));
        return pmmlModel;
    }

    public static LocalTransformations getBiasLocalTransformation(
            LocalTransformations lt) {

        DerivedField field = new DerivedField(OpType.CONTINUOUS,
                DataType.DOUBLE).withName(new FieldName(AdapterConstants.biasValue));
        // field.withName(new FieldName(s));
        field.withExpression(new Constant(String.valueOf(AdapterConstants.bias)));
        lt.withDerivedFields(field);
        return lt;
    }

    private static List<String> getSchemaFieldViaUsageType(MiningSchema schema,
            FieldUsageType type) {
        List<String> targetFields = new ArrayList<String>();

        for (MiningField f : schema.getMiningFields()) {
            FieldUsageType uType = f.getUsageType();
            if (uType == type)
                targetFields.add(f.getName().getValue());
        }
        return targetFields;
    }
    public static List<String> getSchemaTargetFields(MiningSchema schema ) {
        return getSchemaFieldViaUsageType(schema,FieldUsageType.TARGET);
    }
    public static List<String> getSchemaActiveFields(MiningSchema schema ) {
        return getSchemaFieldViaUsageType(schema,FieldUsageType.ACTIVE);
    }
    public static List<String> getSchemaSelectedFields(MiningSchema schema ) {
        List<String> targetFields = new ArrayList<String>();
        for (MiningField f : schema.getMiningFields()) {
            FieldUsageType uType = f.getUsageType();
            if (uType == FieldUsageType.TARGET || uType == FieldUsageType.ACTIVE)
                targetFields.add(f.getName().getValue());
        }
        return targetFields;
    }
    public static NeuralOutputs getOutputFields(MiningSchema schema, int layerID) {
        List<String> outputID = getSchemaFieldViaUsageType(schema,
                FieldUsageType.TARGET);
        NeuralOutputs outputs = new NeuralOutputs();
        int outputFieldsNum = outputID.size();
        outputs.setNumberOfOutputs(outputFieldsNum);
        for (int i = 0; i < outputFieldsNum; i++) {
            DerivedField field = new DerivedField(OpType.CONTINUOUS,
                    DataType.DOUBLE);
            field.withExpression(new FieldRef(new FieldName(outputID.get(i))));
            outputs.withNeuralOutputs(new NeuralOutput(field, String
                    .valueOf(layerID + "," + i)));
        }
        return outputs;
    }

    public static RegressionModel getRegressionTable(double[] weights,
            double intercept, RegressionModel pmmlModel) {
        RegressionTable table = new RegressionTable();
        MiningSchema schema = pmmlModel.getMiningSchema();
        // TODO may not need target field in LRModel
        pmmlModel.withFunctionName(MiningFunctionType.REGRESSION)
                .withNormalizationMethod(
                        RegressionNormalizationMethodType.LOGIT);
        List<String> outputFields = getSchemaFieldViaUsageType(schema, FieldUsageType.TARGET);
        // TODO only one outputField, what if we have more than one outputField
        pmmlModel.withTargetFieldName(new FieldName(outputFields.get(0)));
        table.withTargetCategory(outputFields.get(0));

        List<String> activeFields = getSchemaFieldViaUsageType(schema, FieldUsageType.ACTIVE);
        int index = 0;
        for (String s : activeFields) {
            table.withNumericPredictors(new NumericPredictor(new FieldName(s),
                    weights[index++]));
        }
        return pmmlModel;
    }
    

    public static String[] getDataDicHeaders(PMML pmml) {
        DataDictionary dictionary = pmml.getDataDictionary();
        List<DataField> fields = dictionary.getDataFields();
        int len = fields.size();
        String[] headers = new String[len];
        for (int i = 0; i < len; i++) {
            headers[i] = fields.get(i).getName().getValue();
        }
        return headers;
    }
}
