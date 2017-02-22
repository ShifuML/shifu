/*
 * Copyright [2013-2016] PayPal Software Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.core.pmml;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.dmg.pmml.DataDictionary;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.FieldUsageType;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.NeuralOutput;
import org.dmg.pmml.NeuralOutputs;
import org.dmg.pmml.NormContinuous;
import org.dmg.pmml.NumericPredictor;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMML;
import org.dmg.pmml.RegressionModel;
import org.dmg.pmml.RegressionNormalizationMethodType;
import org.dmg.pmml.RegressionTable;

import com.google.common.primitives.Ints;

/**
 * 
 * This class contains common utilities that will be used in the shifu plugins.
 * 
 */
public class PMMLAdapterCommonUtil {

    private static List<String> getSchemaFieldViaUsageType(final MiningSchema schema, final FieldUsageType type) {
        List<String> targetFields = new ArrayList<String>();

        for(MiningField f: schema.getMiningFields()) {
            FieldUsageType uType = f.getUsageType();
            if(uType == type)
                targetFields.add(f.getName().getValue());
        }
        return targetFields;
    }

    /**
     * This function returns the target field names based on the given mining
     * schema
     * 
     * @param schema
     *            the schema
     * @return target field names
     */
    public static List<String> getSchemaTargetFields(final MiningSchema schema) {
        return getSchemaFieldViaUsageType(schema, FieldUsageType.TARGET);
    }

    /**
     * This function returns the active field names based on the given mining
     * schema
     * 
     * @param schema
     *            the schema
     * @return active field names
     */
    public static List<String> getSchemaActiveFields(final MiningSchema schema) {
        return getSchemaFieldViaUsageType(schema, FieldUsageType.ACTIVE);
    }

    /**
     * This function returns all used field names based on the given mining
     * schema
     * 
     * @param schema
     *            the schema
     * @return field names
     */
    public static List<String> getSchemaSelectedFields(final MiningSchema schema) {
        List<String> targetFields = new ArrayList<String>();
        for(MiningField f: schema.getMiningFields()) {
            FieldUsageType uType = f.getUsageType();
            if(uType == FieldUsageType.TARGET || uType == FieldUsageType.ACTIVE)
                targetFields.add(f.getName().getValue());
        }
        return targetFields;
    }

    /**
     * Create PMML neural output for the neural network models
     * 
     * @param schema
     *            the schema
     * @param layerID
     *            which layer the output neuron lies
     * @return neural outputs
     */
    public static NeuralOutputs getOutputFields(final MiningSchema schema, final int layerID) {
        List<String> outputID = getSchemaFieldViaUsageType(schema, FieldUsageType.TARGET);
        NeuralOutputs outputs = new NeuralOutputs();
        int outputFieldsNum = outputID.size();
        outputs.setNumberOfOutputs(outputFieldsNum);
        for(int i = 0; i < outputFieldsNum; i++) {
            DerivedField field = new DerivedField(OpType.CONTINUOUS, DataType.DOUBLE);
            field.withExpression(new FieldRef(new FieldName(outputID.get(i))));
            outputs.withNeuralOutputs(new NeuralOutput(field, String.valueOf(layerID + "," + i)));
        }
        return outputs;
    }

    /**
     * Generate Regression Table based on the weight list, intercept and partial
     * PMML model
     * 
     * @param weights
     *            weight list for the Regression Table
     * @param intercept
     *            the intercept
     * @param pmmlModel
     *            partial PMMl model
     * @return regression model instance
     */
    public static RegressionModel getRegressionTable(final double[] weights, final double intercept,
            RegressionModel pmmlModel) {
        RegressionTable table = new RegressionTable();
        MiningSchema schema = pmmlModel.getMiningSchema();
        // TODO may not need target field in LRModel
        pmmlModel.withFunctionName(MiningFunctionType.REGRESSION).withNormalizationMethod(
                RegressionNormalizationMethodType.LOGIT);
        List<String> outputFields = getSchemaFieldViaUsageType(schema, FieldUsageType.TARGET);
        // TODO only one outputField, what if we have more than one outputField
        pmmlModel.withTargetFieldName(new FieldName(outputFields.get(0)));
        table.withTargetCategory(outputFields.get(0));

        List<String> activeFields = getSchemaFieldViaUsageType(schema, FieldUsageType.ACTIVE);
        int index = 0;
        for(DerivedField dField: pmmlModel.getLocalTransformations().getDerivedFields()) {
            Expression expression = dField.getExpression();
            if(expression instanceof NormContinuous) {
                NormContinuous norm = (NormContinuous) expression;
                if(activeFields.contains(norm.getField().getValue()))
                    table.withNumericPredictors(new NumericPredictor(dField.getName(), weights[index++]));
            }

        }
        pmmlModel.withRegressionTables(table);
        return pmmlModel;
    }

    /**
     * get the header names from the PMML data dictionary
     * 
     * @param pmml
     *            the pmml model
     * @return headers
     */
    public static String[] getDataDicHeaders(final PMML pmml) {
        DataDictionary dictionary = pmml.getDataDictionary();
        List<DataField> fields = dictionary.getDataFields();
        int len = fields.size();
        String[] headers = new String[len];
        for(int i = 0; i < len; i++) {
            headers[i] = fields.get(i).getName().getValue();
        }
        return headers;
    }

    /**
     * get the column indexes for all active fields in the input data set
     * 
     * @param pmml
     *            the pmml model
     * @return active id
     */
    public static int[] getActiveID(PMML pmml) {
        return getDicFieldIDViaType(pmml, FieldUsageType.ACTIVE);
    }

    /**
     * get the column index for the target fields in the input data set
     * 
     * @param pmml
     *            the pmml model
     * @return target id
     */
    public static int[] getTargetID(PMML pmml) {
        return getDicFieldIDViaType(pmml, FieldUsageType.TARGET);
    }

    /**
     * Based on the usage type, get the column indexes for corresponding fields
     * in the input data set
     * 
     * @param pmml
     *            the pmml model
     * @param type
     *            the type
     * @return dic fields
     */
    public static int[] getDicFieldIDViaType(PMML pmml, FieldUsageType type) {
        List<Integer> activeFields = new ArrayList<Integer>();
        HashMap<String, Integer> dMap = new HashMap<String, Integer>();
        int index = 0;
        for(DataField dField: pmml.getDataDictionary().getDataFields())
            dMap.put(dField.getName().getValue(), index++);
        for(MiningField mField: pmml.getModels().get(0).getMiningSchema().getMiningFields()) {
            if(mField.getUsageType() == type)
                activeFields.add(dMap.get(mField.getName().getValue()));
        }

        return Ints.toArray(activeFields);
    }

}
