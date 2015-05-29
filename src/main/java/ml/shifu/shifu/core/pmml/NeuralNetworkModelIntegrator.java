/**
 * Copyright [2012-2014] PayPal Software Foundation
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

import org.dmg.pmml.*;

import java.util.HashMap;
import java.util.List;

/**
 * This class glues the partial PMML neural network model with the neural layers
 * part, by adding bias field and neural input layer.
 */
public class NeuralNetworkModelIntegrator {

    /**
     * Given the partial neural network model, return the neural network model
     * by adding bias neuron and neural input layer.
     *
     * @param model
     * @return
     */
    public NeuralNetwork adaptPMML(NeuralNetwork model) {
        model.withNeuralInputs(getNeuralInputs(model));
        model.setLocalTransformations(getLocalTranformations(model));
        return model;
    }

    private NeuralInputs getNeuralInputs(final NeuralNetwork model) {
        NeuralInputs nnInputs = new NeuralInputs();
        // get HashMap for local transform and MiningSchema fields
        HashMap<FieldName, FieldName> miningTransformMap = new HashMap<FieldName, FieldName>();
        for (DerivedField dField : model.getLocalTransformations()
                .getDerivedFields()) {
            // Apply z-scale normalization on numerical variables
            if (dField.getExpression() instanceof NormContinuous) {
                miningTransformMap.put(
                        ((NormContinuous) dField.getExpression()).getField(),
                        dField.getName());
            }
            // Apply bin map on categorical variables
            else if (dField.getExpression() instanceof MapValues) {
                miningTransformMap.put(
                        ((MapValues) dField.getExpression()).getFieldColumnPairs().get(0).getField(),
                        dField.getName());
            }
        }
        List<MiningField> miningList = model.getMiningSchema()
                .getMiningFields();
        int index = 0;
        for (int i = 0; i < miningList.size(); i++) {
            MiningField mField = miningList.get(i);
            if (mField.getUsageType() != FieldUsageType.ACTIVE)
                continue;
            FieldName mFieldName = mField.getName();
            FieldName fName = (miningTransformMap.containsKey(mFieldName)) ? miningTransformMap
                    .get(mFieldName) : mFieldName;
            DerivedField field = new DerivedField(OpType.CONTINUOUS,
                    DataType.DOUBLE).withName(fName).withExpression(
                    new FieldRef(fName));
            nnInputs.withNeuralInputs(new NeuralInput(field, "0," + (index++)));
        }

        DerivedField field = new DerivedField(OpType.CONTINUOUS,
                DataType.DOUBLE).withName(
                new FieldName(PluginConstants.biasValue)).withExpression(
                new FieldRef(new FieldName(PluginConstants.biasValue)));
        nnInputs.withNeuralInputs(new NeuralInput(field,
                PluginConstants.biasValue));
        return nnInputs;
    }

    private LocalTransformations getLocalTranformations(NeuralNetwork model) {
        // delete target
        List<DerivedField> derivedFields = model.getLocalTransformations()
                .getDerivedFields();

        // add bias
        DerivedField field = new DerivedField(OpType.CONTINUOUS,
                DataType.DOUBLE).withName(new FieldName(
                PluginConstants.biasValue));
        field.withExpression(new Constant(String.valueOf(PluginConstants.bias)));
        derivedFields.add(field);
        return new LocalTransformations().withDerivedFields(derivedFields);
    }

}
