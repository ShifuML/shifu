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

import java.util.HashMap;
import java.util.List;

import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Discretize;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.LocalTransformations;
import org.dmg.pmml.MapValues;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.MiningField.UsageType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.NormContinuous;
import org.dmg.pmml.regression.NumericPredictor;
import org.dmg.pmml.regression.RegressionModel;
import org.dmg.pmml.regression.RegressionModel.NormalizationMethod;
import org.dmg.pmml.regression.RegressionTable;

/**
 * The class that converts an LR model to a PMML RegressionModel.
 * This class extends the abstract class
 * PMMLModelBuilder(pmml.RegressionModel,LR).
 */
public class PMMLLRModelBuilder implements PMMLModelBuilder<RegressionModel, ml.shifu.shifu.core.LR> {

    public RegressionModel adaptMLModelToPMML(ml.shifu.shifu.core.LR lr,
            RegressionModel pmmlModel) {
        pmmlModel.setNormalizationMethod(NormalizationMethod.LOGIT);
        pmmlModel.setMiningFunction(MiningFunction.REGRESSION);
        RegressionTable table = new RegressionTable();
        table.setIntercept(lr.getBias());
        LocalTransformations lt = pmmlModel.getLocalTransformations();
        List<DerivedField> df = lt.getDerivedFields();

        HashMap<FieldName, FieldName> miningTransformMap = new HashMap<FieldName, FieldName>();
        for(DerivedField dField: df) {
            // Apply z-scale normalization on numerical variables
            if(dField.getExpression() instanceof NormContinuous) {
                miningTransformMap.put(((NormContinuous) dField.getExpression()).getField(), dField.getName());
            }
            // Apply bin map on categorical variables
            else if(dField.getExpression() instanceof MapValues) {
                miningTransformMap.put(((MapValues) dField.getExpression()).getFieldColumnPairs().get(0).getField(),
                        dField.getName());
            } else if(dField.getExpression() instanceof Discretize) {
                miningTransformMap.put(((Discretize) dField.getExpression()).getField(), dField.getName());
            }
        }

        List<MiningField> miningList = pmmlModel.getMiningSchema().getMiningFields();
        int index = 0;
        for(int i = 0; i < miningList.size(); i++) {
            MiningField mField = miningList.get(i);
            if(mField.getUsageType() != UsageType.ACTIVE)
                continue;
            FieldName mFieldName = mField.getName();
            FieldName fName = mFieldName;
            while(miningTransformMap.containsKey(fName)) {
                fName = miningTransformMap.get(fName);
            }

            NumericPredictor np = new NumericPredictor();
            np.setName(fName);
            np.setCoefficient(lr.getWeights()[index++]);
            table.addNumericPredictors(np);
        }

        pmmlModel.addRegressionTables(table);
        return pmmlModel;
    }
}
